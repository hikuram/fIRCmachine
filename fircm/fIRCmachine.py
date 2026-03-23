import os
import sys
import shutil
import argparse
from datetime import datetime
from time import perf_counter as timepfc
from typing import List

# Third-party
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

# Project modules
import default_config as g
from instant_plot import instant_plot
from dmf import DirectMaxFlux, interpolate_fbenm
from sella import Sella, Constraints
from sella_ext_AdaptiveIRC import AdaptiveIRC
from pyscf_exporter import export_pyscf_single_point

# --- Separated Modules ---
from ase_calculators import make_calculator
from traj_utils import extract_peaks_from_traj, traj_to_xyz, write_energies, \
    split_traj_to_xyz, select_highest_peak_file

# Overwrite global variables
#g.INIT_PATH_SEARCH_ON = False
# Example settings are described in README or default_config.py.

# FB-ENM/DMF optimization (first stage)
def run_initial_path_search():
    reactant = read("reactant.xyz")
    product = read("product.xyz")
    
    # == Refine input geometries ===================
    t_opt_start = timepfc()
    if g.REFINE_INPUT_ON:
        if g.USE_SELLA_IN_OPT:
            reactant = opt_sella_img("reactant.xyz")
            product = opt_sella_img("product.xyz")
        else:
            reactant = opt_img("reactant.xyz")
            product = opt_img("product.xyz")
    t_opt = timepfc() - t_opt_start
    txt = f"* Optimize_Total        | {t_opt:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)
    
    # == Run DMF ===================
    
    t_dmf_start = timepfc()
    mepopt_dmf(reactant, product)
    t_dmf = timepfc() - t_dmf_start
    txt = f"* FB-ENM/DMF_Total      | {t_dmf:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)

# Repeat for each local maximum
def process_local_maxima():
    df_new = pd.read_csv(g.R_CSV)
    # Detect and save local maxima
    peak_files = []
    peak_files, g.PEAK_IDX = extract_peaks_from_traj(g.I_TRAJ, "lmax.xyz", prominence=0.01)

    # Write CSV (accepts a pair of elements or lists)
    def write_result(column_name, value):
        if not isinstance(column_name, list):
            column_name = [column_name]
            value = [value]
        for i, cn in enumerate(column_name):
            df_new.at[df_new.index[idx], column_name[i]] = value[i]
        try:
            df_new.to_csv(g.R_CSV, index=False)
        except Exception as e:
            print(f"Warning: An error occurred while writing {g.R_CSV}: {e}")

    # Sub-iteration 1: ignore endpoints
    irc_trajs_str = ""
    t_tsopt_irc_start = timepfc()
    for i, peak_file in enumerate(peak_files):
        if len(peak_files) > 2:
            if i == 0 or i == len(peak_files) - 1:
                continue

        base_name = os.path.splitext(peak_file)[0]
        idx = int(base_name.split('_')[-1].split('.')[0])  # index of local maximum
        atoms = read(peak_file)
        atoms.info["charge"] = g.CHARGE
        atoms.info["spin"] = g.MULT

        # == Run TS optimization ===================
        if g.TSOPT_ON:
            t_tsopt_start = timepfc()
            try:
                tsopt_img(base_name + ".xyz")
            except Exception as e:
                print(f"Warning: TSopt failed: {e}")
            t_tsopt = timepfc() - t_tsopt_start
            write_result('time_TSopt [s]', t_tsopt)
            print(f"tsopt {t_tsopt} sec")

        # == Run IRC ===================
        if g.IRC_ON:
            target_xyz = base_name + "_tsopt.xyz"
            
            if not os.path.exists(target_xyz):
                print(f"Warning: Skipping IRC for {base_name} (Missing TS structure).")
                write_result(['time_IRC [s]', 'deltaE_irc0 [kcal/mol]', 'deltaE_irc1 [kcal/mol]'], [None, None, None])
            else:
                t_irc_start = timepfc()
                try:
                    irc_result = irc_img(target_xyz)
                    
                    if os.path.exists(base_name + "_tsopt_irc0/irc.traj"):
                        write_energies(base_name + "_tsopt_irc0/irc.traj")
                        irc_trajs_str += f" {g.CURRENT_DIR}/{base_name}_tsopt_irc0/irc.traj"
                    if os.path.exists(base_name + "_tsopt_irc1/irc.traj"):
                        write_energies(base_name + "_tsopt_irc1/irc.traj")
                        irc_trajs_str += f" {g.CURRENT_DIR}/{base_name}_tsopt_irc1/irc.traj"
                        
                except Exception as e:
                    print(f"Warning: IRC failed for {base_name}: {e}")
                    irc_result = [None, None]
                    
                t_irc = timepfc() - t_irc_start
                write_result(
                    ['time_IRC [s]', 'deltaE_irc0 [kcal/mol]', 'deltaE_irc1 [kcal/mol]'],
                    [t_irc] + irc_result
                )
                print(f"irc {t_irc} sec")
        # ==

    t_tsopt_irc = timepfc() - t_tsopt_irc_start
    txt = f"* TSopt/IRC_Total       | {t_tsopt_irc:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)
    if irc_trajs_str.strip():
        g.SUGGESTIONS.append(f"python3 cattraj.py -i{irc_trajs_str} -o {g.CURRENT_DIR}/irc_cat/irc_cat.traj")

    # Optional workflow: pick representative optimized points for thermochemistry.
    vib_files = peak_files
    if g.PICK_OPTPOINTS_ON:
        g.ORIG_R_CSV = g.R_CSV
        vib_files, opt_indices = make_optpoints_traj(peak_files)
        optpoints_csv = "optpoints/result_optpoints.csv"
        write_energies("optpoints/optpoints.traj", csv_name=optpoints_csv, previous_image=opt_indices)
        df_new = pd.read_csv(optpoints_csv)
        g.R_CSV = optpoints_csv

    # Sub-iteration 2: include endpoints or reduced representative points
    t_vib_sum = 0
    t_refine_sum = 0
    for i, peak_file in enumerate(vib_files):
        base_name = os.path.splitext(peak_file)[0]
        idx = i if g.PICK_OPTPOINTS_ON else int(base_name.split('_')[-1].split('.')[0])
        atoms = read(peak_file)
        atoms.info["charge"] = g.CHARGE
        atoms.info["spin"] = g.MULT

        # == Vibrations and IdealGasThermo ===================
        if g.VIB_ON:
            t_vib_start = timepfc()
            try:
                vib_result = vib_img(base_name + ".xyz")
            except Exception as e:
                print(f"Warning: Vibrations failed: {e}")
                vib_result = [None, None, None, None]
            t_vib = timepfc() - t_vib_start
            t_vib_sum += t_vib
            write_result(
                ['time_vib [s]', 'ZPE [kcal/mol]', 'E_0K [kcal/mol]', 'H [kcal/mol]', 'G [kcal/mol]'],
                [t_vib] + vib_result
            )
            print(f"vibrations {t_vib} sec")

        # == Refinement ===================
        if g.REFINE_ENERGY_ON:
            t_refine_start = timepfc()
            try:
                refine_result = refine_energy_img(base_name + ".xyz", refine_type=g.REFINE_CALC_TYPE)
                energy_ref_eV, energy_ref_kcal = refine_result
            except Exception as e:
                print(f"Warning: refinement failed: {e}")
                energy_ref_eV = None
                energy_ref_kcal = None

            t_refine = timepfc() - t_refine_start
            t_refine_sum += t_refine
            write_result(
                ['time_refine [s]', 'energy_refine [eV]', 'energy_refine [kcal/mol]'],
                [t_refine, energy_ref_eV, energy_ref_kcal]
            )

            if (
                'G [kcal/mol]' in df_new.columns
                and 'energy [kcal/mol]' in df_new.columns
                and pd.notna(df_new.at[df_new.index[idx], 'G [kcal/mol]'])
                and pd.notna(df_new.at[df_new.index[idx], 'energy [kcal/mol]'])
                and energy_ref_kcal is not None
            ):
                thermal_corr_G = (
                    df_new.at[df_new.index[idx], 'G [kcal/mol]']
                    - df_new.at[df_new.index[idx], 'energy [kcal/mol]']
                )
                G_refine_kcal = energy_ref_kcal + thermal_corr_G
                write_result('G_refine [kcal/mol] (HL//LL)', G_refine_kcal)

            print(f"refinement {t_refine} sec")
        # ==

    txt = f"* Vibrations_Total      | {t_vib_sum:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)
    txt = f"* Refinement_Total      | {t_refine_sum:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)


# Run MEP optimization with FB-ENM/DMF
def mepopt_dmf(reactant_atoms: Atoms, product_atoms: Atoms) -> None:
    # Read reactant and product
    ref_images = [reactant_atoms, product_atoms]
    
    # Generate initial path using FB-ENM
    quiet_stdout = {"print_level": 0, "file_print_level": 5}
    mxflx_fbenm = interpolate_fbenm(ref_images, correlated=True, ipopt_options=quiet_stdout)
    write('DMF_init.xyz', mxflx_fbenm.images)
    
    # Write initial path and its coefficients
    write('DMF_init.traj', mxflx_fbenm.images)
    coefs = mxflx_fbenm.coefs.copy()
    np.save('DMF_init_coefs', coefs)
    
    # Set up and solve Direct MaxFlux
    mxflx = DirectMaxFlux(ref_images, coefs=coefs, nmove=g.NMOVE, update_teval=g.UPDATE_TEVAL)
    # Set up calculator
    for img in mxflx.images:
        img.info["charge"] = g.CHARGE
        img.info["spin"] = g.MULT
        img.calc = make_calculator(g.CALC_TYPE, img, "DMF_init")
    # Solve
    mxflx.add_ipopt_options({'output_file': 'DMF_ipopt.out', "print_level": 0, "file_print_level": 5})
    try:
        mxflx.solve(tol=g.DMF_CONVERGENCE)
    except Exception as e:
        write("DMF_last_before_error.xyz", mxflx.images)
        write("DMF_last_before_error.traj", mxflx.images)
        sys.exit(f"abort: DirectMaxFlux.solve failed: {e}")
    
    # DMF_final.traj: Recompute SPC for mxflx.images (some frames lack energy)
    final_images = []
    for img in mxflx.images:
        # Copy atoms and info
        atoms = Atoms(positions=img.get_positions(), numbers=img.get_atomic_numbers())
        atoms.info["charge"] = g.CHARGE
        atoms.info["spin"] = g.MULT
        atoms.calc = make_calculator(g.CALC_TYPE, atoms, "DMF_final")
        try:
            # Explicitly calculate energy
            _ = atoms.get_potential_energy()
        except Exception as e:
            print(f"Warning: failed to compute energy for image {len(final_images)}: {e}")
        final_images.append(atoms)
    
    # x(tmax): path and history
    images_tmax = mxflx.history.images_tmax
    write('DMF_tmax.traj', images_tmax)
    traj_to_xyz(images_tmax, 'DMF_tmax.xyz')
    # final_images: save images to .traj
    write('DMF_final.traj', final_images)
    traj_to_xyz(final_images, 'DMF_final.xyz')
    # Write results
    write_energies('DMF_final.traj', g.R_CSV)
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/DMF_final.traj")

# Write text file
def write_line(txtfile_name, txt):
    with open(txtfile_name, 'a', encoding='utf-8') as f:
        f.write(txt)

# Run optimization with ASE
def opt_img(xyz_name: str) -> Atoms:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info["charge"] = g.CHARGE
    img.info["spin"] = g.MULT
    img.calc = make_calculator(g.CALC_TYPE, img, img_name)
    # Set up an ASE optimizer (L-BFGS)
    opt = LBFGS(img, trajectory=img_name+"_opt.traj", logfile=img_name+"_opt.log")
    opt.run(fmax=0.01, steps=10000)
    write(img_name+"_opt.xyz", img)
    images = read(img_name+"_opt.traj", index=':')
    traj_to_xyz(images, img_name+"_opt.traj.xyz")
    
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{img_name}_opt.traj")
    return img


# Run optimization with Sella
def opt_sella_img(xyz_name: str) -> Atoms:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info["charge"] = g.CHARGE
    img.info["spin"] = g.MULT
    img.calc = make_calculator(g.CALC_TYPE, img, img_name)
    # Set up a Sella Dynamics object (order=0)
    dyn = Sella(
        img, internal=g.SELLA_INTERNAL, order=0, constraints=None,
        trajectory=img_name+'_opt.traj', logfile=img_name+"_opt.log"
    )
    dyn.run(fmax=4e-4, steps=1000)
    write(img_name+"_opt.xyz", img)
    images = read(img_name+"_opt.traj", index=':')
    traj_to_xyz(images, img_name+"_opt.traj.xyz")
    
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{img_name}_opt.traj")
    return img


# Run TS optimization with Sella
def tsopt_img(xyz_name: str) -> Atoms:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info["charge"] = g.CHARGE
    img.info["spin"] = g.MULT
    img.calc = make_calculator(g.CALC_TYPE, img, img_name)
    if g.SELLA_INTERNAL_AUTO:
        # Check the symmetry of the initial structure
        _, _, g.SELLA_INTERNAL = get_symmetry_info(img, tol=1e-3)
    # Set up a Sella Dynamics object
    dyn = Sella(
        img, internal=g.SELLA_INTERNAL, order=1, constraints=None,
        trajectory=img_name+'_tsopt.traj', logfile=img_name+"_tsopt.log"
    )
    dyn.run(fmax=4e-4, steps=1000)
    write(img_name+"_tsopt.xyz", img)
    images = read(img_name+"_tsopt.traj", index=':')
    traj_to_xyz(images, img_name+"_tsopt.traj.xyz")
    
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{img_name}_tsopt.traj")
    return img


# Run IRC with Sella
def irc_img(xyz_name: str) -> List[float]:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info["charge"] = g.CHARGE
    img.info["spin"] = g.MULT
    img.calc = make_calculator(g.CALC_TYPE, img, img_name)
    # Set up a Sella IRC object
    opt = AdaptiveIRC(
        img, trajectory=img_name+'_irc.traj', logfile=img_name+"_irc.log",
        dx=g.IRC_DX_INIT, max_dx=g.IRC_DX_MAX, min_dx=g.IRC_DX_MIN,
        eta=1e-4, gamma=0.4
    )
    opt.run(fmax=1e-2, steps=1000, direction='forward')
    write(img_name+"_forward.xyz", img)
    hoge = read(img_name+"_irc.traj", index=':')[::-1]
    
    opt.run(fmax=1e-2, steps=1000, direction='reverse')
    write(img_name+"_reverse.xyz", img)
    fuga = read(img_name+"_irc.traj", index=':')[len(hoge):]
    
    rearr_images = hoge + fuga
    tgt_dir = img_name+"_irc0"
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
    write(tgt_dir+"/irc.traj", rearr_images)
    traj_to_xyz(rearr_images, tgt_dir+"/irc.traj.xyz")
    
    rearr_images.reverse()
    tgt_dir = img_name+"_irc1"
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
    write(tgt_dir+"/irc.traj", rearr_images)
    traj_to_xyz(rearr_images, tgt_dir+"/irc.traj.xyz")
    
    rearr_energies = []
    for rimg in rearr_images:
        rearr_energies.append(rimg.get_potential_energy())
    deltaE_irc0 = g.EV_TO_KCAL_MOL * (max(rearr_energies) - rearr_energies[-1])
    deltaE_irc1 = g.EV_TO_KCAL_MOL * (max(rearr_energies) - rearr_energies[0])
    
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{img_name}_irc0/irc.traj")
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{img_name}_irc1/irc.traj")
    g.SUGGESTIONS.append(f"python3 sVIBmachine.py -d {g.CURRENT_DIR}/{img_name}_irc0 -c {g.CHARGE} -i {g.CURRENT_DIR}/{img_name}_irc0/irc.traj")
    g.SUGGESTIONS.append(f"python3 sVIBmachine.py -d {g.CURRENT_DIR}/{img_name}_irc1 -c {g.CHARGE} -i {g.CURRENT_DIR}/{img_name}_irc1/irc.traj")
    
    return [deltaE_irc0, deltaE_irc1]


def refine_energy_img(xyz_name, refine_type="pyscf_high"):
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info["charge"] = g.CHARGE
    img.info["spin"] = g.MULT

    img.calc = make_calculator(refine_type, img, img_name + "_refine")
    energy_eV = img.get_potential_energy()
    energy_kcal = energy_eV * g.EV_TO_KCAL_MOL
    
    try:
        export_pyscf_single_point(img, prefix=img_name+"_refine")
    except Exception as e:
        print(f"Warning: export_pyscf_single_point failed: {e}")
    
    img.calc = None
    return [energy_eV, energy_kcal]

# 
def get_symmetry_info(atoms, tol=1e-3):
    """
    Analyze the point group of the molecule using PySCF and return
    the geometry type ('linear'/'nonlinear') and symmetry number (sigma)
    required for ASE's IdealGasThermo.
    """
    import re
    from pyscf import gto, symm
    
    if len(atoms) == 1:
        return 'monatomic', 1, True
    orig_tol = symm.geom.TOLERANCE
    symm.geom.TOLERANCE = tol
    
    try:
        # Convert ASE Atoms to PySCF atom list format
        atom_list = [[atom.symbol, atom.position] for atom in atoms]
        
        # Build a lightweight PySCF Mole object to detect symmetry
        mol = gto.Mole()
        mol.atom = atom_list
        mol.charge = g.CHARGE
        mol.spin = g.MULT - 1
        mol.basis = {'default': [[0, (1.0, 1.0)]]} # Dummy basis just to allow build() to pass
        mol.symmetry = True
        mol.verbose = 0       # Suppress PySCF output
        mol.build()
        
        pg = mol.topgroup
    except Exception as e:
        print(f"Warning: Failed to determine symmetry with PySCF ({e}). Falling back to nonlinear, sigma=1.")
        return 'nonlinear', 1, True
    finally:
        symm.geom.TOLERANCE = orig_tol
        
    # Determine if the molecule is linear
    linear_groups = ['Cinfv', 'Dinfh', 'Coov', 'Dooh']
    geometry = 'linear' if pg in linear_groups else 'nonlinear'
    
    # Calculate symmetry number from the Point Group symbol
    sym_num = 1
    if pg in ['Cinfv', 'Coov']:
        sym_num = 1
    elif pg in ['Dinfh', 'Dooh']:
        sym_num = 2
    elif pg in ['T', 'Td', 'Th']:
        sym_num = 12
    elif pg in ['O', 'Oh']:
        sym_num = 24
    elif pg in ['I', 'Ih']:
        sym_num = 60
    else:
        # Parse C_n, D_n, S_n groups (e.g., "C3v" -> letter="C", n=3)
        m = re.search(r'^([CDS])(\d+)', pg)
        if m:
            letter = m.group(1)
            n = int(m.group(2))
            if letter == 'C':
                sym_num = n
            elif letter == 'D':
                sym_num = 2 * n
            elif letter == 'S':
                sym_num = n // 2

    # Determine whether to use internal coordinates.
    # Internal coordinates mathematically fail for linear molecules.
    # High-symmetry planar/spherical groups (e.g., D3h, Oh) can also cause ODE solver singularities.
    # Cs, C2v, etc., are perfectly safe for internal coordinates.
    risky_point_groups = ['D3h', 'D4h', 'D6h', 'Td', 'Oh', 'Ih', 'C3v']
    internal_safe = True
    if geometry == 'linear':
        internal_safe = False
    elif pg in risky_point_groups:
        internal_safe = False
    print(
        f"  [coords] Detected Point Group: {pg} -> "
        f"geometry='{geometry}', symmetrynumber={sym_num}, "
        f"internal_safe={internal_safe}"
    )
    return geometry, sym_num, internal_safe


def generate_vibration_xyz(atoms, vib, mode_index, output, steps=10, scale=1.0):
    freqs = vib.get_frequencies()
    natoms = len(atoms)
    numbers = atoms.get_atomic_numbers()
    modes = [vib.get_mode(i) for i in range(len(freqs))]

    """
    Generate an .xyz animation of vibration along the selected mode.
    Parameters:
    - atoms: ASE Atoms object (original geometry)
    - vib: ASE Vibrations object
    - mode_index: Index of vibration mode to animate (default: 0)
    - steps: Number of steps for half cycle (default: 10)
    - scale: Scaling factor for mode displacement (default: 1.0)
    """
    mode = vib.get_mode(mode_index)  # (N_atoms, 3) displacement vectors
    mode = np.array(mode)
    original_positions = atoms.get_positions()
    images = []

    def generate_half_cycle(sign):
        for i in range(steps):
            factor = sign * (i + 1) / steps
            displaced = original_positions + factor * scale * mode
            new_atoms = atoms.copy()
            new_atoms.set_positions(displaced)
            images.append(new_atoms.copy())

        for i in range(steps):
            factor = sign * (steps - i - 1) / steps
            displaced = original_positions + factor * scale * mode
            new_atoms = atoms.copy()
            new_atoms.set_positions(displaced)
            images.append(new_atoms.copy())

    generate_half_cycle(+1)  # +mode -> original
    generate_half_cycle(-1)  # -mode -> original
    write(output, images)
    print(f"[Info] Wrote {len(images)} frames to {output}")

# Run vibrations and thermodynamics
def vib_img(xyz_name):
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info["charge"] = g.CHARGE
    img.info["spin"] = g.MULT
    img.calc = make_calculator(g.CALC_TYPE, img, img_name)
    #forces = img.get_forces()
    electronic_energy = img.get_potential_energy()
    vib = Vibrations(img, name="vib_temp")
    vib.run()
    vib.summary(log=img_name+'_vibsummary.txt')
    vib.get_frequencies()
    #generate_vibration_xyz(atoms, vib, 0, steps, scale, vib_filename)
    for mode in range(0, 3):
        vib_filename = f"{img_name}_vib_{mode}.xyz"
        generate_vibration_xyz(img, vib, mode, output=vib_filename)
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{img_name}_vib_*.xyz")

    # Ideal-gas limit
    # Use ignore_imag_modes=True
    vib_energies = vib.get_energies()
    # Dynamically obtain symmetry and geometry via PySCF
    geom_type, sym_num, _ = get_symmetry_info(img, tol=1e-3)
    
    thermo = IdealGasThermo(
        vib_energies=vib_energies, potentialenergy=electronic_energy,
        atoms=img, geometry=geom_type, symmetrynumber=sym_num, spin=(g.MULT-1)/2,
        ignore_imag_modes=True
    )
    energy_eV = electronic_energy
    zpe_eV = vib.get_zero_point_energy()  # Units: eV
    H_eV = thermo.get_enthalpy(temperature=298.15)
    G_eV = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.0)
        # Room temperature=298.15 # Standard atmosphere=101325.0
    
    zpe_kcal = g.EV_TO_KCAL_MOL * zpe_eV
    E_0K_kcal = g.EV_TO_KCAL_MOL * (zpe_eV + energy_eV)
    H_kcal = g.EV_TO_KCAL_MOL * H_eV
    G_kcal = g.EV_TO_KCAL_MOL * G_eV
    vib.clean()
    
    return [zpe_kcal, E_0K_kcal, H_kcal, G_kcal]

def make_optpoints_traj(peak_files: List[str], out_traj: str = "optpoints/optpoints.traj") -> List[str]:
    """
    Build a reduced 3-point trajectory (start, highest TS-like peak, end) 
    for downstream VIB/refinement jobs.
    """

    out_dir = os.path.dirname(out_traj)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if len(peak_files) < 2:
        raise ValueError("peak_files must contain at least the two endpoints")

    start_file = peak_files[0]
    end_file = peak_files[-1]
    middle_file = select_highest_peak_file(peak_files)

    branch_plan = [(start_file, 0)]
    if middle_file is not None:
        branch_plan.append((middle_file, int(os.path.splitext(middle_file)[0].split('_')[-1].split('.')[0])))
    branch_plan.append((end_file, int(os.path.splitext(end_file)[0].split('_')[-1].split('.')[0])))
    branch_indices = [prev_idx for _, prev_idx in branch_plan]
    
    branch_images = []
    for src_file, previous_idx in branch_plan:
        use_file = src_file
        if src_file == middle_file:
            base_name = os.path.splitext(middle_file)[0]
            tsopt_file = base_name + "_tsopt.xyz"
            if g.TSOPT_ON and os.path.exists(tsopt_file):
                use_file = tsopt_file
        
        # Optimize again if requested
        if g.OPT_OPTPOINTS_AGAIN_ON:
            if src_file == middle_file:
                atoms = tsopt_img(use_file)
            else:
                if g.USE_SELLA_IN_OPT:
                    atoms = opt_sella_img(use_file)
                else:
                    atoms = opt_img(use_file)
        else :
            atoms = read(use_file)
            atoms.info["charge"] = g.CHARGE
            atoms.info["spin"] = g.MULT
        
        branch_images.append(atoms)

    write(out_traj, branch_images)
    traj_to_xyz(branch_images, out_traj + ".xyz")

    xyz_prefix = os.path.splitext(out_traj)[0]
    branch_xyz_files = split_traj_to_xyz(out_traj, xyz_prefix)

    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{out_traj}")
    g.SUGGESTIONS.append(
        f"python3 sVIBmachine.py -d {g.CURRENT_DIR}/optpoints "
        f"-c {g.CHARGE} -m {g.CALC_TYPE} -i {g.CURRENT_DIR}/{out_traj}"
    )

    return branch_xyz_files, branch_indices
    

# Finishing steps
def finalize_run():
    csv_targets = []
    if g.PICK_OPTPOINTS_ON and hasattr(g, 'ORIG_R_CSV'):
        csv_targets.append((g.ORIG_R_CSV, g.PEAK_IDX))
        csv_targets.append((g.R_CSV, None))
    else:
        csv_targets.append((g.R_CSV, g.PEAK_IDX))

    for csv_file, peak_idx in csv_targets:
        if not os.path.exists(csv_file):
            continue

        # Write relative energy (kcal/mol)
        df = pd.read_csv(csv_file)
        if g.VIB_ON:
            try:
                if df["E_0K [kcal/mol]"].notna().any():
                    df["Delta E_0K vs. reactant [kcal/mol]"] = df["E_0K [kcal/mol]"] - df.loc[0, "E_0K [kcal/mol]"]
                if df["H [kcal/mol]"].notna().any():
                    df["Delta H vs. reactant [kcal/mol]"] = df["H [kcal/mol]"] - df.loc[0, "H [kcal/mol]"]
                if df["G [kcal/mol]"].notna().any():
                    df["Delta G vs. reactant [kcal/mol]"] = df["G [kcal/mol]"] - df.loc[0, "G [kcal/mol]"]
                if "G_refine [kcal/mol] (HL//LL)" in df.columns and df["G_refine [kcal/mol] (HL//LL)"].notna().any():
                    df["Delta G_refine vs. reactant [kcal/mol] (HL//LL)"] = df["G_refine [kcal/mol] (HL//LL)"] - df.loc[0, "G_refine [kcal/mol] (HL//LL)"]
                df.to_csv(csv_file, index=False)
            except Exception as e:
                print(f"Warning: An error occurred while writing {csv_file}: {e}")
        
        # plot
        if g.SAVE_FIG_ON:
            figname = f"fig_{os.path.splitext(os.path.basename(csv_file))[0]}.png"
            instant_plot(df, peak_idx, figname)
    
    # Suggest next steps
    if g.WRITE_SUGGESTIONS_ON and len(g.SUGGESTIONS)>0:
        print("(suggestion) your next steps may be ...")
        with open("suggestions.txt", "a", encoding='utf-8') as f:
            for elem in g.SUGGESTIONS:
                print(elem)
                f.write(f"{elem}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full IRC jobs starting with reactant.xyz and product.xyz')
    parser.add_argument("-d", "--directory", type=str, required=True, help="path to the destination folder")
    parser.add_argument("-c", "--charge", type=int, required=True, help="system total charge")
    parser.add_argument("-m", "--method", type=str, required=False, default="orbmol", help="calculation method of the PES")
    if g.INIT_PATH_SEARCH_ON:
        parser.add_argument("-r", "--reactant", type=str, required=True, help="inputfile for the reactant .xyz file")
        parser.add_argument("-p", "--product", type=str, required=True, help="inputfile for the product .xyz file")
    else:
        parser.add_argument("-i", "--input", type=str, required=True, default="input.traj", help="input .traj or .xyz file")
    parser.add_argument("-rs", "--result", type=str, required=False, default="result.csv", help="resulting dataframe .csv file")
    args = parser.parse_args()
    
    t_total_start = timepfc()
    if g.INIT_PATH_SEARCH_ON:
        if not os.path.exists(args.directory):
            os.makedirs(args.directory, exist_ok=True)
        else:
            print(f"canceled: {args.directory} already exists")
            sys.exit()
        shutil.copy(args.reactant, args.directory)
        shutil.copy(args.reactant, args.directory+"/reactant.xyz")
        shutil.copy(args.product, args.directory)
        shutil.copy(args.product, args.directory+"/product.xyz")
    else:
        if not os.path.exists(args.input):
            print(f"canceled: cannot load {args.input}")
            sys.exit()
        if not os.path.exists(args.directory):
            os.makedirs(args.directory, exist_ok=True)
        input_name = os.path.basename(args.input)
        if not os.path.exists(args.directory+"/"+input_name):
            shutil.copy(args.input, args.directory)
        g.I_TRAJ = input_name
    os.chdir(args.directory)
    g.CURRENT_DIR = args.directory
    g.CHARGE = args.charge
    g.CALC_TYPE = args.method
    g.R_CSV = args.result
    if os.path.exists(g.R_CSV):
        print(f"info: {g.R_CSV} will be overwritten")
    else:
        print(f"info: {g.R_CSV} will be made")
    
    # Main
    if g.INIT_PATH_SEARCH_ON:
        run_initial_path_search()
        g.I_TRAJ = "DMF_final.traj" # ignores args.input
    elif not g.PRESERVE_CSV_ON:
        if g.INIT_RECALC_MODE_ON:
            #Ignore the file's energy, strictly recalculate
            write_energies(g.I_TRAJ, g.R_CSV, energy_recalc=True)
        else:
            write_energies(g.I_TRAJ, g.R_CSV)
    process_local_maxima()
    
    # Finish
    finalize_run()
    t_total = timepfc() - t_total_start
    txt = f"* Total_Time            | {t_total:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)
    print(f"finished at: {datetime.now()}")
