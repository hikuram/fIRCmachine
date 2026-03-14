import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from time import perf_counter as timepfc

# Third-party
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from scipy.signal import find_peaks
import cupy

# Project modules
import default_config as g
from instant_plot import instant_plot
from dmf import DirectMaxFlux, interpolate_fbenm
from sella import Sella, Constraints
from sella_ext_AdaptiveIRC import AdaptiveIRC

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
            if i == 0 or i == len(peak_files)-1:
                continue
        
        base_name = os.path.splitext(peak_file)[0]
        idx = int(base_name.split('_')[-1].split('.')[0]) # index of local maximum
        atoms = read(peak_file)
        atoms.info["charge"] = g.CHARGE
        atoms.info["spin"] = g.MULT
        
        # == Run TS optimization ===================
        if g.TSOPT_ON:
            t_tsopt_start = timepfc()
            try:
                tsopt_img(base_name+".xyz")
            except Exception as e:
                print(f"Warning: TSopt failed: {e}")
            t_tsopt = timepfc() - t_tsopt_start
            write_result('time_TSopt [s]', t_tsopt)
            print(f"tsopt {t_tsopt} sec")
        
        # == Run IRC ===================
        if g.IRC_ON:
            t_irc_start = timepfc()
            try:
                irc_result = irc_img(base_name+"_tsopt.xyz")
            except Exception as e:
                print(f"Warning: IRC failed: {e}")
            t_irc = timepfc() - t_irc_start
            write_result(
                ['time_IRC [s]', 'deltaE_irc0 [kcal/mol]', 'deltaE_irc1 [kcal/mol]'],
                [t_irc]+irc_result
            )
            print(f"irc {t_irc} sec")
            
            write_energies(base_name+"_tsopt_irc0/irc.traj")
            write_energies(base_name+"_tsopt_irc1/irc.traj")
            irc_trajs_str += f" {g.CURRENT_DIR}/{base_name}_tsopt_irc0/irc.traj"
        # ==
        
    t_tsopt_irc = timepfc() - t_tsopt_irc_start
    txt = f"* TSopt/IRC_Total       | {t_tsopt_irc:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)
    g.SUGGESTIONS.append(f"python3 cattraj.py -i{irc_trajs_str} -o {g.CURRENT_DIR}/irc_cat/irc_cat.traj")
    
    # Sub-iteration 2: include endpoints
    t_vib_sum = 0
    t_refine_sum = 0
    for i, peak_file in enumerate(peak_files):
        base_name = os.path.splitext(peak_file)[0]
        idx = int(base_name.split('_')[-1].split('.')[0]) # index of local maximum
        atoms = read(peak_file)
        atoms.info["charge"] = g.CHARGE
        atoms.info["spin"] = g.MULT
        
        # == Vibrations and IdealGasThermo ===================
        if g.VIB_ON:
            t_vib_start = timepfc()
            try:
                vib_result = vib_img(base_name+".xyz")
            except Exception as e:
                print(f"Warning: Vibrations failed: {e}")
            t_vib = timepfc() - t_vib_start
            t_vib_sum += t_vib
            write_result(
                ['time_vib [s]', 'ZPE [kcal/mol]', 'E_0K [kcal/mol]', 'H [kcal/mol]', 'G [kcal/mol]'],
                [t_vib]+vib_result
            )
            print(f"vibrations {t_vib} sec")
        
        # == Other jobs ===================
        if g.OTHER_JOBS_EXAMPLE_ON:
            from ase.geometry import get_distances
            _, iadist = get_distances(atoms.positions, atoms.positions[0])
            print(iadist[1][0])
            write_result('distance(1, 0) [angs]', iadist[1][0])
            
#            from morfeus import SASA
#            sasa = SASA(atoms.symbols, atoms.positions, probe_radius=1.4)
#            print(sasa.atom_areas[1])
#            write_result('SASA [angs^2]', sasa.atom_areas[1])
#            
#            from morfeus import BiteAngle
#            ba = BiteAngle(atoms.positions, BA_idx[0], BA_idx[1], BA_idx[2])
#            print(ba.angle)
#            write_result('SASA [angs^2]', sasa.atom_areas[1])
#            
#            from morfeus import SolidAngle
#            sa = SolidAngle(atoms.symbols, atoms.positions, 1)
#            print(sa.solid_angle)
#            print(sa.G)
#            write_result(['Solid angle [sr]', 'G parameter [%]'], [sa.solid_angle, sa.G])
#            
#            import rmsd
#            ref = atoms[0]
#            tgt = atoms[1]
#            min_rmsd = rmsd.kabsch_rmsd(ref, tgt)
#            print(min_rmsd)
#            write_result('RMSD [angs]', min_rmsd)
        # ==
        
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
    txt = f"* Refinement_Total      | {t_refine_sum:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)

# PySCF config cache
_PYSCF_CONFIG_CACHE = None
_PYSCF_PROFILE_CACHE = {}

def load_pyscf_config():
    global _PYSCF_CONFIG_CACHE
    if _PYSCF_CONFIG_CACHE is None:
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyscf_config.json")
        config_path = getattr(g, "PYSCF_CONFIG_FILE", default_path)
        with open(config_path, "r", encoding="utf-8") as f:
            _PYSCF_CONFIG_CACHE = json.load(f)
    return _PYSCF_CONFIG_CACHE

def get_pyscf_profile(calc_type):
    global _PYSCF_PROFILE_CACHE
    if calc_type in _PYSCF_PROFILE_CACHE:
        return _PYSCF_PROFILE_CACHE[calc_type]

    config = load_pyscf_config()
    if calc_type not in config:
        raise KeyError(f"Missing PySCF profile in config: {calc_type}")

    profile = dict(config[calc_type])
    profile["calc_type"] = calc_type
    profile["is_3c"] = str(profile.get("xc", "")).endswith("3c")
    _PYSCF_PROFILE_CACHE[calc_type] = profile
    return profile

def build_pyscf_method_common(atoms, base_name, profile):
    from pyscf import M
    from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf

    mol = M(
        atom=ase_atoms_to_pyscf(atoms),
        basis=profile.get("basis"),
        ecp=profile.get("ecp"),
        charge=g.CHARGE,
        spin=g.MULT - 1,
        output=base_name + '_pyscf.log',
        verbose=profile.get("verbose", 4)
    )
    mf = mol.RKS(
        xc=profile["xc"],
        disp=profile.get("disp"),
        conv_tol=profile.get("conv_tol", 6e-10),
        max_cycle=profile.get("max_cycle", 400)
    )

    if profile.get("with_solvent", False):
        solvent_model = str(profile.get("solvent_model", "")).upper()
        if solvent_model == "SMD":
            mf = mf.SMD()
        else:
            raise NotImplementedError(f"Unsupported solvent model: {solvent_model}")
        mf.with_solvent.solvent = profile.get("solvent", "water")
        if profile.get("eps") is not None:
            mf.with_solvent.eps = profile["eps"]

    mf.grids.level = profile.get("grids_level", 5)
    if hasattr(mf, "nlcgrids") and profile.get("nlcgrids_level") is not None:
        mf.nlcgrids.level = profile.get("nlcgrids_level")

    if g.DEVICE == "cuda":
        cupy.get_default_memory_pool().free_all_blocks()
        mf = mf.to_gpu()
    return mf

def build_pyscf_standard(atoms, base_name, profile):
    from gpu4pyscf.tools.ase_interface import PySCF

    mf = build_pyscf_method_common(atoms, base_name, profile)
    return PySCF(method=mf)

def build_pyscf_3c(atoms, base_name, profile):
    from redox.utils.pyscf_utils import PySCFCalculator, build_3c_method

    config = {}
    config["xc"] = profile["xc"]
    config["charge"] = g.CHARGE
    config["spin"] = g.MULT - 1
    config["verbose"] = profile.get("verbose", 4)
    config["output"] = base_name + '_pyscf.log'
    config["inputfile"] = [
        (ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    ]
    if profile.get("with_solvent", False):
        config["with_solvent"] = True
        config["solvent"] = {
            "method": profile.get("solvent_model", "SMD"),
            "eps": profile.get("eps", 78.3553),
            "solvent": profile.get("solvent", "water")
        }

    if not str(config["xc"]).endswith("3c"):
        raise NotImplementedError("When a 3c profile is specified, the xc string must end with '3c'.")

    mf = build_3c_method(config)
    return PySCFCalculator(mf, xc_3c=profile["xc"])

# Set calculator
def make_calculator(type, atoms, base_name):
    # PySCF
    if type in ["pyscf", "pyscf_high"]:
        profile = get_pyscf_profile(type)
        if profile["is_3c"]:
            calculator = build_pyscf_3c(atoms, base_name, profile)
        else:
            calculator = build_pyscf_standard(atoms, base_name, profile)

    # orbmol
    elif type == "orbmol":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v3_conservative_omol(
            device=g.DEVICE,
            precision="float64",   # "float32"/ "float32-highest"/ "float64"
        )
        calculator = ORBCalculator(orbff, device=g.DEVICE)

    # orbmol+alpb
    elif type == "orbmol+alpb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        from ase.calculators.mixing import LinearCombinationCalculator
        from dual_tblite_delta import DualTBLite
        orbff = pretrained.orb_v3_conservative_omol(
            device=g.DEVICE,
            precision="float64",   # "float32"/ "float32-highest"/ "float64"
        )
        solvation = ("alpb", "water")
        acc = 0.02
        calc_mlip =  ORBCalculator(orbff, device=g.DEVICE)
        calc_delta = DualTBLite(method="GFN1-xTB", charge=g.CHARGE, multiplicity=g.MULT, solvation=solvation, accuracy=acc, verbosity=0)
        calculator = LinearCombinationCalculator([calc_mlip, calc_delta], [1, 1])

    else:
        sys.exit("error: incorrect calc type")
    return calculator


# Parse input trajectory
from typing import List

def extract_peaks_from_traj(trajfile: str, maxima_filename: str, prominence: float = 0.01) -> List[str]:
    # Load all frames from trajectory
    traj = read(trajfile, index=':')
    energies = []
    for i, atoms in enumerate(traj):
        try:
            energy = atoms.get_potential_energy()
        except Exception as e:
            print(f"Warning: missing value for {trajfile} atom {i}: {e}")
            energy = np.nan
        energies.append(energy)
    energies = np.array(energies)

    # Fill NaN
    def forward_fill_nan(arr):
        filled = arr.copy()
        last_valid = np.nan
        for i in range(len(filled)):
            if not np.isnan(filled[i]):
                last_valid = filled[i]
            else:
                filled[i] = last_valid
        return filled
    energies_filled = forward_fill_nan(energies)

    # peak detection
    peaks, _ = find_peaks(energies_filled, prominence=prominence)
    # Input basename (e.g., input.xyz -> input)
    base_name = os.path.splitext(os.path.basename(maxima_filename))[0]
    print(f"Detected {len(peaks)} peak(s). Saving structures:")
    peak_files = []
    
    # Add first and last frames
    endpoints = np.array([0, len(traj) - 1])
    g.PEAK_IDX = np.unique(np.concatenate([peaks, endpoints]))

    for idx in g.PEAK_IDX:
        atoms = traj[idx]
        filename = f"{base_name}_{idx}.xyz"
        peak_files.append(filename)
        write(filename, atoms)
        print(f"  → {filename} (energy = {energies[idx]:.6f})")

    return peak_files, g.PEAK_IDX


# Run MEP optimization with FB-ENM/DMF
def mepopt_dmf(reactant_atoms: Atoms, product_atoms: Atoms) -> None:
    # Read reactant and product
    ref_images = [reactant_atoms, product_atoms]
    
    # Generate initial path using FB-ENM
    mxflx_fbenm = interpolate_fbenm(ref_images, correlated=True)
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
    mxflx.add_ipopt_options({'output_file': 'DMF_ipopt.out'})
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
        dx=g.IRC_DX_MAX, max_dx=g.IRC_DX_MAX, min_dx=g.IRC_DX_MIN,
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
    img.calc = None

    return [energy_eV, energy_kcal]

# 
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
    # Guess geometry='nonlinear', symmetrynumber=1
    # Use ignore_imag_modes=True
    vib_energies = vib.get_energies()
    
    thermo = IdealGasThermo(
        vib_energies=vib_energies, potentialenergy=electronic_energy,
        atoms=img, geometry='nonlinear', symmetrynumber=1, spin=(g.MULT-1)/2,
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


# Convert traj to xyz
def traj_to_xyz(traj, out_xyz_path):
    """
    Convert ASE traj to .xyz

    Parameters:
        traj (list of ase.Atoms), out_xyz_path (str): .xyz
    """
    try:
        for atoms in traj:
            atoms.info = {str(k): v for k, v in atoms.info.items()}
        write(out_xyz_path, traj)
    except Exception as e:
        print(f"Warning: An error occurred while writing {out_xyz_path}: {e}")

def write_energies(traj_name, csv_name=None, energy_recalc=False):
    if not csv_name:
        csv_name = os.path.splitext(traj_name)[0] + "_energy.csv"
    data = []
    tmp_name = traj_name + ".tmp"
    if energy_recalc:
        traj_out = Trajectory(tmp_name, "w")
    else:
        traj_out = None

    traj_in = Trajectory(traj_name)
    try:
        for i, atoms in enumerate(traj_in):
            if energy_recalc:
                atoms.info = {"charge": g.CHARGE, "spin": g.MULT}
                atoms.calc = make_calculator(g.CALC_TYPE, atoms, "energy_recalc")
                #atoms.calc = make_calculator(g.CALC_TYPE, atoms, f"energy_recalc_{i}")
            try:
                energy_ev = atoms.get_potential_energy()
                energy_hartree = energy_ev * g.EV_TO_HARTREE
                energy_kcal = energy_ev * g.EV_TO_KCAL_MOL
                data.append([i, energy_ev, energy_hartree, energy_kcal])
            except Exception as e:
                print(f"Warning: missing value for {traj_name} frame {i}: {e}")
                data.append([i, None, None, None])
            if energy_recalc:
                traj_out.write(atoms)
                atoms.calc = None
                del atoms
    finally:
        traj_in.close()
        if traj_out is not None:
            traj_out.close()
    if energy_recalc:
        os.replace(tmp_name, traj_name)
        
    df = pd.DataFrame(data,
        columns=["# image", "energy [eV]", "energy [hartree]", "energy [kcal/mol]"]
        )
    # Relative energy (kcal/mol)
    if df["energy [kcal/mol]"].notna().any():
        ref = df.loc[0, "energy [kcal/mol]"]
        df["Delta E vs. reactant [kcal/mol]"] = df["energy [kcal/mol]"] - ref
    else:
        df["Delta E vs. reactant [kcal/mol]"] = None
    # Write
    df.to_csv(csv_name, index=False)


# Finishing steps
def finalize_run():
    # Write relative energy (kcal/mol)
    df = pd.read_csv(g.R_CSV)
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
            df.to_csv(g.R_CSV, index=False)
        except Exception as e:
            print(f"Warning: An error occurred while writing {g.R_CSV}: {e}")
    
    # plot
    if g.SAVE_FIG_ON:
        figname = f"fig_{os.path.splitext(os.path.basename(g.R_CSV))[0]}.png"
        instant_plot(df, g.PEAK_IDX, figname)
    
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
            os.mkdir(args.directory)
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
            os.mkdir(args.directory)
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
