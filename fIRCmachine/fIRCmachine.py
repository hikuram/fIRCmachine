import os
import sys
import shutil
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
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from pyscf import M
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
from gpu4pyscf.tools.ase_interface import PySCF
from dmf import DirectMaxFlux, interpolate_fbenm
from sella import Sella, Constraints, IRC

# overwrite global variables
# ...Example settings are described in README or default_config.py...
#
##g.CHARGE = 0
#g.MULT = 1
#g.NMOVE = 40
#g.UPDATE_TEVAL = False
#g.DMF_CONVERGENCE = "tight"
#
#g.SELLA_INTERNAL = True
#g.IRC_DX = 0.08
#
#g.EV_TO_KCAL_MOL = 23.0605
#g.EV_TO_HARTREE = 1 / 27.2114  # ≒ 0.0367493

# FB-ENM/DMF in first
def init_path_search():
    # Optimization of the reaction path by FB-ENM/DMF
    # Start timer (total)
    
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

# following iterations for all lmax
def iter_lmax():
    df_new = pd.read_csv(g.R_CSV)
    # detect & save local maxima
    peak_files = []
    peak_files = extract_peaks_from_traj(g.I_TRAJ, "lmax.xyz", prominence=0.01)
    
    # write csv (accepts a pair of elements or lists)
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
    
    # sub iter 1: ignores endpoints
    irc_trajs_str = ""
    t_tsopt_irc_start = timepfc()
    for i, peak_file in enumerate(peak_files):
        if len(peak_files) > 2:
            if i == 0 or i == len(peak_files)-1:
                continue
        
        base_name = os.path.splitext(peak_file)[0]
        idx = int(base_name.split('_')[-1].split('.')[0]) #index of lmax
        atoms = read(peak_file)
        atoms.info["charge"] = g.CHARGE
        atoms.info["spin"] = g.MULT
        
        # == do TSopt ===================
        if g.TSOPT_ON:
            t_tsopt_start = timepfc()
            try:
                tsopt_img(base_name+".xyz")
            except Exception as e:
                print(f"Warning: TSopt failed: {e}")
            t_tsopt = timepfc() - t_tsopt_start
            write_result('time_TSopt [s]', t_tsopt)
            print(f"tsopt {t_tsopt} sec")
        
        # == do IRC ===================
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
    
    # sub iter 2: includes endpoints
    t_vib_sum_start = timepfc()
    for i, peak_file in enumerate(peak_files):
        base_name = os.path.splitext(peak_file)[0]
        idx = int(base_name.split('_')[-1].split('.')[0]) #index of lmax
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
            write_result(
                ['time_vib [s]', 'ZPE [kcal/mol]', 'E_0K [kcal/mol]', 'H [kcal/mol]', 'G [kcal/mol]'],
                [t_vib]+vib_result
            )
            print(f"vibrations {t_vib} sec")
        
        # == Other jobs ===================
        if g.OTHER_JOBS_EXAMPLE_ON:
            from ase.geometry import get_distances
            _, d_Ir = get_distances(atoms.positions, atoms.positions[0])
            print(d_Ir[1][0])
            write_result('d_Ir_betaH [angs]', d_Ir[1][0])
            
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
        # under construction
        # write results
            #    df_new.at[df_new.index[idx], 'energy_DFT [eV]'] = np.nan #energy_eV
            #    df_new.at[df_new.index[idx], 'energy_DFT [kcal/mol]'] = np.nan #energy_eV * g.EV_TO_KCAL_MOL
            #    df_new.at[df_new.index[idx], 'time_DFT [s]'] = np.nan #t_dft
        # ==
        
    t_vib_sum = timepfc() - t_vib_sum_start
    txt = f"* Vibrations_Total      | {t_vib_sum:>12.2f} s  *\n"
    write_line(g.time_log_name, txt)


# set calculator
def myCalculator(type, atoms, base_name):
    #pyscf
    if type == "pyscf":
        # pyscf config
        mol = M(atom=ase_atoms_to_pyscf(atoms), basis="def2-mTZVP",
            ecp="def2-TZVP", charge=g.charge, spin=g.mult-1,
            output=base_name+'_pyscf.log', verbose=4
        )
        mf = mol.RKS(xc="b973c", disp="d3bj", conv_tol=6e-10, max_cycle=400)
        mf = mf.SMD()
        mf.with_solvent.solvent = "water"
        #mf.with_solvent.eps = 78.3553  # water
        mf.grids.level = 5
        mf.nlcgrids.level = 4
        if g.device == "cuda":
            cupy.get_default_memory_pool().free_all_blocks()
            mf = mf.to_gpu()
        calculator = PySCF(method=mf)
        
    if type == "pyscf_3c":
        from redox.utils.pyscf_utils import PySCFCalculator, build_3c_method
        # build method
        config = {}
        config["xc"] = "r2scan3c"
        config["with_solvent"] = True
        config["solvent"] = {"method": "COSMO", "eps": 78.3553, "solvent": "water"}
        config["charge"] = g.charge
        config["spin"] = g.mult - 1
        input_atoms_list = [(ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())]
        config["inputfile"] = input_atoms_list
        config["verbose"] = 4
        config["output"] = base_name+'_pyscf.log'
        if "xc" in config and config["xc"].endswith("3c"):
            xc_3c = config["xc"]
            mf = build_3c_method(config)
        else:
            xc_3c = None
            mf = build_method(config)
        calculator = PySCFCalculator(mf, xc_3c=xc_3c)
        
    #pyscf_fine
    elif type == "pyscf_fine":
        # pyscf config
        mol = M(atom=ase_atoms_to_pyscf(atoms), basis="def2-TZVPD",
            ecp="def2-TZVPD", charge=g.charge, spin=g.mult-1,
            output=base_name+'_pyscf.log', verbose=4
        )
        mf = mol.RKS(xc="wb97m-v", conv_tol=6e-10, max_cycle=400)
        mf.grids.level = 5
        mf.nlcgrids.level = 4
        #if g.device == "cuda":
        #    mf = mf.to_gpu()
        calculator = PySCF(method=mf)
        
    #orbmol
    elif type == "orbmol":
        orbff = pretrained.orb_v3_conservative_omol(
            device=g.device,
            precision="float64",   # "float32"/ "float32-highest"/ "float64"
        )
        calculator = ORBCalculator(orbff, device=g.device)
        
    #orbmol+alpb
    elif type == "orbmol+alpb":
        orbff = pretrained.orb_v3_conservative_omol(
            device=g.device,
            precision="float64",   # "float32"/ "float32-highest"/ "float64"
        )
        solvation = ("alpb","water")
        acc = 0.1
        calc_mlip =  ORBCalculator(orbff, device=g.device)
        calc_xtb_sol = TBLite(method="GFN2-xTB", charge=g.charge, multiplicity=g.mult, solvation=solvation, accuracy=acc, verbosity=0, mixer_damping=0.1)
        calc_xtb_vac = TBLite(method="GFN2-xTB", charge=g.charge, multiplicity=g.mult, accuracy=acc, verbosity=0, mixer_damping=0.1)
        calculator = LinearCombinationCalculator([calc_mlip, calc_xtb_sol, calc_xtb_vac], [1, 1, -1])
        
    else:
        sys.exit("error: incorrect calc type")
    return calculator

        
    #orbmol+alpb
    calc_mlip =  ORBCalculator(orbff, device=device)
    calc_xtb_sol = TBLite(method="GFN2-xTB", charge=charge, multiplicity=mult, solvation=solvation, accuracy=acc, verbosity=0, mixer_damping=0.1)
    calc_xtb_vac = TBLite(method="GFN2-xTB", charge=charge, multiplicity=mult, accuracy=acc, verbosity=0, mixer_damping=0.1)
    calculator = LinearCombinationCalculator([calc_mlip, calc_xtb_sol, calc_xtb_vac], [1, 1, -1])
    

# parse input traj
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

    # fill NaN
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
    # Input basename (e.g., input.xyz → input)
    base_name = os.path.splitext(os.path.basename(maxima_filename))[0]
    print(f"Detected {len(peaks)} peak(s). Saving structures:")
    peak_list = []
    
    # Add first and last frames
    endpoints = np.array([0, len(traj) - 1])
    peaks = np.unique(np.concatenate([peaks, endpoints]))

    for idx in peaks:
        atoms = traj[idx]
        filename = f"{base_name}_{idx}.xyz"
        peak_list.append(filename)
        write(filename, atoms)
        print(f"  → {filename} (energy = {energies[idx]:.6f})")

    return peak_list


# run MEPopt with FB-ENM/DMF
def mepopt_dmf(reactant_atoms: Atoms, product_atoms: Atoms) -> None:
    # Read reactant and product
    ref_images = [reactant_atoms, product_atoms]
    
    # Generate initial path by FB-ENM
    mxflx_fbenm = interpolate_fbenm(ref_images, correlated=True)
    write('DMF_init.xyz', mxflx_fbenm.images)
    
    # Write initial path and its coefficients
    write('DMF_init.traj', mxflx_fbenm.images)
    coefs = mxflx_fbenm.coefs.copy()
    np.save('DMF_init_coefs', coefs)
    
    # Set up and solve Direct MaxFlux
    mxflx = DirectMaxFlux(ref_images, coefs=coefs, nmove=g.nmove, update_teval=g.update_teval)
    # Set up calculator
    for img in mxflx.images:
        img.info = {"charge": g.charge, "spin": g.mult}
        img.calc = myCalculator(g.calc_type, img, "DMF_init")
    # do solve
    mxflx.add_ipopt_options({'output_file': 'DMF_ipopt.out'})
    try:
        mxflx.solve(tol=g.DMF_convergence)
    except Exception as e:
        write("DMF_last_before_error.xyz", mxflx.images)
        write("DMF_last_before_error.traj", mxflx.images)
        sys.exit("abort: DirectMaxFlux.solve failed: {e}")
    
    # DMF_final.traj: Recalculate SPC for mxflx.images (some frames lack energy)
    final_images = []
    for img in mxflx.images:
        # Copy atoms and info
        atoms = Atoms(positions=img.get_positions(), numbers=img.get_atomic_numbers())
        atoms.info = {"charge": g.charge, "spin": g.mult}
        atoms.calc = myCalculator(g.calc_type, atoms, "DMF_final")
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
    # write result
    write_energies('DMF_final.traj', g.r_csv)
    g.suggestions.append(f"ase gui {g.current_dir}/DMF_final.traj")

# write txt
def write_line(txtfile_name, txt):
    with open(txtfile_name, 'a', encoding='utf-8') as f:
        f.write(txt)

# run Opt with ASE
def opt_img(xyz_name: str) -> Atoms:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info = {"charge": g.CHARGE, "spin": g.MULT}
    img.calc = myCalculator(g.CALC_TYPE, img, img_name)
    # Set up a ASE optimizer object with L-BFGS
    opt = LBFGS(img, trajectory=img_name+"_opt.traj", logfile=img_name+"_opt.log")
    opt.run(fmax=0.01, steps=10000)
    write(img_name+"_opt.xyz", img)
    images = read(img_name+"_opt.traj", index=':')
    traj_to_xyz(images, img_name+"_opt.traj.xyz")
    
    g.SUGGESTIONS.append(f"ase gui {g.CURRENT_DIR}/{img_name}_opt.traj")
    return img


# run Opt with Sella
def opt_sella_img(xyz_name: str) -> Atoms:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info = {"charge": g.CHARGE, "spin": g.MULT}
    img.calc = myCalculator(g.CALC_TYPE, img, img_name)
    # Set up a Sella Dynamics object with "order=0"
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


# run TSopt with Sella
def tsopt_img(xyz_name: str) -> Atoms:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info = {"charge": g.CHARGE, "spin": g.MULT}
    img.calc = myCalculator(g.CALC_TYPE, img, img_name)
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


# run IRC with Sella
def irc_img(xyz_name: str) -> List[float]:
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info = {"charge": g.CHARGE, "spin": g.MULT}
    img.calc = myCalculator(g.CALC_TYPE, img, img_name)
    # Set up a Sella IRC object
    opt = IRC(img, trajectory=img_name+'_irc.traj',
        logfile=img_name+"_irc.log",
        dx=g.IRC_DX, eta=1e-4, gamma=0.4
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

    generate_half_cycle(+1)  # +mode → original
    generate_half_cycle(-1)  # -mode → original
    write(output, images)
    print(f"[Info] Wrote {len(images)} frames to {output}")


# run Vib and Thermo
def vib_img(xyz_name):
    img = read(xyz_name)
    img_name = os.path.splitext(xyz_name)[0]
    img.info["charge"] = g.charge
    img.info["spin"] = g.mult
    img.calc = myCalculator(g.calc_type, img, img_name)
    #forces = img.get_forces()
    vib = Vibrations(img, name="vib_temp")
    vib.run()
    vib.summary(log=img_name+'_vibsummary.txt')
    vib.get_frequencies()
    #generate_vibration_xyz(atoms, vib, 0, steps, scale, vib_filename)
    for mode in range(0, 3):
        vib_filename = f"{img_name}_vib_{mode}.xyz"
        generate_vibration_xyz(img, vib, mode, output=vib_filename)
    g.suggestions.append(f"ase gui {g.current_dir}/{img_name}_vib_*.xyz")

    # Ideal-gas limit
    # guess geometry='nonlinear', symmetrynumber=1
    # use ignore_imag_modes=True
    potentialenergy = img.get_potential_energy()
    vib_energies = vib.get_energies()
    
    thermo = IdealGasThermo(
        vib_energies=vib_energies, potentialenergy=potentialenergy,
        atoms=img, geometry='nonlinear', symmetrynumber=1, spin=(g.mult-1)/1,
        ignore_imag_modes=True
    )
    energy_eV = vib.atoms.get_potential_energy()
    zpe_eV = vib.get_zero_point_energy()  # unit: eV
    H_eV = thermo.get_enthalpy(temperature=298.15)
    G_eV = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.0)
        # room temperature=298.15 # standard atmosphere=101325.0
    
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

# Extract enegies from traj to csv
def write_energies(traj_name, csv_name=None):
    if not csv_name:
        csv_name = os.path.splitext(traj_name)[0]+"_energy.csv"
    images = read(traj_name, index=":")
    # traj: energies
    data = []
    for i, atoms in enumerate(images):
        try:
            energy_ev = atoms.get_potential_energy()
            energy_hartree = energy_ev * g.EV_TO_HARTREE
            energy_kcal = energy_ev * g.EV_TO_KCAL_MOL
            data.append([i, energy_ev, energy_hartree, energy_kcal])
        except Exception as e:
            print(f"Warning: missing value for {traj_name}.traj atom {i}: {e}")
            data.append([i, None, None, None])
    df = pd.DataFrame(data,
        columns=["image", "energy [eV]", "energy [hartree]", "energy [kcal/mol]"]
        )
    # Relative energy (kcal/mol)
    if df["energy [kcal/mol]"].notna().any():
        ref = df.loc[0, "energy [kcal/mol]"]
        df["Delta E vs. reactant [kcal/mol]"] = df["energy [kcal/mol]"] - ref
    else:
        df["Delta E vs. reactant [kcal/mol]"] = None
    # write
    df.to_csv(csv_name, index=False)


# finishing work
def finishing():
    # write relative * energy (kcal/mol)
    if g.VIB_ON:
        try:
            df = pd.read_csv(g.R_CSV)
            if df["E_0K [kcal/mol]"].notna().any():
                df["Delta E_0K vs. reactant [kcal/mol]"] = df["E_0K [kcal/mol]"] - df.loc[0, "E_0K [kcal/mol]"]
            if df["H [kcal/mol]"].notna().any():
                df["Delta H vs. reactant [kcal/mol]"] = df["H [kcal/mol]"] - df.loc[0, "H [kcal/mol]"]
            if df["G [kcal/mol]"].notna().any():
                df["Delta G vs. reactant [kcal/mol]"] = df["G [kcal/mol]"] - df.loc[0, "G [kcal/mol]"]
            df.to_csv(g.R_CSV, index=False)
        except Exception as e:
            print(f"Warning: An error occurred while writing {g.R_CSV}: {e}")
    
    # suggest the next steps
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
    parser.add_argument("-m", "--method", type=str, required=False, help="calculation method of the PES")
    if g.INIT_PATH_SEARCH_ON:
        parser.add_argument("-r", "--reactant", type=str, required=True, help="inputfile for the reactant .xyz file")
        parser.add_argument("-p", "--product", type=str, required=True, help="inputfile for the product .xyz file")
    else:
        parser.add_argument("-i", "--input", type=str, required=True, default="input.traj", help="input .traj file (ignored if the DMF path search is enabled)")
    parser.add_argument("-rs", "--result", type=str, required=False, default="result.csv", help="resulting dataframe csv")
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
    if args.method:
        g.CALC_TYPE = args.method
    g.R_CSV = args.result
    if os.path.exists(g.R_CSV):
        print(f"info: {g.R_CSV} will be overwritten")
    else:
        print(f"info: {g.R_CSV} will be made")
        write_energies(g.I_TRAJ, g.R_CSV)
    
    # main
    if g.INIT_PATH_SEARCH_ON:
        init_path_search()
        g.I_TRAJ = "DMF_final.traj" #ignores args.input
    iter_lmax()
    
    # finish
    finishing()
    t_total = timepfc() - t_total_start
    txt = f"* Total_Time            | {t_total:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)
    print(f"finished at: {datetime.now()}")

