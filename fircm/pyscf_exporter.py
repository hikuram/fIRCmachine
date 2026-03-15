import json
import numpy as np
from pyscf.tools import molden

def export_pyscf_single_point(atoms, prefix="job"):
    """
    Extracts SCF results and parameters from an ASE Atoms object
    calculated via gpu4pyscf, and exports them to JSON and Molden files.
    
    Args:
        atoms (ase.Atoms): The ASE Atoms object after calculation.
        prefix (str): Prefix for the output filenames.
        
    Returns:
        dict: The extracted data dictionary.
    """
    calc = atoms.calc
    
    # 1. Get the GPU method object (which holds the converged results)
    mf_gpu = getattr(calc, 'method_scan', None)
    if mf_gpu is None:
        mf_gpu = getattr(calc, 'method', None)
        
    if mf_gpu is None:
        raise RuntimeError("Could not find the PySCF method object in atoms.calc.")

    # 2. Create a clean CPU SCF object for structural/parameter reference
    if hasattr(mf_gpu, 'base'):
        mf_cpu = mf_gpu.base.to_cpu()
    else:
        mf_cpu = mf_gpu.to_cpu()

    # 3. MANUALLY INJECT the converged results from GPU to CPU
    # This is required because .to_cpu() on a Scanner often strips the dynamic results.
    for attr in ['e_tot', 'mo_energy', 'mo_coeff', 'mo_occ', 'converged']:
        val = getattr(mf_gpu, attr, None)
        if val is not None:
            # Convert CuPy array to NumPy array
            if hasattr(val, 'get'):
                val = val.get()
            # Handle UKS/UHF where results might be a tuple/list of CuPy arrays
            elif isinstance(val, (list, tuple)):
                val = np.array([v.get() if hasattr(v, 'get') else v for v in val])
            setattr(mf_cpu, attr, val)

    # Manually inject scf_summary
    scf_summary_gpu = getattr(mf_gpu, 'scf_summary', {})
    scf_summary_cpu = {}
    for k, v in scf_summary_gpu.items():
        scf_summary_cpu[k] = v.get() if hasattr(v, 'get') else v
    mf_cpu.scf_summary = scf_summary_cpu

    # 4. Extract data
    mol = mf_cpu.mol
    data = {}
    data['name'] = prefix

    # --- Structure Information ---
    data['symbols'] = [mol.atom_symbol(i) for i in range(mol.natm)]
    data['positions'] = mol.atom_coords(unit='ANG').tolist()

    # --- Molecule Settings ---
    data['charge'] = mol.charge
    data['spin'] = mol.spin
    data['basis'] = mol.basis
    data['ecp'] = mol.ecp
    data['symmetry'] = mol.symmetry

# --- DFT/SCF Settings ---
    data['xc'] = getattr(mf_gpu, 'xc', 'HF')
    data['nlc'] = getattr(mf_gpu, 'nlc', '')
    data['disp'] = getattr(mf_gpu, 'disp', None)
    data['scf_conv_tol'] = getattr(mf_gpu, 'conv_tol', None)
    data['max_cycle'] = getattr(mf_gpu, 'max_cycle', None)

    # --- Grids Settings ---
    if hasattr(mf_gpu, 'grids'):
        atom_grid = getattr(mf_gpu.grids, 'atom_grid', None)
        if isinstance(atom_grid, dict) and not atom_grid:
            atom_grid = None
        elif isinstance(atom_grid, tuple):
            atom_grid = list(atom_grid)
            
        data['grids_atom_grid'] = atom_grid
        data['grids_level'] = getattr(mf_gpu.grids, 'level', None)
        
        prune_func = getattr(mf_gpu.grids, 'prune', None)
        data['grids_prune'] = prune_func.__name__ if hasattr(prune_func, '__name__') else str(prune_func)

    if hasattr(mf_gpu, 'nlcgrids'):
        nlc_atom_grid = getattr(mf_gpu.nlcgrids, 'atom_grid', None)
        if isinstance(nlc_atom_grid, dict) and not nlc_atom_grid:
            nlc_atom_grid = None
        elif isinstance(nlc_atom_grid, tuple):
            nlc_atom_grid = list(nlc_atom_grid)
            
        data['nlcgrids_atom_grid'] = nlc_atom_grid
        data['nlcgrids_level'] = getattr(mf_gpu.nlcgrids, 'level', None)

    # --- Solvent Settings ---
    if getattr(mf_gpu, 'with_solvent', None) is not None:
        data['solvent_method'] = getattr(mf_gpu.with_solvent, 'method', 'unknown')
        data['solvent_name'] = getattr(mf_gpu.with_solvent, 'solvent', None)
        data['solvent_eps'] = getattr(mf_gpu.with_solvent, 'eps', None)
    else:
        data['solvent_method'] = None
        data['solvent_name'] = None
        data['solvent_eps'] = None

    # --- Energy Results ---
    data['e_tot'] = getattr(mf_cpu, 'e_tot', None)
    data['e1'] = scf_summary_cpu.get('e1', 0.0)
    data['e_coul'] = scf_summary_cpu.get('coul', 0.0)
    data['e_xc'] = scf_summary_cpu.get('exc', 0.0)
    data['e_disp'] = scf_summary_cpu.get('dispersion', 0.0)
    data['e_solvent'] = scf_summary_cpu.get('e_solvent', 0.0)

    # --- Orbital Energies ---
    mo_energy = getattr(mf_cpu, 'mo_energy', None)
    
    if mo_energy is not None:
        data['mo_energy'] = mo_energy.tolist()

        if mo_energy.ndim == 2:
            na, nb = mf_cpu.nelec
            data['e_lumo_alpha'] = mo_energy[0][na] if na < len(mo_energy[0]) else None
            data['e_lumo_beta']  = mo_energy[1][nb] if nb < len(mo_energy[1]) else None
            data['e_homo_alpha'] = mo_energy[0][na-1] if na > 0 else None
            data['e_homo_beta']  = mo_energy[1][nb-1] if nb > 0 else None
            data['na'] = na
            data['nb'] = nb
        else:
            nocc = mol.nelectron // 2
            data['e_lumo'] = mo_energy[nocc] if nocc < len(mo_energy) else None
            data['e_homo'] = mo_energy[nocc-1] if nocc > 0 else None
            data['nocc'] = nocc
            data['e_lumo_alpha'] = data['e_lumo']
            data['e_lumo_beta']  = data['e_lumo']
            data['e_homo_alpha'] = data['e_homo']
            data['e_homo_beta']  = data['e_homo']
            data['na'] = nocc
            data['nb'] = nocc
    else:
        print("Warning: mo_energy is still None. Extraction failed.")
        data['mo_energy'] = None

    # --- Population Analysis ---
    try:
        mul_pop, dip_mom = mf_cpu.analyze(verbose=0)
        data['mulliken_pop'] = mul_pop[0].tolist()
        data['mulliken_charge'] = mul_pop[1].tolist()
        data['dip_moment'] = dip_mom.tolist()
    except Exception as e:
        print(f"Warning: Population analysis failed: {e}")

    # --- Export JSON ---
    json_filename = f"{prefix}_pyscf.json"
    try:
        with open(json_filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved JSON data to {json_filename}")
    except Exception as e:
        print(f"Warning: JSON export failed: {e}")

    # --- Export Molden ---
    molden_filename = f"{prefix}.molden"
    try:
        molden.from_scf(mf_cpu, molden_filename)
        print(f"Saved Molden file to {molden_filename}")
    except Exception as e:
        print(f"Warning: Molden export failed: {e}")

    return data
