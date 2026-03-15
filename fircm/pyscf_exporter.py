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
    # 1. Extract the PySCF SCF object and convert it to CPU (NumPy based)
    mf_gpu = atoms.calc.mf
    mf_cpu = mf_gpu.cpu()
    mol = mf_cpu.mol

    data = {}
    data['name'] = prefix

    # --- Structure Information (Ground Truth from PySCF) ---
    data['symbols'] = [mol.atom_symbol(i) for i in range(mol.natm)]
    # Convert coordinates to Angstrom
    data['positions'] = mol.atom_coords(unit='ANG').tolist()

    # --- Molecule Settings ---
    data['charge'] = mol.charge
    data['spin'] = mol.spin
    data['basis'] = mol.basis
    data['ecp'] = mol.ecp
    data['symmetry'] = mol.symmetry

    # --- DFT/SCF Settings ---
    data['xc'] = getattr(mf_cpu, 'xc', 'HF')
    data['disp'] = getattr(mf_cpu, 'disp', None)
    data['scf_conv_tol'] = mf_cpu.conv_tol
    data['max_cycle'] = mf_cpu.max_cycle

    # Grids
    if hasattr(mf_cpu, 'grids'):
        data['grids_atom_grid'] = getattr(mf_cpu.grids, 'atom_grid', None)
        data['grids_level'] = getattr(mf_cpu.grids, 'level', None)
        prune_func = getattr(mf_cpu.grids, 'prune', None)
        data['grids_prune'] = prune_func.__name__ if prune_func else None

    # NLC Grids (for rVdW functionals etc.)
    if hasattr(mf_cpu, 'nlcgrids'):
        data['nlcgrids_atom_grid'] = getattr(mf_cpu.nlcgrids, 'atom_grid', None)
        data['nlcgrids_level'] = getattr(mf_cpu.nlcgrids, 'level', None)

    # Solvent model (PCM/SMD)
    if getattr(mf_cpu, 'with_solvent', None) is not None:
        data['solvent_method'] = getattr(mf_cpu.with_solvent, 'method', 'unknown')
        data['solvent_eps'] = getattr(mf_cpu.with_solvent, 'eps', None)
    else:
        data['solvent_method'] = None

    # --- Energy Results ---
    data['e_tot'] = mf_cpu.e_tot
    scf_summary = getattr(mf_cpu, 'scf_summary', {})
    data['e1'] = scf_summary.get('e1', 0.0)
    data['e_coul'] = scf_summary.get('coul', 0.0)
    data['e_xc'] = scf_summary.get('exc', 0.0)
    data['e_disp'] = scf_summary.get('dispersion', 0.0)
    data['e_solvent'] = scf_summary.get('e_solvent', 0.0)

    # --- Orbital Energies ---
    mo_energy = mf_cpu.mo_energy
    data['mo_energy'] = mo_energy.tolist()

    if mo_energy.ndim == 2:
        # Open-shell (UKS/UHF)
        na, nb = mf_cpu.nelec
        # Boundary checks included for robustness
        data['e_lumo_alpha'] = mo_energy[0][na] if na < len(mo_energy[0]) else None
        data['e_lumo_beta']  = mo_energy[1][nb] if nb < len(mo_energy[1]) else None
        data['e_homo_alpha'] = mo_energy[0][na-1] if na > 0 else None
        data['e_homo_beta']  = mo_energy[1][nb-1] if nb > 0 else None
        data['na'] = na
        data['nb'] = nb
    else:
        # Closed-shell (RKS/RHF)
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

    # --- Population Analysis ---
    try:
        # verbose=0 suppresses printing to stdout during analysis
        mul_pop, dip_mom = mf_cpu.analyze(verbose=0)
        data['mulliken_pop'] = mul_pop[0].tolist()
        data['mulliken_charge'] = mul_pop[1].tolist()
        data['dip_moment'] = dip_mom.tolist()
    except Exception as e:
        print(f"Warning: Population analysis failed: {e}")

    # --- Export JSON ---
    json_filename = f"{prefix}_pyscf.json"
    with open(json_filename, 'w') as f:
        # Ensure fallback for any custom types (though all should be standard Python types now)
        json.dump(data, f, indent=4)
    print(f"Saved JSON data to {json_filename}")

    # --- Export Molden ---
    molden_filename = f"{prefix}.molden"
    try:
        molden.from_scf(mf_cpu, molden_filename)
        print(f"Saved Molden file to {molden_filename}")
    except Exception as e:
        print(f"Warning: Molden export failed: {e}")

    return data
