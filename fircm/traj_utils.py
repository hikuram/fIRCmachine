"""
traj_utils.py
Utility functions for processing ASE trajectory (.traj) files and extracting data.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional
from ase.io import write
from ase.io.trajectory import Trajectory
from scipy.signal import find_peaks
from utils import log, read
try:
    import rmsd
    HAS_RMSD = True
except ImportError:
    HAS_RMSD = False

# Project modules
import default_config as g
from ase_calculators import make_calculator

def extract_peaks_from_traj(trajfile: str, maxima_filename: str, prominence: float = 0.01) -> List[str]:
    """
    Read a trajectory file, find local energy maxima (peaks), and save them as .xyz files.
    Returns a list of generated .xyz filenames and their indices.
    """
    traj = read(trajfile, index=':')
    energies = []
    for i, atoms in enumerate(traj):
        try:
            energy = atoms.get_potential_energy()
        except Exception as e:
            log("Warn", f"Missing value for {trajfile} atom {i}: {e}")
            energy = np.nan
        energies.append(energy)
    energies = np.array(energies)

    # Fill NaN values to avoid breaking the peak finding algorithm
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

    # Detect peaks
    peaks, _ = find_peaks(energies_filled, prominence=prominence)
    base_name = os.path.splitext(os.path.basename(maxima_filename))[0]
    log("Info", f"Detected {len(peaks)} peak(s) (excluding endpoints). Saving structures:")
    
    if len(peaks) > 0:
        max_peak_idx = peaks[np.argmax(energies_filled[peaks])]
        g.HIGHEST_PEAK_FILE = f"{base_name}_{max_peak_idx}.xyz"
    else:
        g.HIGHEST_PEAK_FILE = None

    peak_files = []
    # Always include the first and last frames as endpoints
    endpoints = np.array([0, len(traj) - 1])
    g.PEAK_IDX = np.unique(np.concatenate([peaks, endpoints]))

    for idx in g.PEAK_IDX:
        atoms = traj[idx]
        filename = f"{base_name}_{idx}.xyz"
        peak_files.append(filename)
        write(filename, atoms)
        log("I/O", f"Wrote {filename} (energy = {energies[idx]:.6f})")

    return peak_files, g.PEAK_IDX

def split_traj_to_xyz(trajfile: str, prefix: str) -> List[str]:
    """Split a trajectory file into multiple single-frame .xyz files."""
    traj = read(trajfile, index=":")
    xyz_files = []

    for i, atoms in enumerate(traj):
        filename = f"{prefix}_{i}.xyz"
        write(filename, atoms)
        xyz_files.append(filename)

    return xyz_files

def traj_to_xyz(traj, out_xyz_path):
    """Convert an ASE trajectory list to an .xyz file format."""
    try:
        for atoms in traj:
            atoms.info = {str(k): v for k, v in atoms.info.items()}
        write(out_xyz_path, traj)
    except Exception as e:
        log("Warn", f"An error occurred while writing {out_xyz_path}: {e}")

def write_energies(traj_name, csv_name=None, energy_recalc=False, previous_image=None):
    """
    Extract energies from a trajectory file and write them to a CSV file.
    Optionally recalculates the single-point energy for each frame.
    """
    if not csv_name:
        csv_name = os.path.splitext(traj_name)[0] + "_energy.csv"
    data = []
    tmp_name = traj_name + ".tmp"
    
    traj_out = Trajectory(tmp_name, "w") if energy_recalc else None
    traj_in = Trajectory(traj_name)
    
    calc_rmsd = getattr(g, 'CALC_RMSD_ON', False) and HAS_RMSD
    if getattr(g, 'CALC_RMSD_ON', False) and not HAS_RMSD:
        log("Warn", "rmsd module is not installed. Skipping RMSD calculation.")

    pos_0 = None
    pos_prev = None
    
    try:
        for i, atoms in enumerate(traj_in):
            if energy_recalc:
                atoms.info = {"charge": g.CHARGE, "spin": g.MULT}
                atoms.calc = make_calculator(g.CALC_TYPE, atoms, "energy_recalc")
            try:
                energy_ev = atoms.get_potential_energy()
                energy_hartree = energy_ev * g.EV_TO_HARTREE
                energy_kcal = energy_ev * g.EV_TO_KCAL_MOL
            except Exception as e:
                log("Warn", f"Missing value for {traj_name} frame {i}: {e}")
                energy_ev, energy_hartree, energy_kcal = None, None, None

            # === RMSD calculation (Heavy atoms only) ===
            rmsd_0 = None
            rmsd_prev = None
            if calc_rmsd:
                # Extract positions and atomic numbers
                pos = atoms.get_positions()
                atomic_numbers = atoms.get_atomic_numbers()
                
                # Filter out light elements (Hydrogen, Z=1)
                heavy_mask = atomic_numbers > 1
                heavy_pos = pos[heavy_mask]
                
                if len(heavy_pos) > 0:
                    # Translate centroid to origin
                    heavy_pos_centered = heavy_pos - rmsd.centroid(heavy_pos)
                    
                    if i == 0:
                        pos_0 = heavy_pos_centered
                        pos_prev = heavy_pos_centered
                        rmsd_0 = 0.0
                        rmsd_prev = 0.0
                    else:
                        # Calculate optimal rotation and RMSD using Kabsch algorithm
                        rmsd_0 = rmsd.kabsch_rmsd(pos_0, heavy_pos_centered)
                        rmsd_prev = rmsd.kabsch_rmsd(pos_prev, heavy_pos_centered)
                        pos_prev = heavy_pos_centered
                else:
                    # Fallback if the system only contains hydrogens (edge case)
                    rmsd_0 = 0.0
                    rmsd_prev = 0.0
            # ===========================================

            row = [i, energy_ev, energy_hartree, energy_kcal]
            if calc_rmsd:
                row.extend([rmsd_0, rmsd_prev])
            data.append(row)
            
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
        
    cols = ["# image", "energy [eV]", "energy [hartree]", "energy [kcal/mol]"]
    if calc_rmsd:
        # Update column names to clarify heavy-atom usage
        cols.extend(["Heavy-RMSD vs frame 0 [Å]", "Heavy-RMSD vs prev frame [Å]"])
        
    df = pd.DataFrame(data, columns=cols)
    
    if previous_image is not None:
        if len(previous_image) != len(df):
            raise ValueError("Length of previous_image must match the number of frames")
        df["previous_#image"] = previous_image
        cols_reorder = ["# image", "previous_#image"] + [c for c in df.columns if c not in ["# image", "previous_#image"]]
        df = df[cols_reorder]
        
    # Calculate relative energy vs reactant
    if df["energy [kcal/mol]"].notna().any():
        ref = df.loc[0, "energy [kcal/mol]"]
        df["Delta E vs. reactant [kcal/mol]"] = df["energy [kcal/mol]"] - ref
    else:
        df["Delta E vs. reactant [kcal/mol]"] = None
        
    df.to_csv(csv_name, index=False)
