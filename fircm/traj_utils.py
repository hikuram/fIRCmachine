"""
traj_utils.py
Utility functions for processing ASE trajectory (.traj) files and extracting data.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional
from ase.io import read, write
from ase.io.trajectory import Trajectory
from scipy.signal import find_peaks
from utils import log

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

def select_highest_peak_file(peak_files: List[str]) -> Optional[str]:
    """Select the highest-energy internal peak from the detected peak files."""
    if len(peak_files) <= 2:
        return None

    max_energy = -np.inf
    max_peak_file = None

    for peak_file in peak_files[1:-1]:
        atoms = read(peak_file)
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            energy = -np.inf

        if energy > max_energy:
            max_energy = energy
            max_peak_file = peak_file

    return max_peak_file

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
    
    try:
        for i, atoms in enumerate(traj_in):
            if energy_recalc:
                atoms.info = {"charge": g.CHARGE, "spin": g.MULT}
                atoms.calc = make_calculator(g.CALC_TYPE, atoms, "energy_recalc")
            try:
                energy_ev = atoms.get_potential_energy()
                energy_hartree = energy_ev * g.EV_TO_HARTREE
                energy_kcal = energy_ev * g.EV_TO_KCAL_MOL
                data.append([i, energy_ev, energy_hartree, energy_kcal])
            except Exception as e:
                log("Warn", f"Missing value for {traj_name} frame {i}: {e}")
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
        
    df = pd.DataFrame(data, columns=["# image", "energy [eV]", "energy [hartree]", "energy [kcal/mol]"])
    
    if previous_image is not None:
        if len(previous_image) != len(df):
            raise ValueError("Length of previous_image must match the number of frames")
        df["previous_#image"] = previous_image
        cols = ["# image", "previous_#image"] + [c for c in df.columns if c not in ["# image", "previous_#image"]]
        df = df[cols]
        
    # Calculate relative energy vs reactant
    if df["energy [kcal/mol]"].notna().any():
        ref = df.loc[0, "energy [kcal/mol]"]
        df["Delta E vs. reactant [kcal/mol]"] = df["energy [kcal/mol]"] - ref
    else:
        df["Delta E vs. reactant [kcal/mol]"] = None
        
    df.to_csv(csv_name, index=False)
