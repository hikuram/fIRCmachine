# Workflow Details: How fIRCmachine Operates

The workflow of `fIRCmachine` is not just a sequential execution of scripts; it is heavily designed around "robustness" against noise and numerical artifacts on the Potential Energy Surface (PES).

## 1. Initial Path Search (DMF, NEB, SCAN)
Given the input structures, the toolkit generates a reliable initial path. The method is controlled by `INIT_PATH_METHOD`.

* **FB-ENM & DirectMaxFlux (DMF):**
  Uses `interpolate_fbenm` for initial interpolation and `pydmf` for path optimization. It robustly handles large geometric changes without atomic collisions.
* **Nudged Elastic Band (NEB):**
  A standard ASE NEB implementation, enhanced to cleanly separate the final optimized path (`init_path.traj`) from the noisy optimization history (`NEB_history.traj`). It also fully respects `FIXED_ATOMS` constraints across all intermediate images.
* **Relaxed PES Scan (SCAN):**
  Performs constrained optimization along a specified internal coordinate (bond, angle, dihedral). If the geometry breaks due to excessive strain (e.g., forcing a bond too far), a `try...except` block catches the SCF/optimization failure, stops the scan early, and safely preserves the path generated up to that point.
* **Peak Extraction (`extract_peaks_from_traj`):**
  Scans the energy along the generated path and automatically extracts local maxima as TS (Transition State) candidates.

## 2. TS Optimization & Adaptive IRC
Strict TS optimization and IRC calculations are performed on the extracted peak structures using Sella.

* **Automatic Symmetry Detection:**
  The `get_symmetry_info` function calls PySCF under the hood to determine the point group. Since internal coordinates can cause singularity errors in linear molecules (e.g., `Dinfh`) or highly symmetric molecules (e.g., `Oh`, `Td`), the toolkit automatically detects these and falls back to Cartesian coordinates for optimization.
* **AdaptiveIRC (Noise-Resilient IRC):**
  To prevent IRC from crashing due to minor numerical noise inherent in Machine Learning Interatomic Potentials (MLIPs), a custom class `AdaptiveIRC` is implemented.
  * **Dynamic Step Sizing:** Increases `dx` when the path is stable and shrinks `dx` upon convergence failure.
  * **Rollback Mechanism:** If retries fail, it rolls back to the last safely accepted step (`max_rollback=4`) and attempts to re-enter the path with a smaller `dx`.

## 3. Vibrational Analysis & Thermochemistry
The vibrational analysis (VIB) for calculating thermodynamic quantities (e.g., Gibbs free energy) includes unique routines to eliminate practical artifacts.

* **True TS Mode Protection (`vib_img`):**
  To distinguish true TS modes from numerical noise (e.g., methyl group rotations appearing as small imaginary frequencies), a strict threshold of **40 cm^-1** (`ts_recognition_threshold_cm1`) is applied. Imaginary modes below this are treated as noise (flat PES at a local minimum), while the largest imaginary mode is only protected as a "True TS" if it exceeds this threshold.
* **Truhlar's Floor Correction:**
  To prevent overestimation of entropy from low-frequency modes, a Truhlar's Floor correction is applied. Modes below **50 cm^-1** (`freq_cutoff_cm1`) are uniformly raised to 50 cm^-1 to calculate a stable standard Gibbs energy (`G [kcal/mol]`).
