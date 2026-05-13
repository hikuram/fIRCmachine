# Computation Backends: Calculators in fIRCmachine

`fIRCmachine` extends the ASE `Calculator` interface to provide powerful backends tailored to different trade-offs between computational cost and accuracy. Backends are switched via `CALC_TYPE` in `default_config.py`.

## 1. Machine Learning Interatomic Potentials (MLIP)
* **`orbmol` (Orbital Materials v3):**
  Utilizes the `orb_v3_conservative_omol` model from the `orb_models` package. It runs on the GPU (`cuda`) with `float64` precision, enabling fast and accurate PES exploration. Highly recommended for DMF and initial IRC calculations.
* **`orbmol+alpb` (Delta ML Approach):**
  An advanced hybrid method that calculates the gas-phase energy using an MLIP and adds the solvation energy difference (Delta) using a low-cost semi-empirical quantum chemistry method (xTB).
  * Implementation: Uses ASE's `LinearCombinationCalculator`.
  * $E_{total} = E_{MLIP(gas)} + (E_{GFN1-xTB(solv)} - E_{GFN1-xTB(gas)})$
  * Powered by the custom `DualTBLite` class (in `dual_tblite_delta.py`), it dramatically reduces SCF cycles by reusing the wavefunction (Result container) from the solvated calculation as the initial guess for the gas-phase calculation.

## 2. First-Principles (PySCF / gpu4pyscf)
Used for rigorous electronic structure calculations and final energy refinements. Profiles are managed in `pyscf_config.json`.

* **Hardware Acceleration:**
  When `g.DEVICE == "cuda"`, calculations are automatically offloaded to `gpu4pyscf`, significantly reducing computation time.
* **3c-Functional Support (`pyscf_3c.py`):**
  Includes a dedicated wrapper to correctly implement composite density functionals like `r2scan-3c` and `b97-3c`. It features `get_gradient_method` and `get_Hessian_method` to inject the necessary **dispersion correction gradients and Hessians**, which are often missing in standard ASE PySCF interfaces.
* **Solvation:**
  Setting `"with_solvent": true` in the JSON config automatically applies the SMD solvation model.
* **Data Export (`pyscf_exporter.py`):**
  Automatically extracts orbital energies (HOMO/LUMO), Mulliken charges, and dipole moments upon job completion, exporting them in JSON and Molden formats for easy downstream visualization and analysis.
