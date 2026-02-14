# Copilot Instructions for fIRCmachine

## Overview
fIRCmachine is a Python-based computational chemistry toolkit for reaction path and vibrational analysis. It leverages ASE, PySCF, Sella, and custom forcefield modules to perform IRC, TS optimization, and vibrational calculations. The main entry points are:
- `fIRCmachine.py`: Full workflow (IRC, TS, VIB, etc.)
- `pIRCmachine.py`: IRC-focused workflow
- `sVIBmachine.py`: Vibrational analysis workflow

## Architecture & Data Flow
- **Global config:** All scripts import and modify `default_config.py` as `g`, controlling workflow toggles and parameters (all UPPERCASE, e.g., `g.CALC_TYPE`).
- **Component scripts:**
  - `fIRCmachine.py`: Orchestrates all steps, reads/writes trajectories, manages calculation types.
  - `pIRCmachine.py`/`sVIBmachine.py`: Specialize by toggling config flags for IRC or VIB only.
- **External dependencies:** ASE, PySCF, Sella, orb_models, gpu4pyscf, dmf, cupy.
- **Data flow:** Input trajectories (ASE format) → calculation steps (TS, IRC, VIB) → output files/logs.

## Developer Workflows
- **Run full workflow:**
  ```bash
  python fIRCmachine/fIRCmachine.py -d <directory>
  ```
- **Run IRC only:**
  ```bash
  python fIRCmachine/pIRCmachine.py -d <directory>
  ```
- **Run VIB only:**
  ```bash
  python fIRCmachine/sVIBmachine.py -d <directory>
  ```
- **Config tweaks:** Edit `default_config.py` or override in script (see commented lines in each script).

## Project Conventions
- **Global config pattern:** Use `g.<PARAM>` (all UPPERCASE) for runtime config, override in script as needed.
- **Calculation type:** Set `CALC_TYPE` and `DEVICE` in config for backend selection.
- **Logging:** Timing and suggestions are logged to files (see `TIME_LOG_NAME`, `WRITE_SUGGESTIONS_ON`).
- **Input/output:** Trajectories are read/written using ASE; output directory is specified via `-d` argument.

## Integration Points
- **ASE:** For atomic structures, IO, optimization, vibrations.
- **PySCF/gpu4pyscf:** For quantum chemistry calculations.
- **orb_models:** Custom forcefield backend.
- **Sella:** For TS optimization and IRC.
- **DMF:** For direct max flux calculations.

## Examples
- To run a full workflow on GPU:
  ```bash
  python fIRCmachine/fIRCmachine.py -d results --device cuda
  ```
- To customize calculation type:
  Edit `default_config.py`: `CALC_TYPE = "pyscf"`

## Key Files
- `fIRCmachine/fIRCmachine.py`: Main workflow
- `fIRCmachine/pIRCmachine.py`: IRC workflow
- `fIRCmachine/sVIBmachine.py`: VIB workflow
- `fIRCmachine/default_config.py`: Global config

---
**Update this file if new scripts, config flags, or workflows are added.**
