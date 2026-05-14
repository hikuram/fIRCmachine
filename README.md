# fIRCmachine

A pragmatic, highly automated Python toolkit for computational chemistry. 

fIRCmachine orchestrates reaction path searching (FB-ENM/DMF), Intrinsic Reaction Coordinate (IRC) calculations, Transition State (TS) optimizations, and vibrational analysis (VIB). It is designed to be robust, gracefully handling noisy Potential Energy Surfaces (PES) and corrupted trajectory files.

## Features
* **End-to-End Automation:** Seamlessly run DMF, TS optimization, IRC, and VIB analysis from simple reactant/product inputs.
* **Flexible Backends:** Powered by ASE, supporting both high-fidelity First-Principles (e.g., PySCF/gpu4pyscf) and cutting-edge Machine Learning Interatomic Potentials (e.g., `orb_models`, `tblite`).
* **Robust Parsing ("Heartless_read"):** Strictly bypasses ASE's extXYZ metadata parser to prevent crashes from corrupted trajectories or legacy calculator states.
* **Smart Monitoring:** Automatically tracks and plots Heavy-atom RMSD alongside energy profiles to monitor optimization progress.
* **Global Configuration:** Easily tweak calculation parameters and workflow flags via a centralized `default_config.py`.

## Installation

**Prerequisites:** Python 3.8+

Install the required dependencies using `pip`. Note that `ase==3.27.0` is strictly required due to compatibility issues in newer versions.

```bash
pip install -r requirements.txt

```

*Key dependencies include:* `ase`, `pydmf`, `sella`, `orb_models`, `pyscf` (or `gpu4pyscf`), `tblite`, `cupy`.

- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [pydmf](https://github.com/shin1koda/dmf)
- [Sella](https://github.com/zadorlab/sella)
- [orb_models](https://github.com/orbital-materials/orb-models)
- [PySCF](https://pyscf.org/)
- [gpu4pyscf](https://github.com/pyscf/gpu4pyscf)
- [tblite](https://github.com/tblite/tblite)
- [cupy](https://cupy.dev/)
- numpy, pandas, scipy, seaborn
- optional: rmsd for Heavy-RMSD monitoring

## Usage

The toolkit provides specialized entry points depending on your required workflow:

**1. Full Workflow (Path Search -> TS -> IRC -> VIB)**

```bash
# -d: Destination directory for results
# -c: Total charge of the system
# -m: PES method (e.g., orbmol, orbmol+alpb, pyscf)
# -r / -p: Reactant and Product .xyz files
python fircm/fIRCmachine.py -d <dest_dir> -c <charge> -m orbmol -r reactant.xyz -p product.xyz

```

**2. IRC Workflow Only**
*(Requires `INIT_PATH_SEARCH_ON = False` in config)*

```bash
# -i: Input trajectory (.traj) or coordinate (.xyz) file
python fircm/pIRCmachine.py -d <dest_dir> -c <charge> -m orbmol -i input.xyz

```

**3. Vibrational Analysis Only**
*(Requires `INIT_PATH_SEARCH_ON = False` in config)*

```bash
python fircm/sVIBmachine.py -d <dest_dir> -c <charge> -m orbmol -i input.xyz

```

> **Note:** Detailed configuration flags (e.g., toggling specific steps, temperature settings) can be modified directly in `fircm/default_config.py` or overridden in the respective scripts.

## License

[GPL-3.0 License](https://www.google.com/search?q=LICENSE)

## Acknowledgments

Parts of this toolkit were inspired by or adapted from the [ColabReaction](https://github.com/BILAB/ColabReaction) and [redox_benchmark](https://github.com/AM3GroupHub/redox_benchmark) packages. Please respect their respective licenses and copyrights.

---

*For detailed documentation on the architecture, input formats, and advanced configuration, please see the `docs/` directory.*
