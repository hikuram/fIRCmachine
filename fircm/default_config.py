
# Workflow flags
INIT_PATH_SEARCH_ON: bool = True
INIT_RECALC_MODE_ON: bool = False
REFINE_INPUT_ON: bool = True
USE_SELLA_IN_OPT: bool = False
TSOPT_ON: bool = True
IRC_ON: bool = True
PICK_OPTPOINTS_ON: bool = True
OPT_OPTPOINTS_AGAIN_ON: bool = False
VIB_ON: bool = True
REFINE_ENERGY_ON: bool = True
OTHER_JOBS_EXAMPLE_ON: bool = False
CALC_RMSD_ON: bool = True
WRITE_SUGGESTIONS_ON: bool = True
SUGGESTIONS: list = []
SAVE_FIG_ON: bool = True
PRESERVE_CSV_ON: bool = False
CURRENT_DIR: str = "."
TIME_LOG_NAME: str = "timing.log"

# Calculation settings
CALC_TYPE: str = "orbmol"  # "orbmol", "orbmol+alpb", "pyscf", "pyscf_high"
REFINE_CALC_TYPE: str = "pyscf_high"
DEVICE: str = "cuda"  # "cuda" or "cpu"

# TBLite settings for Delta ML approach
# Set to "hybrid" to use GFN1-xTB during DMF and GFN2-xTB for everything else.
TBLITE_METHOD: str = "hybrid"  # "hybrid", "GFN1-xTB", "GFN2-xTB"

# Model settings
MULT: int = 1
THERMO_TEMPERATURE: float = 298.15
THERMO_ATOMOSPHERE: float = 101325.0

# Module-specific settings
# -DMF
NMOVE: int = 40
UPDATE_TEVAL: bool = False
DMF_CONVERGENCE: str = "tight"
# -Sella
SELLA_INTERNAL_AUTO: bool = True
SELLA_INTERNAL: bool = True
IRC_DX_INIT: float = 0.06
IRC_DX_MAX: float = 0.12
IRC_DX_MIN: float = 0.02

# Physical constants
EV_TO_KCAL_MOL: float = 23.0605
EV_TO_HARTREE: float = 1 / 27.2114  # approx. 0.0367493

