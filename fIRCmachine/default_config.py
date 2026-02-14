
# Workflow flags
INIT_PATH_SEARCH_ON: bool = True
REFINE_INPUT_ON: bool = True
USE_SELLA_IN_OPT: bool = False
TSOPT_ON: bool = True
IRC_ON: bool = True
VIB_ON: bool = True
OTHER_JOBS_EXAMPLE_ON: bool = False
WRITE_SUGGESTIONS_ON: bool = True
SUGGESTIONS: list = []
CURRENT_DIR: str = "."
TIME_LOG_NAME: str = "timing.log"

# Calculation settings
CALC_TYPE: str = "orbmol"  # "orbmol", "pyscf", "pyscf_fine"
DEVICE: str = "cuda"  # "cuda" or "cpu"

# Physical constants
MULT: int = 1
NMOVE: int = 40
UPDATE_TEVAL: bool = False
DMF_CONVERGENCE: str = "tight"
SELLA_INTERNAL: bool = True
IRC_DX: float = 0.08

EV_TO_KCAL_MOL: float = 23.0605
EV_TO_HARTREE: float = 1 / 27.2114  # ≒ 0.0367493