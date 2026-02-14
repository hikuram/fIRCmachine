#
init_path_search_on = True
refine_input_on = True
use_sella_in_opt = False
tsopt_on = True
irc_on = True
vib_on = True
other_jobs_example_on = False
write_suggestions_on = True
suggestions = []
current_dir = "."
time_log_name = "timing.log"

calc_type="orbmol" # orbmol or pyscf or pyscf_fine
device="cuda" # cuda or cpu

#charge = 0
mult = 1
nmove = 40
update_teval = False
DMF_convergence = "tight"

sella_internal = True
irc_dx = 0.08

EV_TO_KCAL_MOL = 23.0605
EV_TO_HARTREE = 1 / 27.2114  # ≒ 0.0367493