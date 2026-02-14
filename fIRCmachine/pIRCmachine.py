import os
import sys
import shutil
import argparse
from time import perf_counter as timepfc
from datetime import datetime
from fIRCmachine import *

# overwrite global variables
g.init_path_search_on = False
g.refine_input_on = False
g.use_sella_in_opt = False
g.tsopt_on = True
g.irc_on = True
g.vib_on = False
g.other_jobs_example_on = False
#g.write_suggestions_on = True
#g.suggestions = []
#g.current_dir = "."
#g.time_log_name = "timing.log"
#
#g.calc_type="orbmol" # orbmol or pyscf or pyscf_fine
#g.device="cuda" # cuda or cpu
#
##g.charge = 0
#g.mult = 1
#g.nmove = 40
#g.update_teval = False
#g.DMF_convergence = "tight"
#
#g.sella_internal = True
#g.irc_dx = 0.08
#
#g.EV_TO_KCAL_MOL = 23.0605
#g.EV_TO_HARTREE = 1 / 27.2114  # ≒ 0.0367493


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run IRC calculations with the input trajectory')
    parser.add_argument("-d", "--directory", type=str, required=True, help="path to the destination folder")
    parser.add_argument("-c", "--charge", type=int, required=True, help="system total charge")
    parser.add_argument("-m", "--method", type=str, required=False, help="calculation method of the PES")
    if g.init_path_search_on:
        parser.add_argument("-r", "--reactant", type=str, required=True, help="inputfile for the reactant .xyz file")
        parser.add_argument("-p", "--product", type=str, required=True, help="inputfile for the product .xyz file")
    else:
        parser.add_argument("-i", "--input", type=str, required=True, default="input.traj", help="input .traj file (ignored if the DMF path search is enabled)")
    parser.add_argument("-rs", "--result", type=str, required=False, default="result.csv", help="resulting dataframe csv")
    args = parser.parse_args()
    
    t_total_start = timepfc()
    if g.init_path_search_on:
        if not os.path.exists(args.directory):
            os.mkdir(args.directory)
        else:
            print(f"canceled: {args.directory} already exists")
            sys.exit()
        shutil.copy(args.reactant, args.directory)
        shutil.copy(args.reactant, args.directory+"/reactant.xyz")
        shutil.copy(args.product, args.directory)
        shutil.copy(args.product, args.directory+"/product.xyz")
    else:
        if not os.path.exists(args.input):
            print(f"canceled: cannot load {args.input}")
            sys.exit()
        if not os.path.exists(args.directory):
            os.mkdir(args.directory)
        input_name = os.path.basename(args.input)
        if not os.path.exists(args.directory+"/"+input_name):
            shutil.copy(args.input, args.directory)
        g.i_traj = input_name
    os.chdir(args.directory)
    g.current_dir = args.directory
    g.charge = args.charge
    if args.method:
        g.calc_type = args.method
    g.r_csv = args.result
    if os.path.exists(g.r_csv):
        print(f"info: {g.r_csv} will be overwritten")
    else:
        print(f"info: {g.r_csv} will be made")
        write_energies(g.i_traj, g.r_csv)
    
    # main
    if g.init_path_search_on:
        init_path_search()
        g.i_traj = "DMF_final.traj" #ignores args.input
    iter_lmax()
    
    # finish
    finishing()
    t_total = timepfc() - t_total_start
    txt = f"* Total_Time            | {t_total:>12.2f} s  *\n"
    write_line(g.time_log_name, txt)
    print(f"finished at: {datetime.now()}")

