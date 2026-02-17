import os
import sys
import shutil
import argparse
from time import perf_counter as timepfc
from datetime import datetime
from fIRCmachine import *


# overwrite global variables (all uppercase)
g.INIT_PATH_SEARCH_ON = False
g.REFINE_INPUT_ON = False
g.USE_SELLA_IN_OPT = False
g.TSOPT_ON = False
g.IRC_ON = False
g.VIB_ON = False
g.OTHER_JOBS_EXAMPLE_ON = False
g.WRITE_SUGGESTIONS_ON = False
#g.SUGGESTIONS = []
g.SAVE_FIG_ON = True
g.PRESERVE_CSV_ON = True
#g.CURRENT_DIR = "."
#g.TIME_LOG_NAME = "timing.log"
#g.CALC_TYPE = "orbmol" # orbmol or pyscf or pyscf_fine
#g.DEVICE = "cuda" # cuda or cpu
#g.MULT = 1
#g.NMOVE = 40
#g.UPDATE_TEVAL = False
#g.DMF_CONVERGENCE = "tight"
#g.SELLA_INTERNAL = True
#g.IRC_DX = 0.08
#g.EV_TO_KCAL_MOL = 23.0605
#g.EV_TO_HARTREE = 1 / 27.2114  # ≒ 0.0367493


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run IRC calculations with the input trajectory')
    parser.add_argument("-d", "--directory", type=str, required=True, help="path to the destination folder")
    parser.add_argument("-c", "--charge", type=int, required=True, help="system total charge")
    parser.add_argument("-m", "--method", type=str, required=False, default="orbmol", help="calculation method of the PES")
    if g.INIT_PATH_SEARCH_ON:
        parser.add_argument("-r", "--reactant", type=str, required=True, help="inputfile for the reactant .xyz file")
        parser.add_argument("-p", "--product", type=str, required=True, help="inputfile for the product .xyz file")
    else:
        parser.add_argument("-i", "--input", type=str, required=True, default="input.traj", help="input .traj or .xyz file")
    parser.add_argument("-rs", "--result", type=str, required=False, default="result.csv", help="resulting dataframe .csv file")
    args = parser.parse_args()
    
    t_total_start = timepfc()
    if g.INIT_PATH_SEARCH_ON:
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
        g.I_TRAJ = input_name
    os.chdir(args.directory)
    g.CURRENT_DIR = args.directory
    g.CHARGE = args.charge
    g.CALC_TYPE = args.method
    g.R_CSV = args.result
    if os.path.exists(g.R_CSV):
        print(f"info: {g.R_CSV} will be overwritten")
    else:
        print(f"info: {g.R_CSV} will be made")
    
    # main
    if g.INIT_PATH_SEARCH_ON:
        init_path_search()
        g.I_TRAJ = "DMF_final.traj" #ignores args.input
    elif not g.PRESERVE_CSV_ON:
        write_energies(g.I_TRAJ, g.R_CSV)
    iter_lmax()
    
    # finish
    finishing()
    t_total = timepfc() - t_total_start
    txt = f"* Total_Time            | {t_total:>12.2f} s  *\n"
    write_line(g.TIME_LOG_NAME, txt)
    print(f"finished at: {datetime.now()}")

