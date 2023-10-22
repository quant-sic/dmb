from dmb.data.bose_hubbard_2d.worm.sim import WormInputParameters, WormSimulation
import numpy as np
from pathlib import Path

from dmb.data.bose_hubbard_2d.potential import get_random_trapping_potential
from dmb.utils import REPO_ROOT,REPO_DATA_ROOT
from pathlib import Path
import shutil
import argparse

import datetime
import joblib
import shutil
from dmb.utils.io import ProgressParallel
import gc
from dmb.utils import create_logger

from functools import partial

log = create_logger(__name__)

def continue_simulation(sample_dir):

    sim = WormSimulation.from_dir(sample_dir)

    try:
        sim.run_until_convergence(executable="/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi")
        sim.plot_result()
    except:
        log.error("Simulation failed")
        return

    finally:
        gc.collect()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Run worm simulation for 2D BH model')
    parser.add_argument('--number_of_samples', type=int, default=1,
                        help='number of samples to run')
    parser.add_argument('--number_of_jobs', type=int, default=1,
                        help='number of jobs to run in parallel')
    args = parser.parse_args()

    # load relevant directories

    dataset_dir = REPO_DATA_ROOT / "bose_hubbard_2d"
    sample_dirs = list(dataset_dir.glob("*sample*"))

    # filter finished simulations
    # First attemt. If there is no output.h5 file, the simulation is not finished

    sample_dirs = [sample_dir for sample_dir in sample_dirs if not (sample_dir / "output.h5").exists()]

    print(sorted(list(sample_dirs)))

    # run jobs in parallel
    ProgressParallel(n_jobs=args.number_of_jobs,total=args.number_of_samples,desc="Running Simulations",timeout=999999)(joblib.delayed(continue_simulation)(sample_dir) for sample_dir in sample_dirs)