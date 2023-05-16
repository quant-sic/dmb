from dmb.data.dim_2.worm.helpers.sim import WormInputParameters, WormSimulation
import numpy as np
from pathlib import Path

from dmb.data.dim_2.potential import get_random_trapping_potential
from dmb.utils import REPO_ROOT
from pathlib import Path
import shutil
import argparse

import datetime
import joblib
import shutil
from dmb.utils.io import ProgressParallel
import gc

def draw_random_config():

    L = np.random.randint(low=3,high=8)*2
    U_on = np.random.uniform(low=4.0,high=80)
    V_nn = np.random.uniform(low=0.75/4,high=1.75/4) * U_on
    mu_offset = np.random.uniform(low=-0.5,high=3.0) * U_on

    power,V_trap = get_random_trapping_potential(shape=(L,L),desired_abs_max=mu_offset/2)
    U_on_array = np.full(shape=(L,L),fill_value=U_on)
    V_nn_array = np.expand_dims(np.full(shape=(L,L),fill_value=V_nn),axis=0).repeat(2,axis=0)
    t_hop_array = np.ones((2,L,L))

    mu = mu_offset + V_trap

    return L,U_on,V_nn,mu,t_hop_array,U_on_array,V_nn_array  

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Run worm simulation for 2D BH model')
    parser.add_argument('--number_of_samples', type=int, default=25,
                        help='number of samples to run')
    parser.add_argument('--number_of_jobs', type=int, default=10,
                        help='number of jobs to run in parallel')
    
    args = parser.parse_args()


    def simulate_random(sample_id):
        L,U_on,V_nn,mu,t_hop_array,U_on_array,V_nn_array = draw_random_config()

        thermalization = 10000
        sweeps = 100000

        p = WormInputParameters(Lx=L,Ly=L,Nmeasure2=100,t_hop=t_hop_array,U_on=U_on_array,V_nn=V_nn_array,thermalization=thermalization,mu=mu,sweeps=sweeps)

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_dir=Path(REPO_ROOT/f"data/bose_hubbard_2d/{now}_sample_{sample_id}")
        shutil.rmtree(save_dir,ignore_errors=True)

        sim = WormSimulation(p,save_dir=save_dir)

        sim.save_parameters()
        sim.run_until_convergence(executable="/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi")

        gc.collect()

    
    # run jobs in parallel
    ProgressParallel(n_jobs=args.number_of_jobs,total=args.number_of_samples,desc="Running Simulations")(joblib.delayed(simulate_random)(sample_id) for sample_id in range(args.number_of_samples))