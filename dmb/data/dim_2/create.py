from dmb.data.dim_2.worm.helpers.sim import WormInputParameters, WormSimulation
import numpy as np
from pathlib import Path
from dmb.utils.io import create_logger
import os
from dmb.data.dim_2.potential import get_random_trapping_potential
from dmb.utils import REPO_ROOT
import subprocess
from pathlib import Path
import shutil
import argparse
from dmb.utils.io import create_logger
from dmb.data.dim_2.helpers import write_sbatch_script
from dmb.utils.paths import REPO_DATA_ROOT
from typing import List
import itertools
import datetime
import json
from dmb.data.dim_2.helpers import call_sbatch_and_wait

log = create_logger(__name__)


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

def get_unfinished_samples(data_dir: Path)->List[Path]:
    # Get all unfinished samples
    unfinished_samples = []

    for sample_dir in data_dir.iterdir():

        try:
            sim = WormSimulation.from_dir(sample_dir,"/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi")
            converged, max_rel_error, n_measurements, tau_max = sim.check_convergence(sim.get_results())

            # extend pars to include the new tau max and max_rel_error to json
            with open(sim.save_dir/"pars.json","r") as f:
                pars = json.load(f)
                pars["tau_max"] = tau_max
                pars["max_rel_error"] = max_rel_error

            log.info("pars: "+str(pars))
            
            with open(sim.save_dir/"pars.json","w") as f:
                json.dump(pars,f)

            if not converged:
                unfinished_samples.append(sample_dir)

        except FileNotFoundError:
            log.info(f"Sample {sample_dir} not finished")
    
    return unfinished_samples


def new_samples(number:int):
    for sample_id in range(number):

        L,U_on,V_nn,mu,t_hop_array,U_on_array,V_nn_array = draw_random_config()

        thermalization = 10000
        sweeps = 100000

        p = WormInputParameters(Lx=L,Ly=L,Nmeasure2=100,t_hop=t_hop_array,U_on=U_on_array,V_nn=V_nn_array,thermalization=thermalization,mu=mu,sweeps=sweeps)

        # get current time up to seconds

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")

        save_dir=Path(REPO_ROOT/f"data/bose_hubbard_2d/{now}_sample_{sample_id}")
        
        shutil.rmtree(save_dir,ignore_errors=True)

        sim = WormSimulation(p,worm_executable="/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi",save_dir=save_dir)

        sim.input_parameters.Nmeasure2 = sim.tune(executable="module load gcc && module load openmpi && srun mpirun /zeropoint/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi")
        sim.input_parameters.sweeps = sim.input_parameters.Nmeasure2 * 10000
        sim.input_parameters.thermalization = sim.input_parameters.Nmeasure2 * 100

        sim.save_parameters()

        #save json file with number of sweeps
        with open(sim.save_dir/"pars.json","w") as f:
            json.dump({"sweeps":sim.input_parameters.sweeps},f)

        yield sim

def prepare_unfinished_samples(unfinished_samples:List[Path],sweeps:int):

    for sample_dir in unfinished_samples:
        sim = WormSimulation.from_dir(sample_dir,"/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi")

        with open(sim.save_dir/"pars.json","r") as f:
            pars = json.load(f)
            current_sweeps = pars["sweeps"]

        print(f"Continuing sample {sample_dir} with {current_sweeps+sweeps} sweeps")
        sim._set_extension_sweeps_in_checkpoints(extension_sweeps=current_sweeps+sweeps*int(sim.input_parameters.Nmeasure2))

        with open(sim.save_dir/"pars.json","w") as f:
            json.dump({"sweeps":current_sweeps+sweeps*int(sim.input_parameters.Nmeasure2)},f)

        # remove output file
        #os.remove(sim.save_dir/"output.h5")

        yield sim

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run worm simulation for 2D BH model')
    parser.add_argument('--number_of_samples', type=int, default=100,
                        help='number of samples to run')
    parser.add_argument('--continue_unfinished', type=bool, default=True,
                        help='continue unfinished runs')
    
    args = parser.parse_args()

    data_dir = REPO_DATA_ROOT/"bose_hubbard_2d"
    data_dir.mkdir(parents=True,exist_ok=True)

    unfinished_samples = get_unfinished_samples(data_dir)

    for type,sim in itertools.chain(zip(itertools.repeat("continued"),prepare_unfinished_samples(unfinished_samples,sweeps=1000)),zip(itertools.repeat("new"),new_samples(args.number_of_samples - len(unfinished_samples)))):
        
        write_sbatch_script(script_path=sim.save_dir/"run.sh",worm_executable_path=Path("/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi"),parameters_path=sim.save_dir/"parameters.ini",pipeout_dir=sim.save_dir/"pipe_out")

        call_sbatch_and_wait(script_path=sim.save_dir/"run.sh")

        # try:
        #     log.info(f"Submitting job for {sim.save_dir}. Type: {type}. N sweeps: {sim.input_parameters.sweeps}")
        #     p = subprocess.run("sbatch "+str(sim.save_dir/"run.sh"),check=True,shell=True,cwd=sim.save_dir,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        # except subprocess.CalledProcessError as e:
        #     log.error(e.stderr.decode("utf-8"))
        #     raise e
