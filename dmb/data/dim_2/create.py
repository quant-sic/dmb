from dmb.data.dim_2.worm.helpers.sim import WormInputParameters, WormSimulation
import numpy as np
from pathlib import Path
from dmb.utils.io import create_logger
import os
from dmb.data.dim_2.mu import get_random_trapping_potential

log = create_logger(__name__)

from dmb.utils import REPO_ROOT
import subprocess
from pathlib import Path
import shutil

# subprocess.run(" ".join(["export MPIRUN_OPTIONS='--bind-to core --map-by socket:PE=${SLURM_CPUS_PER_TASK} -report-bindings'","export TMPDIR=/tmp","module load gcc", "&&","module load openmpi", "&&",'srun',"--cpus-per-task=1","--nodes=1","--ntasks-per-node=4","--ntasks=4", 'mpirun','/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi', '/u/bale/paper/worm/runs/test_non_uniform/parameters.ini']),shell=True)



def write_sbatch_script(script_path:Path,worm_executable_path:Path,parameters_path:Path, pipeout_dir:Path):

    script_path.parent.mkdir(exist_ok=True,parents=True)

    with open(script_path,"w") as script_file:

        # write lines
        script_file.write("#!/bin/bash -l\n")
        script_file.write("#SBATCH --job-name=worm\n")

        

        script_file.write("#SBATCH --partition=highfreq\n")

        script_file.write("#SBATCH --time=00:30:00\n")
        script_file.write("#SBATCH --nodes=1\n")
        script_file.write("#SBATCH --ntasks-per-node=4\n")
        script_file.write("#SBATCH --cpus-per-task=1\n")
        script_file.write("#SBATCH --mem=2G\n")

        script_file.write("module load gcc\n")
        script_file.write("module load openmpi\n")
        script_file.write("module load boost\n")

        script_file.write("export MPIRUN_OPTIONS='--bind-to core --map-by socket:PE=${SLURM_CPUS_PER_TASK} -report-bindings'\n")
        script_file.write("export TMPDIR=/tmp\n")

        script_file.write("mpirun "+str(worm_executable_path)+" "+str(parameters_path)+"\n")

    os.chmod(script_path, 0o755)


def draw_random_config():

    L = np.random.randint(low=3,high=8)
    U_on = np.random.uniform(low=4.0,high=80)
    V_nn = np.random.uniform(low=0.75/4,high=1.75/4) * U_on
    mu_offset = np.random.uniform(low=-0.5,high=3.0) * U_on

    power,V_trap = get_random_trapping_potential(shape=(L,L),desired_abs_max=mu_offset/2)
    U_on_array = np.full(shape=(L,L),fill_value=U_on)
    V_nn_array = np.expand_dims(V_nn,axis=0).repeat(2,axis=0)
    t_hop_array = np.ones((2,L,L))

    mu = mu_offset + V_trap

    return L,U_on,V_nn,mu,t_hop_array,U_on_array,V_nn_array    



if __name__ == "__main__":

    number_of_samples = 10


    for sample_id in range(number_of_samples):

        L,U_on,V_nn,mu,t_hop_array,U_on_array,V_nn_array = draw_random_config()

        thermalization = 10000
        sweeps = 1000000
        p = WormInputParameters(Lx=L,Ly=L,Nmeasure2=100,t_hop=t_hop_array,U_on=U_on_array,V_nn=V_nn_array,thermalization=thermalization,mu=mu,sweeps=sweeps)

        save_dir=Path(REPO_ROOT/f"data/bh_2d/{sample_id}")
        
        shutil.rmtree(save_dir,ignore_errors=True)

        sim = WormSimulation(p,worm_executable="/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi",save_dir=save_dir)

        # if not "SLURM_PROCID" in os.environ or os.environ["SLURM_PROCID"] == "0":
        #     log.info("Saving Parameters")
        sim.save_parameters()

        # log.info("Running Simulation")
        # sim.run_until_convergence(tune=False)

        write_sbatch_script(script_path=save_dir/"run.sh",worm_executable_path=Path("/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi"),parameters_path=save_dir/"parameters.ini")

        subprocess.run("sbatch "+str(save_dir/"run.sh"),check=True,shell=True,cwd=save_dir)

