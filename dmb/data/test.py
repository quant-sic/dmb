import subprocess
from pathlib import Path

# subprocess.run(" ".join(["export MPIRUN_OPTIONS='--bind-to core --map-by socket:PE=${SLURM_CPUS_PER_TASK} -report-bindings'","export TMPDIR=/tmp","module load gcc", "&&","module load openmpi", "&&",'srun',"--cpus-per-task=1","--nodes=1","--ntasks-per-node=4","--ntasks=4", 'mpirun','/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi', '/u/bale/paper/worm/runs/test_non_uniform/parameters.ini']),shell=True)

def create_sbatch_script(script_path:Path):

    script_path.parent.mkdir(exist_ok=True,parents=True)

    with open(script_path,"w") as script_file:

        