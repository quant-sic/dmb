from pathlib import Path
import os

def write_sbatch_script(script_path:Path,worm_executable_path:Path,parameters_path:Path, pipeout_dir:Path):

    script_path.parent.mkdir(exist_ok=True,parents=True)
    pipeout_dir.mkdir(exist_ok=True,parents=True)
    
    with open(script_path,"w") as script_file:

        # write lines
        script_file.write("#!/bin/bash -l\n")
        script_file.write("#SBATCH --job-name=worm\n")

        script_file.write("#SBATCH --output="+str(pipeout_dir)+"/%j.out\n")
        script_file.write("#SBATCH --error="+str(pipeout_dir)+"/%j.err\n")

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