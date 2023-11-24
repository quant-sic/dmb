from pathlib import Path
import os
from pathlib import Path
import os
import subprocess
from dmb.utils.io import create_logger
import time
import numpy as np
import subprocess
from dmb.utils.io import create_logger
import asyncio

log = create_logger(__name__)


def write_sbatch_script(
    script_path: Path,
    worm_executable_path: Path,
    parameters_path: Path,
    pipeout_dir: Path,
):
    script_path.parent.mkdir(exist_ok=True, parents=True)
    pipeout_dir.mkdir(exist_ok=True, parents=True)

    with open(script_path, "w") as script_file:
        # write lines
        script_file.write("#!/bin/bash -l\n")
        script_file.write("#SBATCH --job-name=worm\n")

        script_file.write("#SBATCH --output=" + str(pipeout_dir) + "/%j.out\n")
        script_file.write("#SBATCH --error=" + str(pipeout_dir) + "/%j.err\n")

        # randomly choose partition in (standard,highfreq)
        # if np.random.rand() < 0.5:
        #     script_file.write("#SBATCH --partition=highfreq\n")
        # else:
        script_file.write("#SBATCH --partition=standard\n")

        script_file.write("#SBATCH --time=10:00:00\n")
        script_file.write("#SBATCH --nodes=1\n")
        script_file.write("#SBATCH --ntasks-per-node=2\n")
        script_file.write("#SBATCH --cpus-per-task=1\n")
        script_file.write("#SBATCH --mem=2G\n")

        script_file.write("module load gcc\n")
        script_file.write("module load openmpi\n")
        script_file.write("module load boost\n")

        script_file.write(
            "export MPIRUN_OPTIONS='--bind-to core --map-by socket:PE=${SLURM_CPUS_PER_TASK} -report-bindings'\n"
        )
        script_file.write("export TMPDIR=/tmp\n")

        script_file.write(
            "mpirun " + str(worm_executable_path) + " " + str(parameters_path) + "\n"
        )

    os.chmod(script_path, 0o755)


async def call_sbatch_and_wait(script_path: Path, timeout=60 * 60 * 6):
    try:
        p = subprocess.run(
            "sbatch " + str(script_path),
            check=True,
            shell=True,
            cwd=script_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        log.error(e.stderr.decode("utf-8"))
        raise e

    job_id = p.stdout.decode("utf-8").strip().split()[-1]
    log.debug(f"Submitted job {job_id}")

    # wait for job to finish with added timeout
    start_time = time.time()
    while True:
        p = subprocess.run(
            "squeue -j " + job_id,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if len(p.stdout.decode("utf-8").split("\n")) == 2:
            break
        await asyncio.sleep(1)
        if time.time() - start_time > timeout:
            raise TimeoutError("Job did not finish in time")

    log.debug(f"Job {job_id} finished")

    # check if job was successful


def check_if_slurm_is_installed_and_running():
    try:
        subprocess.run("sinfo", check=True, stdout=subprocess.PIPE)
    except FileNotFoundError:
        log.debug("Slurm is not installed.")
        return False

    except subprocess.CalledProcessError:
        log.debug("Slurm is installed but sinfo is not working.")
        return False

    return True
