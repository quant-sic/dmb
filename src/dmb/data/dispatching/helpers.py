"""Helper functions for job submission."""

import asyncio
import subprocess
import time
from enum import Enum
from logging import Logger
from pathlib import Path

from dmb.logging import create_logger


class ReturnCode(Enum):
    """Return codes for job submission."""

    SUCCESS = 0
    FAILURE = 1


logger = create_logger(__name__)


async def call_sbatch_and_wait(
    script_path: Path,
    timeout: int = 48 * 60 * 60,
    logging_instance: Logger = logger,
) -> ReturnCode:
    """Submit a Slurm job and wait for it to finish.

    Args:
        script_path: Path to the script to submit.
        timeout: Timeout in seconds.
        logging_instance: Logger instance.
    """
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
        logging_instance.error(e.stderr.decode("utf-8"))

        return ReturnCode.FAILURE

    job_id = p.stdout.decode("utf-8").strip().split()[-1]
    logging_instance.debug(f"Submitted job {job_id}")

    # wait for job to finish with added timeout
    start_time = time.time()
    try:
        while True:
            process = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                check=True,
            )

            job_state = process.stdout.decode("utf-8").strip()

            if process.returncode != 0:
                logging_instance.debug(
                    f"Error executing squeue: {process.stderr.decode('utf-8')}"
                )
                return ReturnCode.FAILURE

            if job_state not in ("RUNNING", "PENDING"):
                logging_instance.debug(f"Job {job_id} ended {job_state}")
                break

            logging_instance.debug(f"Job {job_id} is {job_state}")
            await asyncio.sleep(1)

            if time.time() - start_time > timeout:
                logging_instance.error(f"Job {job_id} timed out")
                return ReturnCode.FAILURE

    except Exception as e:
        logging_instance.error(f"Error: {e}")
        raise e

    logging_instance.debug(f"Job {job_id} finished")
    return ReturnCode.SUCCESS


def check_if_slurm_is_installed_and_running(
    logging_instance: Logger = logger,
) -> bool:
    """Check if Slurm is installed and running on the system.

    Args:
        logging_instance: Logger instance.

    Returns:
        True if Slurm is installed and running, False otherwise.
    """

    try:
        subprocess.run("sinfo", check=True, stdout=subprocess.PIPE)
    except FileNotFoundError:
        logging_instance.debug("Slurm is not installed.")
        return False

    except subprocess.CalledProcessError:
        logging_instance.debug("Slurm is installed but sinfo is not working.")
        return False

    return True
