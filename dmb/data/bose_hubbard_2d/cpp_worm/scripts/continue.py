import argparse


from dmb.data.bose_hubbard_2d.cpp_worm.worm import WormSimulation, WormSimulationRunner
from dmb.utils import REPO_DATA_ROOT, create_logger
from dotenv import load_dotenv
import os
from pathlib import Path

import asyncio

log = create_logger(__name__)


async def continue_simulation(
    sample_dir: Path,
    target_density_error: float = 0.015,
    tau_threshold: float = 10,
    restart_run: bool = False,
):
    try:
        sim = WormSimulation.from_dir(
            dir_path=sample_dir, worm_executable=os.environ.get("WORM_MPI_EXECUTABLE")
        )
    except OSError as e:
        log.error(f"Exception occured during loading: {e}")
        return

    try:
        if sim.max_density_error <= target_density_error:
            log.info(f"Sample {sample_dir} already converged.")
            return
    except:
        pass

    # check if tuning is necessary
    tuned = False
    try:
        # get tau_max key
        tau_max_keys = list(
            filter(
                lambda k: "tau_max" in k, sim.tune_simulation.record["steps"][-1].keys()
            )
        )
        if len(tau_max_keys) == 0:
            raise Exception("No tau_max key found in record.")
        else:
            tau_max_key = list(tau_max_keys)[0]

        if sim.tune_simulation.record["steps"][-1][tau_max_key] < tau_threshold:
            log.info(f"Sample {sample_dir} already tuned.")
            tuned = True
    except Exception as e:
        log.warning(f"Exception occured during tune check: {e}")

    sim_run = WormSimulationRunner(sim)

    if not tuned:
        log.info(f"Tuning sample {sample_dir}.")
        try:
            await sim_run.tune_nmeasure2()
        except (KeyError, OSError) as e:
            log.error(f"Exception occured during tuning: {e}")

    try:
        await sim_run.run_iterative_until_converged(restart=restart_run)
    except Exception as e:
        log.error(f"Exception occured during run: {e}")


if __name__ == "__main__":
    load_dotenv()

    os.environ["WORM_JOB_NAME"] = "worm_ctd"

    parser = argparse.ArgumentParser(description="Run worm simulation for 2D BH model")
    parser.add_argument("--number_of_concurrent_jobs", type=int, default=1)
    parser.add_argument("--restart_runs", type=bool, default=False)
    parser.add_argument("--target_density_error", type=float, default=0.015)

    args = parser.parse_args()

    # load relevant directories
    dataset_dir = REPO_DATA_ROOT / "bose_hubbard_2d"
    sample_dirs = sorted(dataset_dir.glob("*sample*"))

    semaphore = asyncio.Semaphore(args.number_of_concurrent_jobs)

    async def run_continue_simulation(sample_dir):
        async with semaphore:
            await continue_simulation(
                sample_dir,
                target_density_error=args.target_density_error,
                restart_run=args.restart_runs,
            )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *[run_continue_simulation(sample_dir) for sample_dir in sample_dirs]
        )
    )
    loop.close()
