import argparse
import asyncio
import datetime
import os
import shutil
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from dmb.data.bose_hubbard_2d.cpp_worm.worm import (
    WormInputParameters,
    WormSimulation,
    WormSimulationRunner,
)
from dmb.data.bose_hubbard_2d.potential import get_random_trapping_potential
from dmb.utils import create_logger

log = create_logger(__name__)


def draw_random_config():
    L = np.random.randint(low=4, high=10) * 2
    U_on = (np.random.uniform(low=0.05, high=1) ** (-1)) * 4
    V_nn = np.random.uniform(low=0.75 / 4, high=1.75 / 4) * U_on
    mu_offset = np.random.uniform(low=-0.5, high=3.0) * U_on

    power, V_trap = get_random_trapping_potential(
        shape=(L, L), desired_abs_max=abs(mu_offset) / 2
    )
    U_on_array = np.full(shape=(L, L), fill_value=U_on)
    V_nn_array = np.expand_dims(np.full(shape=(L, L), fill_value=V_nn), axis=0).repeat(
        2, axis=0
    )
    t_hop_array = np.ones((2, L, L))

    mu = mu_offset + V_trap

    return L, U_on, V_nn, mu, t_hop_array, U_on_array, V_nn_array, power, mu_offset


def draw_uniform_config():
    L = np.random.randint(low=4, high=10) * 2
    U_on = (np.random.uniform(low=0.05, high=1) ** (-1)) * 4
    V_nn = np.random.uniform(low=0.75 / 4, high=1.75 / 4) * U_on
    mu_offset = np.random.uniform(low=-0.5, high=3.0) * U_on

    U_on_array = np.full(shape=(L, L), fill_value=U_on)
    V_nn_array = np.expand_dims(np.full(shape=(L, L), fill_value=V_nn), axis=0).repeat(
        2, axis=0
    )
    t_hop_array = np.ones((2, L, L))

    mu = mu_offset * np.ones(shape=(L, L))

    return L, U_on, V_nn, mu, t_hop_array, U_on_array, V_nn_array, None, mu_offset


# def simulate(sample_id,type="random"):
async def simulate(sample_id, type="random"):
    if type == "random":
        (
            L,
            U_on,
            V_nn,
            mu,
            t_hop_array,
            U_on_array,
            V_nn_array,
            power,
            mu_offset,
        ) = draw_random_config()
    elif type == "uniform":
        (
            L,
            U_on,
            V_nn,
            mu,
            t_hop_array,
            U_on_array,
            V_nn_array,
            power,
            mu_offset,
        ) = draw_uniform_config()
    else:
        raise ValueError(f"Unknown type {type}")

    thermalization = 10000
    sweeps = 100000

    p = WormInputParameters(
        Lx=L,
        Ly=L,
        Nmeasure2=100,
        t_hop=t_hop_array,
        U_on=U_on_array,
        V_nn=V_nn_array,
        thermalization=thermalization,
        mu=mu,
        sweeps=sweeps,
        mu_power=power,
        mu_offset=mu_offset,
    )

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    name_prefix = ""
    # get slurm job id if running on cluster
    if "SLURM_JOB_ID" in os.environ:
        name_prefix += os.environ["SLURM_JOB_ID"] + "_"

    # save_dir=Path(REPO_ROOT/f"data/bose_hubbard_2d/{now}_sample_{sample_id}")
    save_dir = Path(
        f"/ptmp/bale/data/bose_hubbard_2d/{now}_sample_{name_prefix}{sample_id}"
    )

    shutil.rmtree(save_dir, ignore_errors=True)

    sim = WormSimulation(
        p, save_dir=save_dir, worm_executable=os.environ["WORM_MPI_EXECUTABLE"]
    )
    sim_run = WormSimulationRunner(worm_simulation=sim)

    try:
        await sim_run.tune_nmeasure2()
        await sim_run.run_iterative_until_converged()
    except Exception as e:
        log.error(f"Exception occured: {e}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run worm simulation for 2D BH model")
    parser.add_argument(
        "--number_of_samples", type=int, default=1, help="number of samples to run"
    )
    parser.add_argument("--number_of_concurrent_jobs", type=int, default=1)
    parser.add_argument(
        "--type", type=str, default="random", choices=["random", "uniform"]
    )

    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.number_of_concurrent_jobs)

    async def run_sample(sample_id):
        async with semaphore:
            await simulate(sample_id, type=args.type)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *[run_sample(sample_id) for sample_id in range(args.number_of_samples)]
        )
    )
    loop.close()
