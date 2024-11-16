"""Script to run worm simulation for 2D Bose-Hubbard model with random
trapping potential."""

import argparse
import asyncio
import datetime
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from dmb.data.bose_hubbard_2d.potential import get_random_trapping_potential
from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters, \
    WormSimulation, WormSimulationRunner
from dmb.data.dispatching import auto_create_dispatcher
from dmb.logging import create_logger
from dmb.paths import REPO_DATA_ROOT

log = create_logger(__name__)


def draw_random_config(
    L_half_min: int = 4,
    L_half_max: int = 10,
    U_on_min: float = 0.05,
    U_on_max: float = 1.0,
    V_nn_z_min: float = 0.75,
    V_nn_z_max: float = 1.75,
    mu_offset_min: float = -0.5,
    mu_offset_max: float = 3.0,
) -> tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any,
           float]:
    """Draw random configuration for the simulation."""

    L = np.random.randint(low=L_half_min, high=L_half_max) * 2
    U_on = (np.random.uniform(low=U_on_min, high=U_on_max)**(-1)) * 4
    V_nn = np.random.uniform(low=V_nn_z_min / 4, high=V_nn_z_max / 4) * U_on
    mu_offset = np.random.uniform(low=mu_offset_min, high=mu_offset_max) * U_on

    power, V_trap = get_random_trapping_potential(shape=(L, L),
                                                  mu_offset=mu_offset)

    U_on_array = np.full(shape=(L, L), fill_value=U_on)
    V_nn_array = np.expand_dims(np.full(shape=(L, L), fill_value=V_nn),
                                axis=0).repeat(2, axis=0)
    t_hop_array = np.ones((2, L, L))

    mu = mu_offset + V_trap

    return L, U_on, V_nn, mu, t_hop_array, U_on_array, V_nn_array, power, mu_offset


def draw_uniform_config() -> (tuple[int, float, float, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, Any, float]):
    """Draw uniform configuration for the simulation."""

    L = np.random.randint(low=4, high=10) * 2
    U_on = (np.random.uniform(low=0.05, high=1)**(-1)) * 4
    V_nn = np.random.uniform(low=0.75 / 4, high=1.75 / 4) * U_on
    mu_offset = np.random.uniform(low=-0.5, high=3.0) * U_on

    U_on_array = np.full(shape=(L, L), fill_value=U_on)
    V_nn_array = np.expand_dims(np.full(shape=(L, L), fill_value=V_nn),
                                axis=0).repeat(2, axis=0)
    t_hop_array = np.ones((2, L, L))

    mu = mu_offset * np.ones(shape=(L, L))

    return L, U_on, V_nn, mu, t_hop_array, U_on_array, V_nn_array, None, mu_offset


async def simulate(
    sample_id: int,
    potential_type: str = "random",
    L_half_min: int = 4,
    L_half_max: int = 10,
    U_on_min: float = 0.05,
    U_on_max: float = 1.0,
    V_nn_z_min: float = 0.75,
    V_nn_z_max: float = 1.75,
    mu_offset_min: float = -0.5,
    mu_offset_max: float = 3.0,
    max_density_error: float = 0.015,
) -> None:
    """Run worm simulation for 2D Bose-Hubbard model."""
    if potential_type == "random":
        (
            L,
            _,
            _,
            mu,
            t_hop_array,
            U_on_array,
            V_nn_array,
            power,
            mu_offset,
        ) = draw_random_config(
            L_half_min=L_half_min,
            L_half_max=L_half_max,
            U_on_min=U_on_min,
            U_on_max=U_on_max,
            V_nn_z_min=V_nn_z_min,
            V_nn_z_max=V_nn_z_max,
            mu_offset_min=mu_offset_min,
            mu_offset_max=mu_offset_max,
        )

    elif potential_type == "uniform":
        (
            L,
            _,
            _,
            mu,
            t_hop_array,
            U_on_array,
            V_nn_array,
            power,
            mu_offset,
        ) = draw_uniform_config()
    else:
        raise ValueError(f"Unknown type {potential_type}")

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

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    name_prefix = ""
    # get slurm job id if running on cluster
    if "SLURM_JOB_ID" in os.environ:
        name_prefix += os.environ["SLURM_JOB_ID"] + "_"

    save_dir = (REPO_DATA_ROOT / (f"bose_hubbard_2d/{potential_type}/simulations/"
                                  f"{now}_sample_{name_prefix}{sample_id}"))

    shutil.rmtree(save_dir, ignore_errors=True)

    sim = WormSimulation(
        p,
        save_dir=save_dir,
        executable=Path(os.environ["WORM_MPI_EXECUTABLE"]),
        dispatcher=auto_create_dispatcher(),
    )
    sim_run = WormSimulationRunner(worm_simulation=sim)

    await sim_run.tune_nmeasure2()
    await sim_run.run_iterative_until_converged(
        max_abs_error_threshold=max_density_error)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run worm simulation for 2D BH model")
    parser.add_argument("--number_of_samples",
                        type=int,
                        default=1,
                        help="number of samples to run")
    parser.add_argument("--number_of_concurrent_jobs", type=int, default=1)
    parser.add_argument("--potential_type",
                        type=str,
                        default="random",
                        choices=["random", "uniform"])
    parser.add_argument("--L_half_min",
                        type=int,
                        default=4,
                        help="minimum half side length of lattice")
    parser.add_argument("--L_half_max",
                        type=int,
                        default=10,
                        help="maximum half side length of lattice")
    parser.add_argument("--U_on_min",
                        type=float,
                        default=0.05,
                        help="minimum U_on value")
    parser.add_argument("--U_on_max",
                        type=float,
                        default=1.0,
                        help="maximum U_on value")
    parser.add_argument("--V_nn_z_min",
                        type=float,
                        default=0.75,
                        help="minimum V_nn_z value")
    parser.add_argument("--V_nn_z_max",
                        type=float,
                        default=1.75,
                        help="maximum V_nn_z value")
    parser.add_argument("--mu_offset_min",
                        type=float,
                        default=-0.5,
                        help="minimum mu_offset value")
    parser.add_argument("--mu_offset_max",
                        type=float,
                        default=3.0,
                        help="maximum mu_offset value")
    parser.add_argument(
        "--max_density_error",
        type=float,
        default=0.015,
        help="max density error",
    )

    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.number_of_concurrent_jobs)

    async def run_sample(sample_id: int) -> None:
        """Run a single sample."""
        async with semaphore:
            await simulate(
                sample_id,
                potential_type=args.potential_type,
                L_half_min=args.L_half_min,
                L_half_max=args.L_half_max,
                U_on_min=args.U_on_min,
                U_on_max=args.U_on_max,
                V_nn_z_min=args.V_nn_z_min,
                V_nn_z_max=args.V_nn_z_max,
                mu_offset_min=args.mu_offset_min,
                mu_offset_max=args.mu_offset_max,
                max_density_error=args.max_density_error,
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        asyncio.gather(
            *[run_sample(sample_id) for sample_id in range(args.number_of_samples)]))
    loop.close()
