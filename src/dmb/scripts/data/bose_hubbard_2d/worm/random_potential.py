"""Script to run worm simulation for 2D Bose-Hubbard model with random
trapping potential."""

import asyncio
import datetime
import os
from typing import Any, Literal

import numpy as np
import typer
from dotenv import load_dotenv

from dmb.data.bose_hubbard_2d.potential import get_random_trapping_potential
from dmb.data.bose_hubbard_2d.worm.simulate import simulate
from dmb.logging import create_logger
from dmb.paths import REPO_DATA_ROOT

log = create_logger(__name__)
app = typer.Typer()


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

    power, V_trap = get_random_trapping_potential(shape=(L, L), mu_offset=mu_offset)

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


@app.command(
    help="Run worm simulation for 2D Bose-Hubbard model with random potential.")
def simulate_random(number_of_samples: int = typer.Option(
    default=1, help="Number of samples to run."),
                    number_of_concurrent_jobs: int = typer.Option(
                        default=1, help="Number of concurrent jobs."),
                    potential_type: Literal["random", "uniform"] = typer.Option(
                        default="random", help="Type of potential."),
                    L_half_min: int = typer.Option(
                        default=4, help="Minimum half side length of lattice."),
                    L_half_max: int = typer.Option(
                        default=10, help="Maximum half side length of lattice."),
                    U_on_min: float = typer.Option(default=0.05,
                                                   help="Minimum U_on value."),
                    U_on_max: float = typer.Option(default=1.0,
                                                   help="Maximum U_on value."),
                    V_nn_z_min: float = typer.Option(default=0.75,
                                                     help="Minimum V_nn_z value."),
                    V_nn_z_max: float = typer.Option(default=1.75,
                                                     help="Maximum V_nn_z value."),
                    mu_offset_min: float = typer.Option(
                        default=-0.5, help="Minimum mu_offset value."),
                    mu_offset_max: float = typer.Option(
                        default=3.0, help="Maximum mu_offset value."),
                    max_density_error: float = typer.Option(
                        default=0.015, help="Maximum density error.")) -> None:

    load_dotenv()

    simulations_dir = REPO_DATA_ROOT / f"bose_hubbard_2d/{potential_type}/simulations"

    semaphore = asyncio.Semaphore(number_of_concurrent_jobs)

    simulation_name_prefix = ""
    if "SLURM_JOB_ID" in os.environ:
        simulation_name_prefix += os.environ["SLURM_JOB_ID"] + "_"

    async def run_sample(sample_idx: int) -> None:
        """Run a single sample."""

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

        async with semaphore:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            await simulate(
                parent_dir=simulations_dir,
                simulation_name=f"{now}_sample_{simulation_name_prefix}{sample_idx}",
                L=L,
                mu=mu,
                t_hop_array=t_hop_array,
                U_on_array=U_on_array,
                V_nn_array=V_nn_array,
                power=power,
                mu_offset=mu_offset,
                run_max_density_error=max_density_error,
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        asyncio.gather(
            *[run_sample(sample_idx) for sample_idx in range(number_of_samples)]))
    loop.close()
