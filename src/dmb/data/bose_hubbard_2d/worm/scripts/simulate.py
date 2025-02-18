"""Script to run worm simulations for the 2D Bose-Hubbard model."""

import datetime
import itertools
import os
import shutil
from pathlib import Path
from typing import cast

import numpy as np

from dmb.data.bose_hubbard_2d.potential import get_random_trapping_potential
from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2dTransforms
from dmb.data.bose_hubbard_2d.worm.dataset import (
    BoseHubbard2dDataset,
    BoseHubbard2dSampleFilterStrategy,
)
from dmb.data.bose_hubbard_2d.worm.simulation import (
    WormInputParameters,
    WormSimulation,
    WormSimulationRunner,
)
from dmb.data.dispatching import auto_create_dispatcher
from dmb.logging import create_logger

log = create_logger(__name__)


def get_missing_samples(
    dataset_dir: Path,
    L: int | list[int],
    ztU: float | list[float],
    zVU: float | list[float],
    muU: float | list[float],
    tolerance_ztU: float = 0.01,
    tolerance_zVU: float = 0.01,
    tolerance_muU: float = 0.01,
    max_density_error: float = 0.015,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Get missing samples from the dataset."""

    bh_dataset = BoseHubbard2dDataset(
        dataset_dir_path=dataset_dir,
        transforms=BoseHubbard2dTransforms(),
        sample_filter_strategy=BoseHubbard2dSampleFilterStrategy(
            max_density_error=max_density_error,
        ),
    )

    # if lists, they must be of the same length
    if (
        not len(
            {
                len(cast(list, x))
                for x in filter(lambda y: isinstance(y, list), [L, ztU, zVU, muU])
            }
        )
        == 1
    ):
        raise ValueError("Lists must be of the same length")

    if any(not isinstance(x, list) for x in [L, ztU, zVU, muU]):
        if not all(isinstance(x, (int, float)) for x in [L, ztU, zVU, muU]):
            raise ValueError("All inputs must be either lists or scalars")

        L_ = [cast(int, L)]
        ztU_ = [cast(float, ztU)]
        zVU_ = [cast(float, zVU)]
        muU_ = [cast(float, muU)]
    else:
        L_ = cast(list[int], L)
        ztU_ = cast(list[float], ztU)
        zVU_ = cast(list[float], zVU)
        muU_ = cast(list[float], muU)

    missing_tuples = []
    for L_i, ztU_i, zVU_i, muU_i in zip(
        *[
            x if isinstance(x, list) else itertools.cycle((x,))
            for x in [L_, ztU_, zVU_, muU_]
        ]
    ):
        if not bh_dataset.has_phase_diagram_sample(
            L=L_i,
            ztU=ztU_i,
            zVU=zVU_i,
            muU=muU_i,
            ztU_tol=tolerance_ztU,
            zVU_tol=tolerance_zVU,
            muU_tol=tolerance_muU,
        ):
            missing_tuples.append((L_i, ztU_i, zVU_i, muU_i))

    L_out, ztU_out, zVU_out, muU_out = zip(*missing_tuples)

    return L_out, ztU_out, zVU_out, muU_out


def draw_random_config(
    L_half_min: int = 4,
    L_half_max: int = 10,
    U_on_min: float = 0.05,
    U_on_max: float = 1.0,
    V_nn_z_min: float = 0.75,
    V_nn_z_max: float = 1.75,
    mu_offset_min: float = -0.5,
    mu_offset_max: float = 3.0,
) -> tuple[
    int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
]:
    """Draw a random configuration for the 2D Bose-Hubbard model."""

    L = np.random.randint(low=L_half_min, high=L_half_max) * 2
    U_on = (np.random.uniform(low=U_on_min, high=U_on_max) ** (-1)) * 4
    V_nn = np.random.uniform(low=V_nn_z_min / 4, high=V_nn_z_max / 4) * U_on
    mu_offset = np.random.uniform(low=mu_offset_min, high=mu_offset_max) * U_on

    power, V_trap = get_random_trapping_potential(shape=(L, L), mu_offset=mu_offset)

    U_on_array = np.full(shape=(L, L), fill_value=U_on)
    V_nn_array = np.expand_dims(np.full(shape=(L, L), fill_value=V_nn), axis=0).repeat(
        2, axis=0
    )
    t_hop_array = np.ones((2, L, L))

    mu = mu_offset + V_trap

    return L, U_on, V_nn, mu, t_hop_array, U_on_array, V_nn_array, power, mu_offset


def draw_uniform_config() -> (
    tuple[
        int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, None, float
    ]
):
    """Draw a uniform configuration for the 2D Bose-Hubbard model."""

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


async def simulate(
    parent_dir: Path,
    simulation_name: str,
    L: int,
    mu: np.ndarray,
    t_hop_array: np.ndarray,
    U_on_array: np.ndarray,
    V_nn_array: np.ndarray,
    power: float,
    mu_offset: float,
    thermalization: int = 10000,
    sweeps: int = 100000,
    nmeasure2: int = 100,
    tune_tau_max_threshold: int = 10,
    run_max_density_error: float = 0.015,
) -> None:
    """Run worm simulation for 2D Bose-Hubbard model."""

    p = WormInputParameters(
        Lx=L,
        Ly=L,
        Nmeasure2=nmeasure2,
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

    save_dir = parent_dir / f"{now}_sample_{name_prefix}{simulation_name}"

    shutil.rmtree(save_dir, ignore_errors=True)

    sim = WormSimulation(
        p,
        save_dir=save_dir,
        executable=Path(os.environ["WORM_MPI_EXECUTABLE"]),
        dispatcher=auto_create_dispatcher(),
    )
    sim_run = WormSimulationRunner(worm_simulation=sim)

    await sim_run.tune_nmeasure2(tau_threshold=tune_tau_max_threshold)
    await sim_run.run_iterative_until_converged(
        max_abs_error_threshold=run_max_density_error
    )
