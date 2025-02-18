"""Script to run worm simulations for the 2D Bose-Hubbard model."""

import datetime
import os
import shutil
from pathlib import Path

import numpy as np

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
    L: list[int],
    ztU: list[float],
    zVU: list[float],
    muU: list[float],
    tolerance_ztU: float = 0.01,
    tolerance_zVU: float = 0.01,
    tolerance_muU: float = 0.01,
    max_density_error: float = 0.015,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Get missing samples from the dataset."""

    # check lists are same length
    if not len(L) == len(ztU) == len(zVU) == len(muU):
        raise ValueError("L, ztU, zVU, muU lists must have same length")

    bh_dataset = BoseHubbard2dDataset(
        dataset_dir_path=dataset_dir,
        transforms=BoseHubbard2dTransforms(),
        sample_filter_strategy=BoseHubbard2dSampleFilterStrategy(
            max_density_error=max_density_error
        ),
    )

    missing_tuples = []
    for L_i, ztU_i, zVU_i, muU_i in zip(L, ztU, zVU, muU):
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
