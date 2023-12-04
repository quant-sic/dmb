"""End-to-end integration tests for the worm simulation module."""

import os
from pathlib import Path

import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters, \
    WormSimulation, WormSimulationRunner
from dmb.data.dispatching import AutoDispatcher
from dmb.logging import create_logger

logger = create_logger(__name__)


@pytest.mark.requires_worm
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ee_run_iterative(
    input_parameters: WormInputParameters,
    tmp_path: Path,
) -> None:
    """Test the iterative running of the worm simulation."""

    simulation = WormSimulation(
        input_parameters,
        save_dir=tmp_path,
        dispatcher=AutoDispatcher(),
        executable=os.environ["WORM_MPI_EXECUTABLE"],
    )

    max_density_error = 0.25

    runner = WormSimulationRunner(simulation)
    await runner.run_iterative_until_converged(
        max_num_measurements_per_nmeasure2=5000,
        min_num_measurements_per_nmeasure2=500,
        num_sweep_increments=10,
        sweeps_to_thermalization_ratio=5,
        max_abs_error_threshold=max_density_error,
        Nmeasure2=50,
    )

    assert (tmp_path / "output.h5").exists()
    assert simulation.record["steps"][-1]["error"] < max_density_error
    assert simulation.valid


@pytest.mark.requires_worm
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ee_tune_nmeasure2(
    input_parameters: WormInputParameters,
    tmp_path: Path,
) -> None:
    """Test the tuning of the nmeasure2 parameter of the worm simulation."""

    simulation = WormSimulation(
        input_parameters,
        save_dir=tmp_path,
        dispatcher=AutoDispatcher(),
        executable=os.environ["WORM_MPI_EXECUTABLE"],
    )

    runner = WormSimulationRunner(simulation)

    tau_max_threshold = 15
    await runner.tune_nmeasure2(
        max_nmeasure2=100,
        min_nmeasure2=10,
        num_measurements_per_nmeasure2=2500,
        tau_threshold=tau_max_threshold,
        step_size_multiplication_factor=1.8,
    )

    assert (simulation.tune_simulation.save_dir / "output.h5").exists()
    assert simulation.tune_simulation.record["steps"][-1][
        "tau_max"] < tau_max_threshold
    assert simulation.tune_simulation.valid


@pytest.mark.requires_worm
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ee_run_combination(
    input_parameters: WormInputParameters,
    tmp_path: Path,
) -> None:
    """Test the combination of tuning and running the worm simulation."""

    simulation = WormSimulation(
        input_parameters,
        save_dir=tmp_path,
        dispatcher=AutoDispatcher(),
        executable=os.environ["WORM_MPI_EXECUTABLE"],
    )

    runner = WormSimulationRunner(simulation)
    max_density_error = 0.25

    await runner.tune_nmeasure2(
        max_nmeasure2=100,
        min_nmeasure2=10,
        num_measurements_per_nmeasure2=2500,
        tau_threshold=15,
        step_size_multiplication_factor=1.8,
    )

    await runner.run_iterative_until_converged(
        max_num_measurements_per_nmeasure2=5000,
        min_num_measurements_per_nmeasure2=500,
        num_sweep_increments=10,
        sweeps_to_thermalization_ratio=5,
        max_abs_error_threshold=max_density_error,
        Nmeasure2=None,
    )

    assert (tmp_path / "output.h5").exists()
    assert simulation.max_density_error < max_density_error
    assert simulation.valid
