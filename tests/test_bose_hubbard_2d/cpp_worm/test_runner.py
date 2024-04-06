from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from dmb.data.bose_hubbard_2d.cpp_worm.worm import WormInputParameters, \
    WormSimulation, WormSimulationRunner


class FakeSimulation:

    def __init__(self, save_dir: Path) -> None:
        self.record = MagicMock()
        self.save_dir = save_dir
        self.input_parameters = MagicMock()

    @property
    def tune_simulation(self) -> FakeSimulation:
        tune_dir = self.save_dir / "tune"
        tune_dir.mkdir(parents=True, exist_ok=True)
        return FakeSimulation(save_dir=tune_dir)

    def save_parameters(self) -> None:
        pass

    async def execute_worm(self, *args, **kwargs) -> None:
        pass

    @property
    def convergence_stats(self) -> None:
        return {}

    def plot_observables(self) -> None:
        pass

    @property
    def output(self) -> None:
        return MagicMock()

    @property
    def uncorrected_max_density_error(self) -> float:
        return MagicMock()


def test_tune_measure2(tmp_path: Path) -> None:
    sim = FakeSimulation(save_dir=tmp_path / "test_sim")
    sim.tune_simulation

    sim_runner = WormSimulationRunner(worm_simulation=sim)
    sim_runner.tune_nmeasure2_sync()

    assert False


# def test_run_untils_converged
# def test_run_continued
# def test_run
