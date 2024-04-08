from functools import cached_property

import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulationRunner
from dmb.logging import create_logger

log = create_logger(__file__)


class FakeWormSimulation:

    def __init__(self, input_parameters):

        self.input_parameters = input_parameters

        self.num_execute_worm_calls = 0
        self.num_execute_worm_continue_calls = 0

        self.file_logger = log

        self.record = {"steps": []}

        self.max_density_error = 0.1
        self.uncorrected_max_density_error = 0.2
        self.max_tau_int = 10

        self.valid = False

    async def execute_worm(self):
        self.num_execute_worm_calls += 1

    async def execute_worm_continue(self):
        self.num_execute_worm_continue_calls += 1

    def save_parameters(self):
        pass

    def set_extension_sweeps_in_checkpoints(self, extension_sweeps: int):
        pass

    def plot_observables(self):
        pass

    @cached_property
    def tune_simulation(self):
        return FakeWormSimulation(self.input_parameters)


class TestWormSimulationRunner:

    @staticmethod
    @pytest.fixture(name="fake_worm_simulation", scope="function")
    def fixture_fake_worm_simulation(input_parameters):
        return FakeWormSimulation(input_parameters)

    @staticmethod
    @pytest.mark.asyncio
    async def test_run(fake_worm_simulation: FakeWormSimulation) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.run()
        assert fake_worm_simulation.num_execute_worm_calls == 1
        assert fake_worm_simulation.num_execute_worm_continue_calls == 0

        await runner.run_continue()
        assert fake_worm_simulation.num_execute_worm_calls == 1
        assert fake_worm_simulation.num_execute_worm_continue_calls == 1

    @staticmethod
    @pytest.mark.asyncio
    async def test_run_iterative_until_converged(
        fake_worm_simulation: FakeWormSimulation) -> None:

        fake_worm_simulation.tune_simulation.record = {
            "steps": [{
                "max_density_error": 0.1,
                "Nmeasure2": 42
            }]
        }
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.run_iterative_until_converged()

    @staticmethod
    def test_run_iterative_until_converged_sync(
        fake_worm_simulation: FakeWormSimulation) -> None:

        fake_worm_simulation.tune_simulation.record = {
            "steps": [{
                "max_density_error": 0.1,
                "Nmeasure2": 42
            }]
        }
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync()

    @staticmethod
    def test_tune_nmeasure2_sync(
        fake_worm_simulation: FakeWormSimulation) -> None:

        fake_worm_simulation.tune_simulation.record = {
            "steps": [{
                "max_density_error": 0.1,
                "Nmeasure2": 42
            }]
        }
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.tune_nmeasure2_sync()

    @staticmethod
    @pytest.mark.asyncio
    async def test_tune_nmeasure2(
            fake_worm_simulation: FakeWormSimulation) -> None:

        fake_worm_simulation.tune_simulation.record = {
            "steps": [{
                "max_density_error": 0.1,
                "Nmeasure2": 42,
                "tau_max": 10
            }]
        }
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.tune_nmeasure2()
