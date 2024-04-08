from functools import cached_property

import numpy as np
import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters, \
    WormSimulationRunner
from dmb.data.dispatching import ReturnCode
from dmb.logging import create_logger

log = create_logger(__file__)


class FakeWormSimulation:

    def __init__(self, input_parameters, max_density_errors: list[float],
                 max_tau_ints: list[int], validities: list[bool],
                 return_codes: list[ReturnCode]):

        self.input_parameters = input_parameters

        self.num_execute_worm_calls = 0
        self.num_execute_worm_continue_calls = 0

        self.file_logger = log

        self._max_density_errors = max_density_errors
        self._max_tau_ints = max_tau_ints
        self._validities = validities
        self._return_codes = return_codes

        self.uncorrected_max_density_error = None

        self.record = {"steps": []}

    @property
    def max_density_error(self):
        return self._max_density_errors[self.num_execute_worm_calls +
                                        self.num_execute_worm_continue_calls]

    @property
    def max_tau_int(self):
        return self._max_tau_ints[self.num_execute_worm_calls +
                                  self.num_execute_worm_continue_calls]

    @property
    def valid(self):
        return self._validities[self.num_execute_worm_calls +
                                self.num_execute_worm_continue_calls]

    async def execute_worm(self):
        self.num_execute_worm_calls += 1
        return self._return_codes[self.num_execute_worm_calls +
                                  self.num_execute_worm_continue_calls - 1]

    async def execute_worm_continue(self):
        self.num_execute_worm_continue_calls += 1
        return self._return_codes[self.num_execute_worm_calls +
                                  self.num_execute_worm_continue_calls - 1]

    def save_parameters(self):
        pass

    def set_extension_sweeps_in_checkpoints(self, extension_sweeps: int):
        pass

    def plot_observables(self):
        pass

    @cached_property
    def tune_simulation(self):
        return FakeWormSimulation(input_parameters=self.input_parameters,
                                  max_density_errors=self._max_density_errors,
                                  max_tau_ints=self._max_tau_ints,
                                  validities=self._validities,
                                  return_codes=self._return_codes)


class TestWormSimulationRunner:

    @staticmethod
    @pytest.fixture(name="num_steps", scope="function")
    def fixture_num_steps(request) -> int:
        return getattr(request, "param", 10)

    @staticmethod
    @pytest.fixture(name="max_density_errors", scope="function")
    def fixture_max_density_errors(request, num_steps) -> list[float]:
        return getattr(request, "param",
                       np.linspace(0.1, 0.5, num_steps+1).tolist())

    @staticmethod
    @pytest.fixture(name="max_tau_ints", scope="function")
    def fixture_max_tau_ints(request, num_steps) -> list[int]:
        return getattr(request, "param",
                       np.linspace(10, 50, num_steps+1).astype(int).tolist())

    @staticmethod
    @pytest.fixture(name="validities", scope="function")
    def fixture_validities(request, num_steps) -> list[bool]:
        return getattr(request, "param", [True] * (num_steps+1))

    @staticmethod
    @pytest.fixture(name="return_codes", scope="function")
    def fixture_return_codes(request, num_steps: int) -> list[ReturnCode]:
        return getattr(request, "param", [ReturnCode.SUCCESS] * (num_steps+1))

    @staticmethod
    @pytest.fixture(name="record_steps", scope="function")
    def fixture_record_steps(request, num_steps: int) -> list[dict]:
        return getattr(request, "param", [{
            "max_density_error": 0.1,
            "tau_max": 10,
            "Nmeasure2": 42
        }] * (num_steps+1))

    @staticmethod
    @pytest.fixture(name="fake_worm_simulation", scope="function")
    def fixture_fake_worm_simulation(
            input_parameters: WormInputParameters,
            max_density_errors: list[float], max_tau_ints: list[int],
            validities: list[bool],
            return_codes: list[ReturnCode]) -> FakeWormSimulation:
        return FakeWormSimulation(input_parameters=input_parameters,
                                  max_density_errors=max_density_errors,
                                  max_tau_ints=max_tau_ints,
                                  validities=validities,
                                  return_codes=return_codes)

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
            fake_worm_simulation: FakeWormSimulation, record_steps: list[dict],
            num_steps: int) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.run_iterative_until_converged(
            num_sweep_increments=num_steps)

    @staticmethod
    def test_run_iterative_until_converged_sync(
            fake_worm_simulation: FakeWormSimulation, record_steps: list[dict],
            num_steps: int) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            num_sweep_increments=num_steps)

    # required tests:
    # - test_run_iterative_until_converged
    # |-> record is written with right keys
    # |-> start with run. if never success, never continue, or continue after success
    # |-> breaks only when max_density_error is reached. If max_density_error is not reached, it should continue until num_sweep_increments

    @staticmethod
    def test_run_iterative_raises_when_nmeasure2_not_set(
            fake_worm_simulation: FakeWormSimulation, num_steps: int) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        with pytest.raises(ValueError, match="Nmeasure2"):
            runner.run_iterative_until_converged_sync()

        runner.run_iterative_until_converged_sync(
            Nmeasure2=42, num_sweep_increments=num_steps - 1)

    @staticmethod
    def test_run_iterative_record_is_written_correctly(
            fake_worm_simulation: FakeWormSimulation, num_steps: int,
            max_density_errors: list[float], max_tau_ints: list[int],
            validities: list[bool]) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            Nmeasure2=42, num_sweep_increments=num_steps)
        assert all(
            key in fake_worm_simulation.record["steps"][step_idx] for key in
            ["sweeps", "error", "elapsed_time", "tau_max", "num_nmeasure2"]
            for step_idx in range(num_steps))

        for step_idx in range(num_steps):
            assert fake_worm_simulation.record["steps"][step_idx][
                "error"] == max_density_errors[step_idx+1]
            assert fake_worm_simulation.record["steps"][step_idx][
                "tau_max"] == max_tau_ints[step_idx+1]

    @staticmethod
    def test_tune_nmeasure2_sync(fake_worm_simulation: FakeWormSimulation,
                                 record_steps: list[dict]) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.tune_nmeasure2_sync()

    @staticmethod
    @pytest.mark.asyncio
    async def test_tune_nmeasure2(fake_worm_simulation: FakeWormSimulation,
                                  record_steps: list[dict]) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.tune_nmeasure2()

    # - test_tune_nmeasure2
    # |-> worm simulation unmodified
    # |-> tune simulation record is written with right keys
    # |-> only start, never continue
    # |-> breaks only when max_tau_int is reached. If max_tau_int is not reached, it should continue until get_tune_nmeasure2_values are exhausted
    # |-> results in nmeasure2 is larger than min_nmeasure2 and smaller than max_nmeasure2
    # |-> plots results
    # |-> saves correct input parameters
    # |-> record is written with right keys
    # |-> record contains right values


# test get_tune_nmeasure2_values
