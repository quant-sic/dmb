from copy import deepcopy
from functools import cached_property

import numpy as np
import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters, \
    WormSimulationRunner
from dmb.data.bose_hubbard_2d.worm.simulation.runner import \
    get_run_iteratively_num_sweeps_values
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
        self.execution_calls = []

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
        self.execution_calls.append("execute_worm")
        return self._return_codes[self.num_execute_worm_calls +
                                  self.num_execute_worm_continue_calls - 1]

    async def execute_worm_continue(self):
        self.num_execute_worm_continue_calls += 1
        self.execution_calls.append("execute_worm_continue")
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
                       np.linspace(0.1, 0.5, num_steps + 1).tolist())

    @staticmethod
    @pytest.fixture(name="max_tau_ints", scope="function")
    def fixture_max_tau_ints(request, num_steps) -> list[int]:
        return getattr(request, "param",
                       np.linspace(10, 50, num_steps + 1).astype(int).tolist())

    @staticmethod
    @pytest.fixture(name="validities", scope="function")
    def fixture_validities(request, num_steps: int) -> list[bool]:
        return getattr(request, "param", [True] * (num_steps + 1))

    @staticmethod
    @pytest.fixture(name="return_codes", scope="function")
    def fixture_return_codes(request, num_steps: int) -> list[ReturnCode]:
        return getattr(request, "param",
                       [ReturnCode.SUCCESS] * (num_steps + 1))

    @staticmethod
    @pytest.fixture(name="record_steps", scope="function")
    def fixture_record_steps(request, num_steps: int) -> list[dict]:
        return getattr(request, "param", [{
            "max_density_error": 0.1,
            "tau_max": 10,
            "Nmeasure2": 42
        }] * (num_steps + 1))

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
            max_density_errors: list[float], max_tau_ints: list[int]) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            Nmeasure2=42, num_sweep_increments=num_steps)
        assert all(
            key in fake_worm_simulation.record["steps"][step_idx] for key in
            ["sweeps", "error", "elapsed_time", "tau_max", "num_nmeasure2"]
            for step_idx in range(num_steps))

        for step_idx in range(num_steps):
            assert fake_worm_simulation.record["steps"][step_idx][
                "error"] == max_density_errors[step_idx + 1]
            assert fake_worm_simulation.record["steps"][step_idx][
                "tau_max"] == max_tau_ints[step_idx + 1]

    @staticmethod
    @pytest.mark.parametrize("num_steps", [3])
    @pytest.mark.parametrize("validities",
                             [[True, False, True], [True, True, False],
                              [False, True, True], [False, False, False]])
    def test_run_iterative_correct_run_calls(
            fake_worm_simulation: FakeWormSimulation, num_steps: int,
            validities: list[bool]) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            Nmeasure2=42, num_sweep_increments=num_steps)

        execution_calls = [
            "execute_worm_continue" if (idx > 0 and valid) else "execute_worm"
            for idx, valid in enumerate(validities)
        ]
        assert fake_worm_simulation.execution_calls == execution_calls
        assert fake_worm_simulation.execution_calls[0] == "execute_worm"

    @staticmethod
    @pytest.mark.parametrize("num_steps", [3], indirect=True)
    @pytest.mark.parametrize("validities",
                             [[True, False, True], [True, True, False],
                              [False, True, True], [False, False, False]],
                             indirect=True)
    def test_run_teratively_input_parameters(
            fake_worm_simulation: FakeWormSimulation, validities: list[bool],
            num_steps: int) -> None:

        Nmeasure2 = 123
        max_num_measurements_per_nmeasure2 = 100
        min_num_measurements_per_nmeasure2 = 10

        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            Nmeasure2=Nmeasure2,
            num_sweep_increments=num_steps,
            max_num_measurements_per_nmeasure2=
            max_num_measurements_per_nmeasure2,
            min_num_measurements_per_nmeasure2=
            min_num_measurements_per_nmeasure2)

        assert fake_worm_simulation.input_parameters.Nmeasure2 == Nmeasure2

        last_execute_worm_call_index = max(
            np.argwhere([
                False if (idx > 0 and valid) else True
                for idx, valid in enumerate(validities)
            ])).item()

        num_sweeps = get_run_iteratively_num_sweeps_values(
            Nmeasure2=Nmeasure2,
            min_num_measurements_per_nmeasure2=10,
            max_num_measurements_per_nmeasure2=
            max_num_measurements_per_nmeasure2,
            num_sweep_increments=num_steps,
        )
        assert fake_worm_simulation.input_parameters.sweeps == num_sweeps[
            last_execute_worm_call_index]

    @staticmethod
    @pytest.mark.parametrize("num_steps", [10], indirect=True)
    @pytest.mark.parametrize("max_density_errors", [
        [1.0, None, 0.5, 0.25, 0.1],
        [1.0, 0.75, 0.5, 0.25, 0.18, 0.18, None, None, 0.5, 0.3, None],
        [1.0, 0.1, 0.5, 0.25, 0.18, 0.18, 0.18],
    ])
    def test_run_iterative_break_iff_max_density_error_reached(
            fake_worm_simulation: FakeWormSimulation,
            num_steps: int,
            max_density_errors: list[float],
            max_abs_error_threshold: float = 0.15) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            Nmeasure2=42,
            num_sweep_increments=num_steps,
            max_abs_error_threshold=max_abs_error_threshold)

        error_reached = np.argwhere([
            error < max_abs_error_threshold if error is not None else False
            for error in max_density_errors
        ])

        first_max_density_error_reached_idx = min(error_reached).item() if len(
            error_reached) > 0 else num_steps

        assert len(fake_worm_simulation.execution_calls
                   ) == first_max_density_error_reached_idx
        assert len(fake_worm_simulation.record["steps"]
                   ) == first_max_density_error_reached_idx

    @staticmethod
    @pytest.mark.asyncio
    async def test_tune_nmeasure2(fake_worm_simulation: FakeWormSimulation,
                                  record_steps: list[dict]) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.tune_nmeasure2()

    @staticmethod
    def test_tune_nmeasure2_sync(fake_worm_simulation: FakeWormSimulation,
                                 record_steps: list[dict]) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.tune_nmeasure2_sync()

    @staticmethod
    def test_tune_nmeasure2_worm_simulation_unmodified(
            fake_worm_simulation: FakeWormSimulation) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)

        initial_input_parameters = deepcopy(
            fake_worm_simulation.input_parameters)
        runner.tune_nmeasure2_sync()
        assert fake_worm_simulation.num_execute_worm_calls == 0
        assert fake_worm_simulation.num_execute_worm_continue_calls == 0
        assert fake_worm_simulation.input_parameters == initial_input_parameters

    # @staticmethod
    # @pytest.mark.asyncio
    # aync def test_tune_nmeasure2_record_is_written_correctly(
    #         fake_worm_simulation: FakeWormSimulation, record_steps: list[dict],
    #         num_steps: int) -> None:
    #     runner = WormSimulationRunner(fake_worm_simulation)
    #     await runner.tune_nmeasure2(
    #         num_sweep_increments=num_steps,
    #         Nmeasure2=42,
    #     )
    #     assert all(key in fake_worm_simulation.tune_simulation.record["steps"]
    #                [step_idx] for key in [
    #                    "sweeps", "thermalization", "Nmeasure2", "tau_max",
    #                    "max_density_error"
    #                ] for step_idx in range(num_steps))

    #     for step_idx in range(num_steps):
    #         assert fake_worm_simulation.record["steps"][step_idx][
    #             "max_density_error"] == record_steps[step_idx][
    #                 "max_density_error"]
    #         assert fake_worm_simulation.record["steps"][step_idx][
    #             "tau_max"] == record_steps[step_idx]["tau_max"]

    # - test_tune_nmeasure2
    # |-> tune simulation record is written with right keys
    # |-> only start, never continue
    # |-> breaks only when max_tau_int is reached. If max_tau_int is not reached, it should continue until get_tune_nmeasure2_values are exhausted
    # |-> results in nmeasure2 is larger than min_nmeasure2 and smaller than max_nmeasure2
    # |-> plots results
    # |-> saves correct input parameters
    # |-> record is written with right keys
    # |-> record contains right values


# test get_tune_nmeasure2_values
