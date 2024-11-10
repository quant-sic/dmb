from __future__ import annotations

import logging
from copy import deepcopy
from functools import cached_property
from pathlib import Path

import numpy as np
import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import (
    WormInputParameters,
    WormSimulationRunner,
)
from dmb.data.bose_hubbard_2d.worm.simulation.observables import SimulationObservables
from dmb.data.bose_hubbard_2d.worm.simulation.runner import (
    get_run_iteratively_num_sweeps_values,
    get_tune_nmeasure2_values,
)
from dmb.data.bose_hubbard_2d.worm.simulation.sim import WormSimulationInterface
from dmb.data.dispatching import ReturnCode
from dmb.logging import create_logger
from tests.data.bose_hubbard_2d.worm.simulation.test_observables import \
    FakeWormOutput

log = create_logger(__file__)


class FakeWormSimulation(WormSimulationInterface):

    def __init__(
        self,
        input_parameters: WormInputParameters,
        max_density_errors: list[float],
        max_tau_ints: list[int],
        validities: list[bool],
        return_codes: list[ReturnCode],
    ) -> None:

        self._input_parameters = input_parameters
        self._file_logger = log

        self.num_execute_worm_calls = 0
        self.num_execute_worm_continue_calls = 0
        self.execution_calls: list[str] = []

        self.plots_calls: list[str] = []

        self._max_density_errors = max_density_errors
        self._max_tau_ints = max_tau_ints
        self._validities = validities
        self._return_codes = return_codes

        self._record: dict[str, list] = {"steps": []}

    @property
    def record(self) -> dict[str, list]:
        return self._record

    @record.setter
    def record(self, record: dict[str, list]) -> None:
        self._record = record

    @property
    def input_parameters(self) -> WormInputParameters:
        return self._input_parameters

    @input_parameters.setter
    def input_parameters(self, input_parameters: WormInputParameters) -> None:
        self._input_parameters = input_parameters

    @property
    def file_logger(self) -> logging.Logger:
        return self._file_logger

    @file_logger.setter
    def file_logger(self, file_logger: logging.Logger) -> None:
        self._file_logger = file_logger

    @property
    def output(self) -> FakeWormOutput:
        return FakeWormOutput(np.random.rand(100, 16, 16))

    @property
    def observables(self) -> SimulationObservables:
        return SimulationObservables(self.output)

    @property
    def uncorrected_max_density_error(self) -> float | None:
        return None

    @property
    def max_density_error(self) -> float:
        return self._max_density_errors[self.num_execute_worm_calls +
                                        self.num_execute_worm_continue_calls]

    @property
    def max_tau_int(self) -> int:
        return self._max_tau_ints[self.num_execute_worm_calls +
                                  self.num_execute_worm_continue_calls]

    @property
    def valid(self) -> bool:
        return self._validities[self.num_execute_worm_calls +
                                self.num_execute_worm_continue_calls]

    async def execute_worm(self, input_file_path: Path | None = None) -> ReturnCode:
        self.num_execute_worm_calls += 1
        self.execution_calls.append("execute_worm")
        return self._return_codes[self.num_execute_worm_calls +
                                  self.num_execute_worm_continue_calls - 1]

    async def execute_worm_continue(self) -> ReturnCode:
        self.num_execute_worm_continue_calls += 1
        self.execution_calls.append("execute_worm_continue")
        return self._return_codes[self.num_execute_worm_calls +
                                  self.num_execute_worm_continue_calls - 1]

    def save_parameters(self) -> None:
        pass

    def set_extension_sweeps_in_checkpoints(self, extension_sweeps: int) -> None:
        pass

    def plot_observables(
            self,
            observable_names: dict[str, list[str]] = {"primary": ["density"]}) -> None:
        self.plots_calls.append("plot_observables")

    @cached_property
    def tune_simulation(self) -> FakeWormSimulation:
        return FakeWormSimulation(
            input_parameters=self.input_parameters,
            max_density_errors=self._max_density_errors,
            max_tau_ints=self._max_tau_ints,
            validities=self._validities,
            return_codes=self._return_codes,
        )

    def plot_phase_diagram_inputs(self) -> None:
        pass

    def plot_inputs(self) -> None:
        pass


class TestWormSimulationRunner:

    @staticmethod
    @pytest.fixture(name="num_steps", scope="function")
    def fixture_num_steps(request: pytest.FixtureRequest) -> int:
        return getattr(request, "param", 10)

    @staticmethod
    @pytest.fixture(name="max_density_errors", scope="function")
    def fixture_max_density_errors(request: pytest.FixtureRequest,
                                   num_steps: int) -> list[float]:
        default_max_density_errors: list[float] = np.linspace(0.1, 0.5,
                                                              num_steps + 1).tolist()
        return getattr(request, "param", default_max_density_errors)

    @staticmethod
    @pytest.fixture(name="max_tau_ints", scope="function")
    def fixture_max_tau_ints(request: pytest.FixtureRequest,
                             num_steps: int) -> list[int]:
        default_max_tau_ints: list[int] = (np.linspace(10, 50, num_steps +
                                                       1).astype(int).tolist())
        return getattr(request, "param", default_max_tau_ints)

    @staticmethod
    @pytest.fixture(name="validities", scope="function")
    def fixture_validities(request: pytest.FixtureRequest,
                           num_steps: int) -> list[bool]:
        return getattr(request, "param", [True] * (num_steps + 1))

    @staticmethod
    @pytest.fixture(name="return_codes", scope="function")
    def fixture_return_codes(request: pytest.FixtureRequest,
                             num_steps: int) -> list[ReturnCode]:
        return getattr(request, "param", [ReturnCode.SUCCESS] * (num_steps + 1))

    @staticmethod
    @pytest.fixture(name="record_steps", scope="function")
    def fixture_record_steps(request: pytest.FixtureRequest,
                             num_steps: int) -> list[dict]:
        return getattr(
            request,
            "param",
            [{
                "max_density_error": 0.1,
                "tau_max": 10,
                "Nmeasure2": 42
            }] * (num_steps + 1),
        )

    @staticmethod
    @pytest.fixture(name="fake_worm_simulation", scope="function")
    def fixture_fake_worm_simulation(
        input_parameters: WormInputParameters,
        max_density_errors: list[float],
        max_tau_ints: list[int],
        validities: list[bool],
        return_codes: list[ReturnCode],
    ) -> FakeWormSimulation:
        return FakeWormSimulation(
            input_parameters=input_parameters,
            max_density_errors=max_density_errors,
            max_tau_ints=max_tau_ints,
            validities=validities,
            return_codes=return_codes,
        )

    @staticmethod
    @pytest.fixture(name="runner", scope="function")
    def fixture_runner(
        fake_worm_simulation: FakeWormSimulation, ) -> WormSimulationRunner:
        return WormSimulationRunner(fake_worm_simulation)

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
        fake_worm_simulation: FakeWormSimulation,
        record_steps: list[dict],
        num_steps: int,
    ) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.run_iterative_until_converged(num_sweep_increments=num_steps)

    @staticmethod
    def test_run_iterative_until_converged_sync(
        fake_worm_simulation: FakeWormSimulation,
        record_steps: list[dict],
        num_steps: int,
    ) -> None:

        fake_worm_simulation.tune_simulation.record = {"steps": record_steps}
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(num_sweep_increments=num_steps)

    @staticmethod
    def test_run_iterative_raises_when_nmeasure2_not_set(
            fake_worm_simulation: FakeWormSimulation, num_steps: int) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        with pytest.raises(ValueError, match="Nmeasure2"):
            runner.run_iterative_until_converged_sync()

        runner.run_iterative_until_converged_sync(Nmeasure2=42,
                                                  num_sweep_increments=num_steps - 1)

    @staticmethod
    def test_run_iterative_record_is_written_correctly(
        fake_worm_simulation: FakeWormSimulation,
        num_steps: int,
        max_density_errors: list[float],
        max_tau_ints: list[int],
    ) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(Nmeasure2=42,
                                                  num_sweep_increments=num_steps)
        assert all(
            key in fake_worm_simulation.record["steps"][step_idx]
            for key in ["sweeps", "error", "elapsed_time", "tau_max", "num_nmeasure2"]
            for step_idx in range(num_steps))

        for step_idx in range(num_steps):
            assert (fake_worm_simulation.record["steps"][step_idx]["error"] ==
                    max_density_errors[step_idx + 1])
            assert (fake_worm_simulation.record["steps"][step_idx]["tau_max"] ==
                    max_tau_ints[step_idx + 1])

    @staticmethod
    @pytest.mark.parametrize("num_steps", [3])
    @pytest.mark.parametrize(
        "validities",
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
            [False, False, False],
        ],
    )
    def test_run_iterative_correct_run_calls(fake_worm_simulation: FakeWormSimulation,
                                             num_steps: int,
                                             validities: list[bool]) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(Nmeasure2=42,
                                                  num_sweep_increments=num_steps)

        execution_calls = [
            "execute_worm_continue" if (idx > 0 and valid) else "execute_worm"
            for idx, valid in enumerate(validities)
        ]
        assert fake_worm_simulation.execution_calls == execution_calls
        assert fake_worm_simulation.execution_calls[0] == "execute_worm"

    @staticmethod
    @pytest.mark.parametrize("num_steps", [3], indirect=True)
    @pytest.mark.parametrize(
        "validities",
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
            [False, False, False],
        ],
        indirect=True,
    )
    def test_run_teratively_input_parameters(fake_worm_simulation: FakeWormSimulation,
                                             validities: list[bool],
                                             num_steps: int) -> None:

        Nmeasure2 = 123
        max_num_measurements_per_nmeasure2 = 100
        min_num_measurements_per_nmeasure2 = 10

        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            Nmeasure2=Nmeasure2,
            num_sweep_increments=num_steps,
            max_num_measurements_per_nmeasure2=max_num_measurements_per_nmeasure2,
            min_num_measurements_per_nmeasure2=min_num_measurements_per_nmeasure2,
        )

        assert fake_worm_simulation.input_parameters.Nmeasure2 == Nmeasure2

        last_execute_worm_call_index = max(
            np.argwhere([
                False if (idx > 0 and valid) else True
                for idx, valid in enumerate(validities)
            ])).item()

        num_sweeps = get_run_iteratively_num_sweeps_values(
            Nmeasure2=Nmeasure2,
            min_num_measurements_per_nmeasure2=10,
            max_num_measurements_per_nmeasure2=max_num_measurements_per_nmeasure2,
            num_sweep_increments=num_steps,
        )
        assert (fake_worm_simulation.input_parameters.sweeps ==
                num_sweeps[last_execute_worm_call_index])

    @staticmethod
    @pytest.mark.parametrize("num_steps", [10], indirect=True)
    @pytest.mark.parametrize(
        "max_density_errors",
        [
            [1.0, None, 0.5, 0.25, 0.1],
            [1.0, 0.75, 0.5, 0.25, 0.18, 0.18, None, None, 0.5, 0.3, None],
            [1.0, 0.1, 0.5, 0.25, 0.18, 0.18, 0.18],
        ],
    )
    def test_run_iterative_break_iff_max_density_error_reached(
        fake_worm_simulation: FakeWormSimulation,
        num_steps: int,
        max_density_errors: list[float],
        max_abs_error_threshold: float = 0.15,
    ) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        runner.run_iterative_until_converged_sync(
            Nmeasure2=42,
            num_sweep_increments=num_steps,
            max_abs_error_threshold=max_abs_error_threshold,
        )

        error_reached = np.argwhere([
            error < max_abs_error_threshold if error is not None else False
            for error in max_density_errors
        ])

        first_max_density_error_reached_idx = (min(error_reached).item()
                                               if len(error_reached) > 0 else num_steps)

        assert (len(
            fake_worm_simulation.execution_calls) == first_max_density_error_reached_idx
                )
        assert (len(
            fake_worm_simulation.record["steps"]) == first_max_density_error_reached_idx
                )

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
        fake_worm_simulation: FakeWormSimulation, ) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)

        initial_input_parameters = deepcopy(fake_worm_simulation.input_parameters)
        runner.tune_nmeasure2_sync()
        assert fake_worm_simulation.num_execute_worm_calls == 0
        assert fake_worm_simulation.num_execute_worm_continue_calls == 0
        assert fake_worm_simulation.input_parameters == initial_input_parameters

    @staticmethod
    @pytest.mark.asyncio
    async def test_tune_nmeasure2_record_is_written_correctly(
            fake_worm_simulation: FakeWormSimulation, record_steps: list[dict]) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.tune_nmeasure2(min_nmeasure2=10,
                                    max_nmeasure2=100,
                                    step_size_multiplication_factor=2)

        assert all(key in step for key in [
            "sweeps",
            "thermalization",
            "Nmeasure2",
            "tau_max",
            "max_density_error",
        ] for step in fake_worm_simulation.tune_simulation.record["steps"])

    @staticmethod
    @pytest.mark.asyncio
    async def test_tune_nmeasure2_correct_run_calls(
            fake_worm_simulation: FakeWormSimulation, num_steps: int) -> None:
        runner = WormSimulationRunner(fake_worm_simulation)
        await runner.tune_nmeasure2(min_nmeasure2=10,
                                    max_nmeasure2=1000,
                                    step_size_multiplication_factor=2)

        assert set(
            fake_worm_simulation.tune_simulation.execution_calls) == {"execute_worm"}

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "max_tau_ints",
        [
            [50, 40, 30, 20, 11, 12, 14, 16, 18, 13],
            [50, 40, 11, 5, 2, 1],
            [100, 50, 70, 80, 90, 110, 120],
        ],
    )
    @pytest.mark.parametrize("max_nmeasure2", [567, 123, 1024])
    @pytest.mark.parametrize("tau_threshold", [10, 5])
    @pytest.mark.parametrize("step_size_multiplication_factor", [1.8, 3])
    @pytest.mark.parametrize("min_nmeasure2", [1, 2])
    async def test_tune_nmeasure2_break_iff_tau_max_is_reached(
        runner: WormSimulationRunner,
        fake_worm_simulation: FakeWormSimulation,
        max_tau_ints: list[int],
        max_nmeasure2: int,
        tau_threshold: int,
        step_size_multiplication_factor: float,
        min_nmeasure2: int,
    ) -> None:
        #     # |-> breaks only when max_tau_int is reached. If max_tau_int is not reached, it should continue until get_tune_nmeasure2_values are exhausted
        await runner.tune_nmeasure2(
            min_nmeasure2=min_nmeasure2,
            max_nmeasure2=max_nmeasure2,
            step_size_multiplication_factor=step_size_multiplication_factor,
            tau_threshold=tau_threshold,
        )

        num_nmeasure2_values = get_tune_nmeasure2_values(
            min_nmeasure2=min_nmeasure2,
            max_nmeasure2=max_nmeasure2,
            step_size_multiplication_factor=step_size_multiplication_factor,
        )

        max_tau_int_reached_indices = np.argwhere(
            [tau_max < 10 for tau_max in max_tau_ints])
        if len(max_tau_int_reached_indices) > 0:
            first_max_tau_int_reached_idx = min(max_tau_int_reached_indices).item()
            assert (
                fake_worm_simulation.tune_simulation.record["steps"][-1]["Nmeasure2"]
                <= max_nmeasure2)
            assert (fake_worm_simulation.tune_simulation.record["steps"][-1]["tau_max"]
                    <= tau_threshold)

        else:
            first_max_tau_int_reached_idx = len(max_tau_ints)
            assert (fake_worm_simulation.tune_simulation.record["steps"][-1]
                    ["Nmeasure2"] == num_nmeasure2_values[-1])
            assert (fake_worm_simulation.tune_simulation.record["steps"][-1]["tau_max"]
                    > tau_threshold)

        assert (len(fake_worm_simulation.tune_simulation.execution_calls)
                <= first_max_tau_int_reached_idx)
        assert set(
            fake_worm_simulation.tune_simulation.execution_calls) == {"execute_worm"}

        # results in NMesure2, which adheres to the constraints
        assert (min_nmeasure2 <=
                fake_worm_simulation.tune_simulation.record["steps"][-1]["Nmeasure2"] <=
                max_nmeasure2)

        assert "plot_observables" in fake_worm_simulation.tune_simulation.plots_calls

    # - test_tune_nmeasure2
    # |-> saves correct input parameters
    # |-> record is written with right keys
    # |-> record contains right values


@pytest.mark.parametrize("min_nmeasure2", [10, 1, 5])
@pytest.mark.parametrize("max_nmeasure2", [1000, 100, 200])
@pytest.mark.parametrize("step_size_multiplication_factor", [2, 1.8, 3])
def test_get_tune_nmeasure2_values(
    min_nmeasure2: int,
    max_nmeasure2: int,
    step_size_multiplication_factor: float,
) -> None:
    nmeasure2_values = get_tune_nmeasure2_values(
        min_nmeasure2=min_nmeasure2,
        max_nmeasure2=max_nmeasure2,
        step_size_multiplication_factor=step_size_multiplication_factor,
    )

    assert nmeasure2_values[0] == min_nmeasure2
    assert nmeasure2_values[-1] < max_nmeasure2
    assert (np.diff(nmeasure2_values) >= 0).all()
