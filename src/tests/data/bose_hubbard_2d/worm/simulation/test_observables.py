import itertools
from typing import cast

import numpy as np
import pytest

from dmb.data.bose_hubbard_2d.worm.simulation.observables import SimulationObservables
from dmb.data.bose_hubbard_2d.worm.simulation.output import Output


class FakeWormOutput(Output):
    """Fake worm output for testing purposes."""

    def __init__(self, densities: np.ndarray) -> None:
        self._densities = densities

    @property
    def densities(self) -> np.ndarray:
        return self._densities


class TestsSimulationObservables:
    """Tests for the SimulationObservables class."""

    @staticmethod
    @pytest.fixture(scope="class", name="fake_output")
    def fixture_fake_output() -> FakeWormOutput:
        densities = np.random.rand(100, 16, 16)
        return FakeWormOutput(densities)

    @staticmethod
    @pytest.fixture(scope="class", name="observables_keys")
    def fixture_observables_keys(fake_output: FakeWormOutput) -> list[tuple[str, str]]:
        return list(
            itertools.chain.from_iterable(
                list(itertools.product((obs_type,), obs_name))
                for obs_type, obs_name in SimulationObservables(
                    fake_output
                ).observable_names.items()
            )
        )

    @staticmethod
    def test_observable_types_and_shapes(
        fake_output: FakeWormOutput, observables_keys: list[tuple[str, str]]
    ) -> None:
        """Test the types and shapes of the observables."""
        simulation_observables = SimulationObservables(fake_output)

        for obs_type, obs_name in observables_keys:

            def type_check(obj: np.ndarray) -> bool:
                return isinstance(obj, np.ndarray)

            def shape_check(obj: np.ndarray) -> bool:
                return obj.shape in ((16, 16), ())

            expectation_value = simulation_observables.get_expectation_value(
                obs_type, obs_name
            )
            assert expectation_value is not None
            assert type_check(expectation_value) and shape_check(expectation_value)

            error_analysis = simulation_observables.get_error_analysis(
                obs_type, obs_name
            )
            assert isinstance(error_analysis, dict)
            assert not any(v is None for v in error_analysis.values())
            assert all(
                type_check(cast(np.ndarray, v)) and shape_check(cast(np.ndarray, v))
                for v in error_analysis.values()
            )
