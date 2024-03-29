from fake_worm_output import FakeWormOutput
from dmb.data.bose_hubbard_2d.cpp_worm.worm.observables import SimulationObservables
import pytest
import numpy as np


class TestsSimulationObservables:

    @staticmethod
    @pytest.fixture(scope="class", name="fake_output")
    def fixture_fake_output():
        densities = np.random.rand(100, 16 * 16)
        return FakeWormOutput(densities)

    @staticmethod
    @pytest.fixture(scope="class", name="instance")
    def fixture_instance(fake_output):
        return SimulationObservables(fake_output)

    def test_observables_are_scalars(self, instance):
        for key in instance.observables:
            assert isinstance(instance[key], (np.ndarray, float))
            assert len(instance[key].shape) == len(instance.observable_shapes[key])

    def test_observables_are_registered(self):
        assert all(
            obs in SimulationObservables.observable_names()
            for obs in [
                "density",
                "density_variance",
                "density_density_corr_0",
                "density_density_corr_1",
                "density_density_corr_2",
                "density_density_corr_3",
                "density_squared",
                "density_max",
                "density_min",
            ]
        )
