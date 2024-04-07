import itertools

import numpy as np
import pytest
from fake_worm_output import FakeWormOutput

from dmb.data.bose_hubbard_2d.cpp_worm.worm.observables import \
    SimulationObservables


class TestsSimulationObservables:

    @staticmethod
    @pytest.fixture(scope="class", name="fake_output")
    def fixture_fake_output():
        densities = np.random.rand(100, 16, 16)
        return FakeWormOutput(densities)

    @staticmethod
    @pytest.fixture(scope="class", name="instance")
    def fixture_instance(fake_output):
        return SimulationObservables(fake_output)

    @staticmethod
    @pytest.fixture(scope="class", name="observables_keys")
    def fixture_observables_keys(instance):
        return list(
            itertools.chain.from_iterable(
                list(itertools.product((obs_type, ), obs_name))
                for obs_type, obs_name in instance.observable_names.items()))

    @staticmethod
    def test_observable_types(instance, observables_keys):

        for obs_type, obs_name in observables_keys:
            type_check = lambda obj: (isinstance(obj, float)
                                      if obs_type == "derived" else isinstance(
                                          obj, np.ndarray))
            shape_check = lambda obj: (np.isscalar(obj) if obs_type ==
                                       "derived" else obj.shape == (16, 16))

            expectation_value = instance.get_expectation_value(
                obs_type, obs_name)
            assert type_check(expectation_value) and shape_check(
                expectation_value)

            error_analysis = instance.get_error_analysis(obs_type, obs_name)
            assert isinstance(error_analysis, dict)
            assert all(
                type_check(v) and shape_check(v)
                for v in error_analysis.values())
