import json
from pathlib import Path

import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters, \
    WormSimulation
from dmb.paths import REPO_DATA_ROOT

from .utils import WormInputParametersDecoder


class FakeDispatcher:

    def __init__(self, *args, **kwargs):
        pass

    def dispatch(self, *args, **kwargs):
        pass


class TestWormSimulation:

    @staticmethod
    @pytest.fixture(scope="class", name="input_parameters")
    def fixture_input_parameters() -> WormInputParameters:
        with open(REPO_DATA_ROOT / "test/input_parameters.json") as f:
            params = json.load(f, cls=WormInputParametersDecoder)["parameters"]

        return WormInputParameters(**params)

    @staticmethod
    @pytest.fixture(scope="class", name="worm_executable")
    def fixture_worm_executable() -> str:
        return "worm_fake_executable"

    @staticmethod
    @pytest.fixture(scope="class", name="test_dispatcher")
    def fixture_test_dispatcher() -> FakeDispatcher:
        return FakeDispatcher()

    @staticmethod
    @pytest.fixture(scope="class", name="test_simulation")
    def fixture_test_simulation(
            input_parameters: WormInputParameters,
            tmp_path_factory: pytest.TempPathFactory, worm_executable: str,
            test_dispatcher: FakeDispatcher) -> WormSimulation:
        return WormSimulation(
            input_parameters=input_parameters,
            save_dir=tmp_path_factory.mktemp("test_simulation"),
            dispatcher=test_dispatcher,
            executable=worm_executable)

    @staticmethod
    def test_from_dir(test_simulation: WormSimulation, worm_executable: str,
                      test_dispatcher):

        loaded_sim = WormSimulation.from_dir(dir_path=test_simulation.save_dir,
                                             dispatcher=FakeDispatcher(),
                                             executable=worm_executable)

        assert loaded_sim.input_parameters == test_simulation.input_parameters

    @staticmethod
    def test_save_parameters(input_parameters: WormInputParameters,
                             tmp_path_factory: pytest.TempPathFactory,
                             worm_executable: str,
                             test_dispatcher: FakeDispatcher):

        test_simulation = WormSimulation(
            input_parameters=input_parameters,
            save_dir=tmp_path_factory.mktemp(
                "test_save_parameters_simulation"),
            dispatcher=test_dispatcher,
            executable=worm_executable)

        test_simulation.save_parameters()
        assert test_simulation.input_parameters.get_ini_path(
            test_simulation.save_dir).exists()
        assert test_simulation.input_parameters.get_h5_path(
            test_simulation.save_dir).exists()
