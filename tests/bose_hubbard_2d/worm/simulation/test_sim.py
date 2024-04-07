import json
from typing import Iterator

import h5py
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

    @staticmethod
    def test_get_tune_dir_path(test_simulation: WormSimulation):
        assert test_simulation.get_tune_dir_path(test_simulation.save_dir) == \
            test_simulation.save_dir / "tune"

    @staticmethod
    def test_get_plot_dir_path(test_simulation: WormSimulation):
        assert test_simulation.get_plot_dir_path(test_simulation.save_dir) == \
            test_simulation.save_dir / "plots"

    @staticmethod
    def test_tune_simulation(test_simulation: WormSimulation):
        tune_sim = test_simulation.tune_simulation
        assert tune_sim.save_dir == test_simulation.get_tune_dir_path(
            test_simulation.save_dir)
        assert tune_sim.input_parameters == test_simulation.input_parameters
        assert tune_sim.executable == test_simulation.executable
        assert tune_sim.dispatcher == test_simulation.dispatcher

        assert tune_sim.save_dir.exists()

    @staticmethod
    @pytest.fixture(scope="class", name="test_checkpoint")
    def fixture_test_checkpoint(
            test_simulation: WormSimulation) -> Iterator[None]:
        checkpoint_path = test_simulation.input_parameters.get_checkpoint_path(
            test_simulation.save_dir)

        #create h5 file
        with h5py.File(checkpoint_path, "w") as f:
            f.create_group("parameters")

        yield

    @staticmethod
    def test_set_get_extension_sweeps_in_checkpoints(
            test_simulation: WormSimulation, test_checkpoint: None):

        for extension_sweeps in (10, 20, 30):
            test_simulation.set_extension_sweeps_in_checkpoints(
                extension_sweeps)
            assert test_simulation.get_extension_sweeps_from_checkpoints() == \
                extension_sweeps
