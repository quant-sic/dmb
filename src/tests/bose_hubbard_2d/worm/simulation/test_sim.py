import asyncio
import itertools
from pathlib import Path
from typing import Iterator

import h5py
import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters, \
    WormSimulation
from dmb.data.dispatching import ReturnCode
from dmb.data.dispatching.dispatcher import Dispatcher

from .test_output import WormOutputTests


class FakeDispatcher(Dispatcher):

    def __init__(self,
                 return_code: ReturnCode = ReturnCode.SUCCESS,
                 expect_input_file_type: str = ".ini"):
        self.expect_ini_input_file = expect_input_file_type
        self.return_code = return_code

    async def dispatch(self, job_name: str, work_directory: Path, pipeout_dir: Path,
                       task: list[str], timeout: int) -> ReturnCode:

        await asyncio.sleep(0.01)

        if not task[-1].endswith(self.expect_ini_input_file):
            raise RuntimeError(f"Expected {self.expect_ini_input_file} input file.")

        return self.return_code


class WormSimulationTests:

    @staticmethod
    @pytest.fixture(scope="class", name="worm_executable")
    def fixture_worm_executable() -> str:
        return "worm_fake_executable"

    @staticmethod
    @pytest.fixture(scope="function", name="test_dispatcher_return_code")
    def fixture_dispatcher_return_code(request: pytest.FixtureRequest) -> ReturnCode:
        return getattr(request, "param", ReturnCode.SUCCESS)

    @staticmethod
    @pytest.fixture(scope="function", name="test_dispatcher_expected_input_file_type")
    def fixture_dispatcher_expect_ini_input_file(request: pytest.FixtureRequest) -> str:
        return getattr(request, "param", "ini")

    @staticmethod
    @pytest.fixture(scope="function", name="test_dispatcher")
    def fixture_test_dispatcher(
            test_dispatcher_return_code: ReturnCode,
            test_dispatcher_expected_input_file_type: str) -> FakeDispatcher:
        return FakeDispatcher(
            return_code=test_dispatcher_return_code,
            expect_input_file_type=test_dispatcher_expected_input_file_type)

    @staticmethod
    @pytest.fixture(scope="function", name="test_simulation")
    def fixture_test_simulation(input_parameters: WormInputParameters,
                                tmp_path_factory: pytest.TempPathFactory,
                                worm_executable: str,
                                test_dispatcher: FakeDispatcher) -> WormSimulation:
        return WormSimulation(input_parameters=input_parameters,
                              save_dir=tmp_path_factory.mktemp("test_simulation"),
                              dispatcher=test_dispatcher,
                              executable=Path(worm_executable))


class TestWormSimulation(WormOutputTests, WormSimulationTests):

    @staticmethod
    def test_parameters_saved(test_simulation: WormSimulation) -> None:
        assert test_simulation.input_parameters.get_ini_path(
            test_simulation.save_dir).exists()
        assert test_simulation.input_parameters.get_h5_path(
            test_simulation.save_dir).exists()

    @staticmethod
    def test_init_for_non_existant_save_dir(
            input_parameters: WormInputParameters, worm_executable: str,
            test_dispatcher: FakeDispatcher,
            tmp_path_factory: pytest.TempPathFactory) -> None:

        save_dir = tmp_path_factory.mktemp("test_init_for_non_existant_save_dir")
        non_existant_save_dir = save_dir / "non_existant_save_dir"
        test_simulation = WormSimulation(input_parameters=input_parameters,
                                         save_dir=non_existant_save_dir)

        assert test_simulation.save_dir == non_existant_save_dir
        assert non_existant_save_dir.exists()

    @staticmethod
    def test_from_dir(test_simulation: WormSimulation, worm_executable: str) -> None:
        WormSimulation.from_dir(dir_path=test_simulation.save_dir)

    @staticmethod
    def test_save_parameters(input_parameters: WormInputParameters,
                             tmp_path_factory: pytest.TempPathFactory,
                             worm_executable: str,
                             test_dispatcher: FakeDispatcher) -> None:

        test_simulation = WormSimulation(
            input_parameters=input_parameters,
            save_dir=tmp_path_factory.mktemp("test_save_parameters_simulation"),
            dispatcher=test_dispatcher,
            executable=Path(worm_executable))

        test_simulation.save_parameters()
        assert test_simulation.input_parameters.get_ini_path(
            test_simulation.save_dir).exists()
        assert test_simulation.input_parameters.get_h5_path(
            test_simulation.save_dir).exists()

    @staticmethod
    def test_get_tune_dir_path(test_simulation: WormSimulation) -> None:
        assert test_simulation.get_tune_dir_path(test_simulation.save_dir) == \
            test_simulation.save_dir / "tune"

    @staticmethod
    def test_get_plot_dir_path(test_simulation: WormSimulation) -> None:
        assert test_simulation.get_plot_dir_path(test_simulation.save_dir) == \
            test_simulation.save_dir / "plots"

    @staticmethod
    def test_tune_simulation(test_simulation: WormSimulation) -> None:
        tune_sim = test_simulation.tune_simulation
        assert tune_sim.save_dir == test_simulation.get_tune_dir_path(
            test_simulation.save_dir)
        assert tune_sim.input_parameters == test_simulation.input_parameters
        assert tune_sim.executable == test_simulation.executable
        assert tune_sim.dispatcher == test_simulation.dispatcher

        assert tune_sim.save_dir.exists()

    @staticmethod
    @pytest.fixture(scope="function", name="test_checkpoint_status")
    def fixture_test_checkpoint_status(request: pytest.FixtureRequest) -> str:
        return getattr(request, "param", "no_checkpoint")

    @staticmethod
    @pytest.fixture(scope="function", name="test_checkpoint")
    def fixture_test_checkpoint(test_simulation: WormSimulation,
                                test_checkpoint_status: str) -> Iterator[None]:
        checkpoint_path = test_simulation.input_parameters.get_checkpoint_path(
            test_simulation.save_dir)

        if test_checkpoint_status == "no_checkpoint":
            yield

        elif test_checkpoint_status == "checkpoint_exists":
            #create h5 file
            with h5py.File(checkpoint_path, "w") as f:
                pass

            yield
            checkpoint_path.unlink()

        elif test_checkpoint_status == "checkpoint_with_parameters":
            #create h5 file
            with h5py.File(checkpoint_path, "w") as f:
                f.create_group("parameters")

            yield
            checkpoint_path.unlink()

    @staticmethod
    @pytest.mark.parametrize(
        "test_checkpoint_status",
        ["no_checkpoint", "checkpoint_exists", "checkpoint_with_parameters"])
    def test_set_get_extension_sweeps_in_checkpoints(
            test_simulation: WormSimulation, test_checkpoint: Iterator[None],
            test_checkpoint_status: str) -> None:

        for extension_sweeps in (10, 20, 30):
            if test_checkpoint_status in [
                    "checkpoint_with_parameters", "checkpoint_exists"
            ]:
                test_simulation.set_extension_sweeps_in_checkpoints(extension_sweeps)
                assert test_simulation.get_extension_sweeps_from_checkpoints() == \
                    extension_sweeps

            elif test_checkpoint_status == "no_checkpoint":
                test_simulation.set_extension_sweeps_in_checkpoints(extension_sweeps)
                assert test_simulation.get_extension_sweeps_from_checkpoints() is None

    @staticmethod
    @pytest.mark.parametrize(
        "test_checkpoint_status",
        ["no_checkpoint", "checkpoint_exists", "checkpoint_with_parameters"])
    def test_get_extension_sweeps_from_checkpoints_no_extension_sweeps(
            test_simulation: WormSimulation, test_checkpoint: Iterator[None],
            test_checkpoint_status: str) -> None:

        assert test_simulation.get_extension_sweeps_from_checkpoints() is None

    @staticmethod
    @pytest.fixture(scope="function", name="output_save_dir_path")
    def fixture_output_save_dir_path(test_simulation: WormSimulation) -> Path:
        return test_simulation.save_dir

    @staticmethod
    def test_output(test_simulation: WormSimulation,
                    sim_output_valid_densities: Iterator[None]) -> None:

        assert test_simulation.output
        assert test_simulation.output.densities is not None

    @staticmethod
    def test_simulation_results(test_simulation: WormSimulation,
                                sim_output_valid_densities: Iterator[None]) -> None:

        assert test_simulation.max_tau_int is not None
        assert test_simulation.uncorrected_max_density_error is not None
        assert test_simulation.max_density_error is not None

    @staticmethod
    def test_observables(test_simulation: WormSimulation,
                         sim_output_valid_densities: Iterator[None]) -> None:

        assert test_simulation.observables
        for obs_type, obs_names in test_simulation.observables.observable_names.items():
            for obs_name in obs_names:
                assert test_simulation.observables.get_expectation_value(
                    obs_type, obs_name) is not None
                for key, obs in test_simulation.observables.get_error_analysis(
                        obs_type, obs_name).items():
                    assert obs is not None

    @staticmethod
    def test_plotting(test_simulation: WormSimulation,
                      sim_output_valid_densities: Iterator[None]) -> None:

        test_simulation.plot_observables(test_simulation.observables.observable_names)

        for obs_name in itertools.chain.from_iterable(
                test_simulation.observables.observable_names.values()):

            produced_files = (
                test_simulation.get_plot_dir_path(test_simulation.save_dir) /
                obs_name).glob("*.png")
            assert any(produced_files)

        test_simulation.plot_inputs()
        assert (test_simulation.get_plot_dir_path(test_simulation.save_dir) /
                "inputs.png").exists()

        test_simulation.plot_phase_diagram_inputs()
        assert (test_simulation.get_plot_dir_path(test_simulation.save_dir) /
                "phase_diagram_inputs.png").exists()

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("test_dispatcher_return_code",
                             [ReturnCode.SUCCESS, ReturnCode.FAILURE],
                             indirect=True)
    @pytest.mark.parametrize("test_dispatcher_expected_input_file_type",
                             [".ini", ".h5"],
                             indirect=True)
    async def test_execute_worm(test_simulation: WormSimulation,
                                test_dispatcher_return_code: ReturnCode,
                                test_dispatcher_expected_input_file_type: str) -> None:
        if test_dispatcher_expected_input_file_type == ".h5":
            with pytest.raises(RuntimeError):
                code = await test_simulation.execute_worm()
        else:
            code = await test_simulation.execute_worm()
            assert code == test_dispatcher_return_code

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("test_dispatcher_return_code",
                             [ReturnCode.SUCCESS, ReturnCode.FAILURE],
                             indirect=True)
    @pytest.mark.parametrize("test_dispatcher_expected_input_file_type",
                             [".ini", ".h5"],
                             indirect=True)
    async def test_execute_worm_continue(
            test_simulation: WormSimulation, test_dispatcher_return_code: ReturnCode,
            test_dispatcher_expected_input_file_type: str) -> None:
        if test_dispatcher_expected_input_file_type == ".ini":
            with pytest.raises(RuntimeError):
                code = await test_simulation.execute_worm_continue()
        else:
            code = await test_simulation.execute_worm_continue()
            assert code == test_dispatcher_return_code
