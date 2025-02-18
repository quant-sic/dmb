from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters
from dmb.data.bose_hubbard_2d.worm.simulation.output import WormOutput


class WormOutputTests:
    @staticmethod
    @pytest.fixture(scope="function", name="density_data")
    def fixture_density_data(request: pytest.FixtureRequest) -> np.ndarray:
        return np.random.rand(*getattr(request, "param", (100, 3, 3)))

    @staticmethod
    @pytest.fixture(scope="function", name="sim_output_valid_densities")
    def fixture_sim_output_valid_densities(
        input_parameters: WormInputParameters,
        output_save_dir_path: Path,
        density_data: np.ndarray,
    ) -> Iterator[None]:
        outputfile_path = input_parameters.get_outputfile_path(
            save_dir_path=output_save_dir_path
        )

        with h5py.File(outputfile_path, "w") as f:
            f.create_group("simulation")
            f["simulation"].create_dataset("densities", data=density_data)

        yield

        outputfile_path.unlink()

    @staticmethod
    @pytest.fixture(scope="function", name="invalidity_reason")
    def fixture_invalidity_reason(request: pytest.FixtureRequest) -> str:
        return getattr(request, "param", "no_densities")

    @staticmethod
    @pytest.fixture(scope="function", name="sim_output_invalid_densities")
    def fixture_sim_output_invalid_densities(
        input_parameters: WormInputParameters,
        output_save_dir_path: Path,
        invalidity_reason: str,
        density_data: np.ndarray,
    ) -> Iterator[None]:
        if invalidity_reason == "no_densities":
            outputfile_path = input_parameters.get_outputfile_path(
                save_dir_path=output_save_dir_path
            )
            with h5py.File(outputfile_path, "w") as f:
                f.create_group("simulation")
            yield

            outputfile_path.unlink()

        elif invalidity_reason == "densities_invalid_shape":
            outputfile_path = input_parameters.get_outputfile_path(
                save_dir_path=output_save_dir_path
            )

            with h5py.File(outputfile_path, "w") as f:
                f.create_group("simulation")
                f["simulation"].create_dataset("densities", data=density_data)
            yield

            outputfile_path.unlink()

        elif invalidity_reason == "no_output_file":
            yield


class TestWormOutput(WormOutputTests):
    @staticmethod
    @pytest.fixture(scope="function", name="output_save_dir_path")
    def fixture_output_save_dir_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
        return tmp_path_factory.mktemp("test_simulation")

    @staticmethod
    @pytest.mark.parametrize(
        "density_data", [(100,), (100, 3), (100, 3, 3, 3)], indirect=True
    )
    @pytest.mark.parametrize(
        "invalidity_reason",
        ["no_densities", "densities_invalid_shape", "no_output_file"],
        indirect=True,
    )
    def test_invalid_densities(
        input_parameters: WormInputParameters,
        sim_output_invalid_densities: Iterator[None],
        output_save_dir_path: Path,
    ) -> None:
        worm_output = WormOutput(
            input_parameters=input_parameters,
            out_file_path=input_parameters.get_outputfile_path(
                save_dir_path=output_save_dir_path
            ),
        )

        assert worm_output.densities is None

    @staticmethod
    @pytest.mark.parametrize("density_data", [(100, 3, 3), (100, 9)], indirect=True)
    def test_valid_densities(
        input_parameters: WormInputParameters,
        sim_output_valid_densities: Iterator[None],
        output_save_dir_path: Path,
        density_data: np.ndarray,
    ) -> None:
        worm_output = WormOutput(
            input_parameters=input_parameters,
            out_file_path=input_parameters.get_outputfile_path(
                save_dir_path=output_save_dir_path
            ),
        )

        assert worm_output.densities is not None
        assert worm_output.densities.shape[-2:] == (
            input_parameters.Lx,
            input_parameters.Ly,
        )
        assert np.array_equal(worm_output.densities.flatten(), density_data.flatten())
