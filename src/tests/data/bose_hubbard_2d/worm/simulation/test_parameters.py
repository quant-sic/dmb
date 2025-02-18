import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters
from dmb.paths import REPO_DATA_ROOT

from .utils import WormInputParametersDecoder


class ValueObjectTests:
    @staticmethod
    def test_setattr(
        type_: type,
        instance_parameters: Mapping[str, Any],
        different_instance_parameters: Mapping[str, Any],
    ) -> None:
        instance = type_(**instance_parameters)
        for key in instance_parameters:
            with pytest.raises(AttributeError):
                setattr(instance, key, different_instance_parameters[key])

    @staticmethod
    def test_delattr(type_: type, instance_parameters: Mapping[str, Any]) -> None:
        instance = type_(**instance_parameters)
        for key in instance_parameters:
            with pytest.raises(AttributeError):
                delattr(instance, key)

    @staticmethod
    def test_eq(
        type_: type,
        instance_parameters: Mapping[str, Any],
        different_instance_parameters: Mapping[str, Any],
    ) -> None:
        instance = type_(**instance_parameters)
        assert instance == instance

        instance_same = type_(**instance_parameters)
        assert instance == instance_same

        class Other(type_): ...

        instance_other = Other(**instance_parameters)
        assert instance != instance_other

        for key in instance_parameters:
            params = {**instance_parameters, key: different_instance_parameters[key]}
            instance_other = type_(**params)

            assert instance != instance_other


class TestWormInputParameters(ValueObjectTests):
    @staticmethod
    @pytest.fixture(scope="class", name="type_")
    def fixture_type() -> type:
        return WormInputParameters

    @staticmethod
    @pytest.fixture(scope="class", name="instance_parameters")
    def fixture_instance_parameters() -> Mapping[str, Any]:
        with open(REPO_DATA_ROOT / "test/input_parameters.json") as f:
            parameters: dict = json.load(f, cls=WormInputParametersDecoder)[
                "parameters"
            ]
        return parameters

    @staticmethod
    @pytest.fixture(scope="class", name="different_instance_parameters")
    def fixture_different_instance_parameters() -> Mapping[str, Any]:
        with open(REPO_DATA_ROOT / "test/input_parameters.json") as f:
            parameters: dict = json.load(f, cls=WormInputParametersDecoder)[
                "parameters_different_values"
            ]
        return parameters

    def test_get_ini_path(self, tmp_path: Path) -> None:
        assert WormInputParameters.get_ini_path(tmp_path) == tmp_path / "parameters.ini"

    def test_get_h5_path(self, tmp_path: Path) -> None:
        assert WormInputParameters.get_h5_path(tmp_path) == tmp_path / "parameters.h5"

    def test_get_outputfile_path(self, tmp_path: Path) -> None:
        assert (
            WormInputParameters.get_outputfile_path(tmp_path) == tmp_path / "output.h5"
        )

    def test_get_checkpoint_path(self, tmp_path: Path) -> None:
        assert (
            WormInputParameters.get_checkpoint_path(tmp_path)
            == tmp_path / "checkpoint.h5"
        )

    def test_save(self, tmp_path: Path, instance_parameters: Mapping[str, Any]) -> None:
        instance = WormInputParameters(**instance_parameters)
        instance.save(tmp_path)
        assert instance.get_ini_path(tmp_path).exists()
        assert instance.get_h5_path(tmp_path).exists()

    def test_from_dir(
        self, tmp_path: Path, instance_parameters: Mapping[str, Any]
    ) -> None:
        instance = WormInputParameters(**instance_parameters)
        instance.save(tmp_path)

        instance_loaded = WormInputParameters.from_dir(tmp_path)
        assert instance == instance_loaded

    def test_plots(
        self, tmp_path: Path, instance_parameters: Mapping[str, Any]
    ) -> None:
        instance = WormInputParameters(**instance_parameters)
        instance.save(tmp_path)

        instance.plot_input_parameters(tmp_path)
        assert (tmp_path / "inputs.png").exists()

        instance.plot_phase_diagram_input_parameters(tmp_path)
        assert (tmp_path / "phase_diagram_inputs.png").exists()
