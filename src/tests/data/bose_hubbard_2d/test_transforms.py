"""Tests for the Bose-Hubbard 2D transforms."""

import itertools
from typing import Literal

import torch
from pytest_cases import case
from pytest_cases import filters as ft
from pytest_cases import fixture, parametrize, parametrize_with_cases

from dmb.data.bose_hubbard_2d.transforms import (
    BoseHubbard2dTransforms,
    D4Group,
    D4GroupTransforms,
    GaussianNoiseTransform,
    TupleWrapperInTransform,
    TupleWrapperOutTransform,
)
from dmb.data.dataset import DMBData
from dmb.data.transforms import (
    DMBDatasetTransform,
    DMBTransform,
    InputOutputDMBTransform,
)


class DMBDataCases:
    """Test cases for DMBData objects."""

    @staticmethod
    def case_all_random() -> DMBData:
        """Return a DMBData object with random input and output."""
        inputs = torch.randn(4, 10, 10)
        outputs = torch.randn(6, 10, 10)
        return DMBData(inputs=inputs, outputs=outputs, sample_id="random")

    @staticmethod
    def case_equal_input_output() -> DMBData:
        """Return a DMBData object with equal input and output."""
        inputs = torch.randn(4, 10, 10)
        outputs = inputs.clone()
        return DMBData(inputs=inputs, outputs=outputs, sample_id="equal_input_output")

    @staticmethod
    @case(tags=["all_values_different"])
    def case_all_values_different() -> DMBData:
        """Return a DMBData object with different input and output."""
        inputs = torch.arange(400).reshape(4, 10, 10).float()
        outputs = inputs + inputs.max() + 1
        return DMBData(inputs=inputs, outputs=outputs, sample_id="all_values_different")


class InputOutputDMBTransformTests:
    """Test class for InputOutputDMBTransform DMB transforms."""

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases)
    def test_types(transform: InputOutputDMBTransform, data: DMBData) -> None:
        """Test that the transform returns the right types."""
        data_out = transform(data)
        assert isinstance(data_out.inputs, torch.Tensor)
        assert isinstance(data_out.outputs, torch.Tensor)


class FakeDMBTransform(DMBTransform):
    """Fake DMB transform that adds 1 to the input."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input plus 1."""
        return x + 1


class TestTupleWrapperTransform(InputOutputDMBTransformTests):
    """Tests for the TupleWrapperInTransform and TupleWrapperOutTransform classes."""

    @staticmethod
    def case_tuple_wrapper_in_transform() -> TupleWrapperInTransform:
        """Return a transform that adds 1 to the input."""
        return TupleWrapperInTransform(FakeDMBTransform())

    @staticmethod
    def case_tuple_wrapper_out_transform() -> TupleWrapperOutTransform:
        """Return a transform that adds 1 to the output."""
        return TupleWrapperOutTransform(FakeDMBTransform())

    @staticmethod
    @fixture(scope="class", name="transform")
    @parametrize_with_cases(
        "wrapper_transform",
        cases=[case_tuple_wrapper_in_transform, case_tuple_wrapper_out_transform],
    )
    def fixture_transform(
        wrapper_transform: TupleWrapperInTransform | TupleWrapperOutTransform,
    ) -> InputOutputDMBTransform:
        """Return the transform variant."""
        return wrapper_transform

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases)
    @parametrize_with_cases(
        "wrapper_transform",
        cases=[case_tuple_wrapper_in_transform, case_tuple_wrapper_out_transform],
    )
    def test_tuple_wrapper_transform(
        wrapper_transform: TupleWrapperInTransform | TupleWrapperOutTransform,
        data: DMBData,
    ) -> None:
        """Test that the right input/output is altered."""
        data_out = wrapper_transform(data)

        if isinstance(wrapper_transform, TupleWrapperInTransform):
            assert torch.allclose(data_out.inputs, data.inputs + 1)
            assert torch.allclose(data_out.outputs, data.outputs)
        else:
            assert torch.allclose(data_out.inputs, data.inputs)
            assert torch.allclose(data_out.outputs, data.outputs + 1)


class TestD4GroupTransforms(InputOutputDMBTransformTests):
    """Tests for the D4GroupTransforms class."""

    @staticmethod
    @fixture(scope="class", name="transform")
    def fixture_transform() -> D4GroupTransforms:
        """Return the transform variant."""
        return D4GroupTransforms()

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases, glob="*equal_input_output*")
    def test_equal_input_output(
        transform: InputOutputDMBTransform, data: DMBData
    ) -> None:
        """Test that after applying the transform, the input and output are equal."""
        x, y = [], []

        for _ in range(1000):
            data_out = transform(data)
            x.append(data_out.inputs)
            y.append(data_out.outputs)
            assert torch.allclose(data_out.inputs, data_out.outputs)

        assert any(not torch.allclose(x_, data.inputs) for x_ in x)


class TestGaussianNoiseTransform(InputOutputDMBTransformTests):
    """Tests for the GaussianNoiseTransform class."""

    @staticmethod
    @parametrize(
        argnames="mean, std",
        argvalues=[
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.1),
            (1.0, 0.1),
        ],
    )
    def case_tuple_wrapper_in_transform(
        mean: float, std: float
    ) -> TupleWrapperInTransform:
        """Return a transform that adds Gaussian noise to the input."""
        return TupleWrapperInTransform(GaussianNoiseTransform(mean=mean, std=std))

    @staticmethod
    @parametrize(
        argnames="mean, std",
        argvalues=[
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.1),
            (1.0, 0.1),
        ],
    )
    def case_tuple_wrapper_out_transform(
        mean: float, std: float
    ) -> TupleWrapperOutTransform:
        """Return a transform that adds Gaussian noise to the output."""
        return TupleWrapperOutTransform(GaussianNoiseTransform(mean=mean, std=std))

    @staticmethod
    @fixture(scope="function", name="transform")
    @parametrize_with_cases(
        "wrapper_transform",
        cases=[case_tuple_wrapper_in_transform, case_tuple_wrapper_out_transform],
    )
    def fixture_transform(
        wrapper_transform: TupleWrapperInTransform | TupleWrapperOutTransform,
    ) -> InputOutputDMBTransform:
        """Return the transform variant."""
        return wrapper_transform

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases)
    @parametrize_with_cases(
        "wrapper_transform",
        cases=[case_tuple_wrapper_in_transform, case_tuple_wrapper_out_transform],
    )
    def test_tuple_wrapper_transform(
        wrapper_transform: TupleWrapperInTransform | TupleWrapperOutTransform,
        data: DMBData,
    ) -> None:
        """Test that the right input/output is altered."""
        data_out = wrapper_transform(data)

        if isinstance(wrapper_transform, TupleWrapperInTransform):
            assert not torch.allclose(data_out.inputs, data.inputs)
            assert torch.allclose(data_out.outputs, data.outputs)
        else:
            assert torch.allclose(data_out.inputs, data.inputs)
            assert not torch.allclose(data_out.outputs, data.outputs)
        # highly unlikely to be equal (minimal chance due to limited
        # float precision), assumes std > 0


class TestBoseHubbard2dTransforms(InputOutputDMBTransformTests):
    """Test the BoseHubbard2dTransforms class."""

    @staticmethod
    def case_input_altering_base_augmentation() -> BoseHubbard2dTransforms:
        """Return a transform that alters the input in the base augmentations."""
        return BoseHubbard2dTransforms(
            base_augmentations=[TupleWrapperInTransform(FakeDMBTransform())],
        )

    @staticmethod
    def case_input_altering_train_augmentation() -> BoseHubbard2dTransforms:
        """Return a transform that alters the input in the train augmentations."""
        return BoseHubbard2dTransforms(
            train_augmentations=[TupleWrapperInTransform(FakeDMBTransform())],
        )

    @staticmethod
    def case_input_altering_base_and_train_augmentation() -> BoseHubbard2dTransforms:
        """Return a transform that alters the input in both base and train."""
        return BoseHubbard2dTransforms(
            base_augmentations=[TupleWrapperInTransform(FakeDMBTransform())],
            train_augmentations=[TupleWrapperInTransform(FakeDMBTransform())],
        )

    @staticmethod
    @fixture(scope="class", name="transform")
    @parametrize_with_cases(
        "transform_variant",
        cases=[
            case_input_altering_base_augmentation,
            case_input_altering_train_augmentation,
            case_input_altering_base_and_train_augmentation,
        ],
    )
    def fixture_transform(
        transform_variant: BoseHubbard2dTransforms,
    ) -> DMBDatasetTransform:
        """Return the transform variant."""
        return transform_variant

    @staticmethod
    @parametrize(
        argnames="mode",
        argvalues=["base", "train"],
    )
    @parametrize_with_cases(
        "transform_variant",
        cases=[
            case_input_altering_base_augmentation,
            case_input_altering_train_augmentation,
            case_input_altering_base_and_train_augmentation,
        ],
    )
    @parametrize_with_cases("data", cases=DMBDataCases)
    def test_mode_output(
        mode: Literal["base", "train"],
        transform_variant: BoseHubbard2dTransforms,
        data: DMBData,
        current_cases: dict,
    ) -> None:
        """Test if the output is altered as expected based on the mode."""

        if (
            current_cases["transform_variant"].id == "input_altering_base_augmentation"
            and mode == "base"
        ):
            expected_change = 1
        elif (
            current_cases["transform_variant"].id == "input_altering_train_augmentation"
            and mode == "base"
        ):
            expected_change = 0
        elif (
            current_cases["transform_variant"].id
            == "input_altering_base_and_train_augmentation"
            and mode == "base"
        ):
            expected_change = 1
        elif (
            current_cases["transform_variant"].id == "input_altering_base_augmentation"
            and mode == "train"
        ):
            expected_change = 1
        elif (
            current_cases["transform_variant"].id == "input_altering_train_augmentation"
            and mode == "train"
        ):
            expected_change = 1
        elif (
            current_cases["transform_variant"].id
            == "input_altering_base_and_train_augmentation"
            and mode == "train"
        ):
            expected_change = 2
        else:
            raise ValueError(
                "Unexpected case: " + current_cases["transform_variant"].id
            )

        transform_variant.mode = mode
        data_out = transform_variant(data)

        assert torch.allclose(data_out.inputs, data.inputs + expected_change)


class TestD4Group:
    """Tests for the D4Group class."""

    @staticmethod
    @fixture(scope="class", name="d4_group")
    def fixture_d4_group() -> D4Group:
        """Return a D4Group instance."""
        return D4Group()

    @staticmethod
    @parametrize_with_cases(
        "data", cases=DMBDataCases, filter=ft.has_tag("all_values_different")
    )
    def test_all_different(data: DMBData, d4_group: D4Group) -> None:
        """Test that all D4 group elements transform the input differently."""

        original = data.inputs

        assert torch.allclose(
            original, d4_group.elements["identity"].transform(original)
        )

        transformed = [
            element.transform(original)
            for element in d4_group.elements.values()
            if element.name != "identity"
        ]

        for v1, v2 in itertools.combinations(transformed, 2):
            assert not torch.allclose(v1, v2)

    @staticmethod
    @parametrize_with_cases(
        "data", cases=DMBDataCases, filter=ft.has_tag("all_values_different")
    )
    def test_inverse(data: DMBData, d4_group: D4Group) -> None:
        """Test that the D4 group transformation inverse transforms are correct."""

        original = data.inputs

        for element in d4_group.elements.values():
            assert torch.allclose(
                original, element.inverse_transform(element.transform(original))
            )
