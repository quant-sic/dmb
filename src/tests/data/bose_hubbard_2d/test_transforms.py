"""Tests for the Bose-Hubbard 2D transforms."""

import functools
from typing import Literal

import torch
from numpy.random import RandomState  # pylint: disable=no-name-in-module
from pytest_cases import fixture, parametrize, parametrize_with_cases

from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2dTransforms, \
    GaussianNoiseTransform, SquareSymmetryGroupTransforms, \
    TupleWrapperInTransform, TupleWrapperOutTransform
from dmb.data.dataset import DMBData
from dmb.data.transforms import DMBTransform, InputOutputDMBTransform


class DMBDataCases:
    """Test cases for DMBData objects."""

    @staticmethod
    def case_all_random() -> DMBData:
        """Return a DMBData object with random input and output."""
        inputs = torch.randn(4, 10, 10)
        outputs = torch.randn(6, 10, 10)
        return DMBData(inputs=inputs, outputs=outputs)

    @staticmethod
    def case_equal_input_output() -> DMBData:
        """Return a DMBData object with equal input and output."""
        inputs = torch.randn(4, 10, 10)
        outputs = inputs.clone()
        return DMBData(inputs=inputs, outputs=outputs)


class InputOutputDMBTransformTests:
    """Test class for InputOutputDMBTransform DMB transforms."""

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases)
    def test_types(transform: InputOutputDMBTransform, data: DMBData) -> None:
        """Test that the transform returns the right types."""
        x, y = transform(data["inputs"], data["outputs"])
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


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
        x, y = wrapper_transform(data["inputs"], data["outputs"])

        if isinstance(wrapper_transform, TupleWrapperInTransform):
            assert torch.allclose(x, data["inputs"] + 1)
            assert torch.allclose(y, data["outputs"])
        else:
            assert torch.allclose(x, data["inputs"])
            assert torch.allclose(y, data["outputs"] + 1)


class TestSquareSymmetryGroupTransforms(InputOutputDMBTransformTests):
    """Tests for the SquareSymmetryGroupTransforms class."""

    @staticmethod
    def case_square_symmetry_group_transforms() -> SquareSymmetryGroupTransforms:
        """Return a square symmetry group transform."""
        return SquareSymmetryGroupTransforms()

    @staticmethod
    def case_non_identity_square_symmetry_group_transforms(
    ) -> (SquareSymmetryGroupTransforms):
        """Return a transform that is not the identity."""
        return SquareSymmetryGroupTransforms(random_number_generator=functools.partial(
            RandomState(42).uniform, low=1 / 8, high=1))

    @staticmethod
    @fixture(scope="class", name="transform")
    @parametrize_with_cases(
        "transform_variant",
        cases=[
            case_square_symmetry_group_transforms,
            case_non_identity_square_symmetry_group_transforms,
        ],
    )
    def fixture_transform(
        transform_variant: SquareSymmetryGroupTransforms, ) -> InputOutputDMBTransform:
        """Return the transform variant."""
        return transform_variant

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases, glob="*equal_input_output*")
    def test_equal_input_output(transform: InputOutputDMBTransform,
                                data: DMBData) -> None:
        """Test that after applying the transform, the input and output are equal."""
        x, y = [], []

        for _ in range(1000):
            x_, y_ = transform(data["inputs"], data["outputs"])
            x.append(x_)
            y.append(y_)
            assert torch.allclose(x_, y_)

        assert any(not torch.allclose(x_, data["inputs"]) for x_ in x)

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases, glob="*equal_input_output*")
    @parametrize_with_cases(
        "non_identity_transform",
        cases=[case_non_identity_square_symmetry_group_transforms],
    )
    def test_non_identity_equal_input_output(
            non_identity_transform: InputOutputDMBTransform, data: DMBData) -> None:
        """Test that the transform is not the identity."""
        x, y = non_identity_transform(data["inputs"], data["outputs"])

        assert torch.allclose(x, y)
        assert not torch.allclose(x, data["inputs"])
        assert not torch.allclose(y, data["outputs"])


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
    def case_tuple_wrapper_in_transform(mean: float,
                                        std: float) -> TupleWrapperInTransform:
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
    def case_tuple_wrapper_out_transform(mean: float,
                                         std: float) -> TupleWrapperOutTransform:
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
        x, y = wrapper_transform(data["inputs"], data["outputs"])

        if isinstance(wrapper_transform, TupleWrapperInTransform):
            assert not torch.allclose(x, data["inputs"])
            assert torch.allclose(y, data["outputs"])
        else:
            assert torch.allclose(x, data["inputs"])
            assert not torch.allclose(y, data["outputs"])
        # highly unlikely to be equal (minimal chance due to limited
        #float precision), assumes std > 0


class TestBoseHubbard2dTransforms(InputOutputDMBTransformTests):
    """Test the BoseHubbard2dTransforms class."""

    @staticmethod
    def case_input_altering_base_augmentation() -> BoseHubbard2dTransforms:
        """Return a transform that alters the input in the base augmentations."""
        return BoseHubbard2dTransforms(
            base_augmentations=[TupleWrapperInTransform(FakeDMBTransform())], )

    @staticmethod
    def case_input_altering_train_augmentation() -> BoseHubbard2dTransforms:
        """Return a transform that alters the input in the train augmentations."""
        return BoseHubbard2dTransforms(
            train_augmentations=[TupleWrapperInTransform(FakeDMBTransform())], )

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
        transform_variant: BoseHubbard2dTransforms, ) -> InputOutputDMBTransform:
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

        if (current_cases["transform_variant"].id == "input_altering_base_augmentation"
                and mode == "base"):
            expected_change = 1
        elif (current_cases["transform_variant"].id
              == "input_altering_train_augmentation" and mode == "base"):
            expected_change = 0
        elif (current_cases["transform_variant"].id
              == "input_altering_base_and_train_augmentation" and mode == "base"):
            expected_change = 1
        elif (current_cases["transform_variant"].id
              == "input_altering_base_augmentation" and mode == "train"):
            expected_change = 1
        elif (current_cases["transform_variant"].id
              == "input_altering_train_augmentation" and mode == "train"):
            expected_change = 1
        elif (current_cases["transform_variant"].id
              == "input_altering_base_and_train_augmentation" and mode == "train"):
            expected_change = 2
        else:
            raise ValueError("Unexpected case: " +
                             current_cases["transform_variant"].id)

        transform_variant.mode = mode
        x, _ = transform_variant(data["inputs"], data["outputs"])

        assert torch.allclose(x, data["inputs"] + expected_change)
