import functools

import numpy as np
import pytest
import torch
from pytest_cases import case, fixture, parametrize, parametrize_with_cases

from dmb.data.bose_hubbard_2d.transforms import GaussianNoise, \
    SquareSymmetryGroupTransforms, TupleWrapperInTransform, \
    TupleWrapperOutTransform
from dmb.data.dataset import DMBData
from dmb.data.transforms import DMBTransform, InputOutputDMBTransform


class DMBDataCases:

    @staticmethod
    def case_all_random() -> DMBData:
        input = torch.randn(4, 10, 10)
        output = torch.randn(6, 10, 10)
        return DMBData(inputs=input, outputs=output)

    @staticmethod
    def case_equal_input_output() -> DMBData:
        input = torch.randn(4, 10, 10)
        output = input.clone()
        return DMBData(inputs=input, outputs=output)


class InputOutputDMBTransformTests:

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases)
    def test_types(transform: InputOutputDMBTransform, data: DMBData) -> None:
        x, y = transform(data["inputs"], data["outputs"])
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


class FakeDMBTransform(DMBTransform):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class TestTupleWrapperTransform(InputOutputDMBTransformTests):

    @staticmethod
    def case_tuple_wrapper_in_transform() -> TupleWrapperInTransform:
        return TupleWrapperInTransform(FakeDMBTransform())

    @staticmethod
    def case_tuple_wrapper_out_transform() -> TupleWrapperOutTransform:
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
        return wrapper_transform

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases)
    @parametrize_with_cases(
        "wrapper_transform",
        cases=[case_tuple_wrapper_in_transform, case_tuple_wrapper_out_transform],
    )
    def test_tuple_wrapper_transform(wrapper_transform: TupleWrapperInTransform
                                     | TupleWrapperOutTransform, data: DMBData) -> None:
        x, y = wrapper_transform(data["inputs"], data["outputs"])

        if isinstance(wrapper_transform, TupleWrapperInTransform):
            assert torch.allclose(x, data["inputs"] + 1)
            assert torch.allclose(y, data["outputs"])
        else:
            assert torch.allclose(x, data["inputs"])
            assert torch.allclose(y, data["outputs"] + 1)


class TestSquareSymmetryGroupTransforms(InputOutputDMBTransformTests):

    @staticmethod
    def case_square_symmetry_group_transforms() -> SquareSymmetryGroupTransforms:
        return SquareSymmetryGroupTransforms()

    @staticmethod
    def case_non_identity_square_symmetry_group_transforms(
    ) -> SquareSymmetryGroupTransforms:
        return SquareSymmetryGroupTransforms(random_number_generator=functools.partial(
            np.random.RandomState(42).uniform, low=1 / 8, high=1))

    @staticmethod
    @fixture(scope="class", name="transform")
    @parametrize_with_cases(
        "transform_variant",
        cases=[
            case_square_symmetry_group_transforms,
            case_non_identity_square_symmetry_group_transforms
        ],
    )
    def fixture_transform(
            transform_variant: SquareSymmetryGroupTransforms
    ) -> InputOutputDMBTransform:
        return transform_variant

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases, glob="*equal_input_output*")
    def test_equal_input_output(transform: InputOutputDMBTransform,
                                data: DMBData) -> None:
        x, y = transform(data["inputs"], data["outputs"])
        assert torch.allclose(x, y)

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases, glob="*equal_input_output*")
    @parametrize_with_cases(
        "non_identity_transform",
        cases=[case_non_identity_square_symmetry_group_transforms],
    )
    def test_non_identity_equal_input_output(
            non_identity_transform: InputOutputDMBTransform, data: DMBData) -> None:
        x, y = non_identity_transform(data["inputs"], data["outputs"])

        assert torch.allclose(x, y)
        assert not torch.allclose(x, data["inputs"])
        assert not torch.allclose(y, data["outputs"])


class TestGaussianNoise(InputOutputDMBTransformTests):

    @staticmethod
    @parametrize(argnames="mean, std",
                 argvalues=[
                     (0.0, 1.0),
                     (1.0, 1.0),
                     (0.0, 0.1),
                     (1.0, 0.1),
                 ])
    def case_tuple_wrapper_in_transform(mean: float,
                                        std: float) -> TupleWrapperInTransform:
        return TupleWrapperInTransform(GaussianNoise(mean=mean, std=std))

    @staticmethod
    @parametrize(argnames="mean, std",
                 argvalues=[
                     (0.0, 1.0),
                     (1.0, 1.0),
                     (0.0, 0.1),
                     (1.0, 0.1),
                 ])
    def case_tuple_wrapper_out_transform(mean: float,
                                         std: float) -> TupleWrapperOutTransform:
        return TupleWrapperOutTransform(GaussianNoise(mean=mean, std=std))

    @staticmethod
    @fixture(scope="function", name="transform")
    @parametrize_with_cases(
        "wrapper_transform",
        cases=[case_tuple_wrapper_in_transform, case_tuple_wrapper_out_transform],
    )
    def fixture_transform(
        wrapper_transform: TupleWrapperInTransform | TupleWrapperOutTransform,
    ) -> InputOutputDMBTransform:
        return wrapper_transform

    @staticmethod
    @parametrize_with_cases("data", cases=DMBDataCases)
    @parametrize_with_cases(
        "wrapper_transform",
        cases=[case_tuple_wrapper_in_transform, case_tuple_wrapper_out_transform],
    )
    def test_tuple_wrapper_transform(wrapper_transform: TupleWrapperInTransform
                                     | TupleWrapperOutTransform, data: DMBData) -> None:
        x, y = wrapper_transform(data["inputs"], data["outputs"])

        if isinstance(wrapper_transform, TupleWrapperInTransform):
            assert not torch.allclose(
                x, data["inputs"]
            )  # highliy unlikely to be equal (minimal chance due to limited float precision), assumes std > 0
            assert torch.allclose(y, data["outputs"])
        else:
            assert torch.allclose(x, data["inputs"])
            assert not torch.allclose(
                y, data["outputs"]
            )  # highliy unlikely to be equal (minimal chance due to limited float precision), assumes std > 0
