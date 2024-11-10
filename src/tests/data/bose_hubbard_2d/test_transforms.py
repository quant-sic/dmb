import pytest
import torch
from pytest_cases import case, fixture, parametrize, parametrize_with_cases

from dmb.data.bose_hubbard_2d.transforms import (
    TupleWrapperInTransform,
    TupleWrapperOutTransform,
)
from dmb.data.dataset import DMBData
from dmb.data.transforms import DMBTransform, InputOutputDMBTransform


class DMBDataCases:

    @case(tags=["bose_hubbard_2d"])
    @staticmethod
    def case_all_random() -> DMBData:
        input = torch.randn(4, 10, 10)
        output = torch.randn(6, 10, 10)
        return DMBData(inputs=input, outputs=output)

    @case(tags=["bose_hubbard_2d"])
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
    def test_tuple_wrapper_transform(
        transform: InputOutputDMBTransform,
        data: DMBData,
        current_cases: pytest.FixtureRequest,
    ) -> None:
        x, y = transform(data["inputs"], data["outputs"])

        print(transform["data"])
        assert False
