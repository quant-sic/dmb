"""Tests for the data.transforms module."""

import torch
from pytest_cases import parametrize_with_cases

from dmb.data.transforms import GroupElement


class GroupElementsCases:
    """Cases for GroupElement tests."""

    @staticmethod
    def case_empty() -> (
        tuple[list[GroupElement], torch.Tensor, torch.Tensor, GroupElement]
    ):
        """Return an empty list of group elements, the initial number,
        the expected result after applying the group elements,
        and the composed group element."""
        return (
            [],
            torch.Tensor([1.0]),
            torch.Tensor([1.0]),
            GroupElement(
                name="identity",
                transform=lambda x: x,
                inverse_transform=lambda x: x,
            ),
        )

    @staticmethod
    def case_group_elements_add() -> (
        tuple[list[GroupElement], torch.Tensor, torch.Tensor, GroupElement]
    ):
        """Return a list of group elements that add to a number, the initial number,
        the expected result after applying the group elements,
        and the composed group element."""
        return (
            [
                GroupElement(
                    name="add_3",
                    transform=lambda x: x + 3,
                    inverse_transform=lambda x: x - 3,
                ),
                GroupElement(
                    name="add_10",
                    transform=lambda x: x + 10,
                    inverse_transform=lambda x: x - 10,
                ),
                GroupElement(
                    name="add_1.5",
                    transform=lambda x: x + 1.5,
                    inverse_transform=lambda x: x - 1.5,
                ),
            ],
            torch.Tensor([1.0]),
            torch.Tensor([15.5]),
            GroupElement(
                name="add_14.5",
                transform=lambda x: x + 14.5,
                inverse_transform=lambda x: x - 14.5,
            ),
        )

    @staticmethod
    def case_group_elements_multiply() -> (
        tuple[list[GroupElement], torch.Tensor, torch.Tensor, GroupElement]
    ):
        """Return a list of group elements that multiply a number, the initial number,
        the expected result after applying the group elements
        and the composed group element."""
        return (
            [
                GroupElement(
                    name="multiply_2",
                    transform=lambda x: x * 2,
                    inverse_transform=lambda x: x / 2,
                ),
                GroupElement(
                    name="multiply_0.5",
                    transform=lambda x: x * 0.5,
                    inverse_transform=lambda x: x / 0.5,
                ),
                GroupElement(
                    name="multiply_3",
                    transform=lambda x: x * 3,
                    inverse_transform=lambda x: x / 3,
                ),
            ],
            torch.Tensor([3.0]),
            torch.Tensor([9.0]),
            GroupElement(
                name="multiply_3",
                transform=lambda x: x * 3,
                inverse_transform=lambda x: x / 3,
            ),
        )


class TestGroupElement:
    """Tests for the GroupElement class."""

    @staticmethod
    @parametrize_with_cases(
        "group_elements, x, expected, composed", cases=GroupElementsCases
    )
    def test_from_group_elements(
        group_elements: list[GroupElement],
        x: torch.Tensor,
        expected: torch.Tensor,
        composed: GroupElement,
    ) -> None:
        """Test that the composed group element transforms as expected."""

        composed = GroupElement.from_group_elements(group_elements)
        assert composed.transform(x) == expected
        assert composed.inverse_transform(expected) == x

        assert composed.transform(x) == expected
        assert composed.inverse_transform(expected) == x
