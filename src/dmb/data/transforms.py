"""Data transforms."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, Literal

import torch
from attrs import define, frozen


@frozen
class GroupElement:
    """Element of a group."""

    name: str
    transform: Callable[[torch.Tensor], torch.Tensor]
    inverse_transform: Callable[[torch.Tensor], torch.Tensor]

    @staticmethod
    def _compose(
        elements: list[GroupElement],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def inner(x: torch.Tensor) -> torch.Tensor:
            for element in reversed(elements):
                x = element.transform(x)
            return x

        return inner

    @staticmethod
    def _compose_inverse(
        elements: list[GroupElement],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def inner(x: torch.Tensor) -> torch.Tensor:
            for element in elements:
                x = element.inverse_transform(x)
            return x

        return inner

    @classmethod
    def from_group_elements(cls, group_elements: list[GroupElement]) -> GroupElement:
        """Compose a group element from a list of group elements.

        Args:
            group_elements: List of group elements.
        """

        return cls(
            name="_".join([element.name for element in group_elements]),
            transform=cls._compose(group_elements),
            inverse_transform=cls._compose_inverse(group_elements),
        )


@frozen
class DMBData:
    """A DMB data sample."""

    inputs: torch.Tensor
    outputs: torch.Tensor
    sample_id: str
    group_elements: list[GroupElement] = []


class DMBTransform(metaclass=ABCMeta):
    """A data augmentation protocol."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the input tensor."""


class InputOutputDMBTransform(metaclass=ABCMeta):
    """A data augmentation protocol for input-output pairs"""

    @abstractmethod
    def __call__(self, dmb_data: DMBData) -> DMBData:
        """Apply the transform to the input-output pair."""


class DMBDatasetTransform(metaclass=ABCMeta):
    """A dataset transform for DMB data."""

    @abstractmethod
    def __call__(self, dmb_data: DMBData) -> DMBData:
        """Apply the transform to the input-output pair."""

    @property
    @abstractmethod
    def mode(self) -> Literal["base", "train"]:
        """Return the mode of the transform."""

    @mode.setter
    @abstractmethod
    def mode(self, mode: Literal["base", "train"]) -> None:
        """Set the mode of the transform."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the transform."""


@define
class IdentityDMBDatasetTransform(DMBDatasetTransform):
    """An identity transform for DMB data."""

    mode: Literal["base", "train"] = "base"

    def __call__(self, dmb_data: DMBData) -> DMBData:
        return dmb_data

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"
