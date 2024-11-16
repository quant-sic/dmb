"""Data transforms."""

from abc import ABCMeta, abstractmethod
from typing import Literal

import torch
from attrs import define


class DMBTransform(metaclass=ABCMeta):
    """A data augmentation protocol."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the input tensor."""


class InputOutputDMBTransform(metaclass=ABCMeta):
    """A data augmentation protocol for input-output pairs"""

    @abstractmethod
    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the transform to the input-output pair."""


class DMBDatasetTransform(metaclass=ABCMeta):
    """A dataset transform for DMB data."""

    @abstractmethod
    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"
