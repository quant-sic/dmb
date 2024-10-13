from typing import Protocol

import torch


class DMBAugmentation(Protocol):
    """A data augmentation protocol."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class InputOutputDMBAugmentation(Protocol):
    """A data augmentation protocol for input-output pairs"""

    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
