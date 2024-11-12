"""Constraints for the model output."""

import torch


class Exponential(torch.nn.Module):
    """Exponential conatraint."""

    def __init__(self, eps: float = 1e-10) -> None:
        """Initialize Exponential constraint.

        Args:
            eps: Small value to prevent numerical instability.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
        """
        return torch.exp(x) + self.eps
