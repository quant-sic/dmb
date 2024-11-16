"""Loss functions for training DMB models."""

from __future__ import annotations

from typing import Any, Literal, cast

import torch

from dmb.logging import create_logger

log = create_logger(__name__)


class MSELoss(torch.nn.Module):
    """Mean Squared Error loss function."""

    def __init__(
        self,
        *args: Any,
        reduction: Literal["mean", "size_mean"] = "mean",
        **kwargs: Any,
    ) -> None:
        """Initialize the loss function.

        Args:
            reduction (Literal["mean"], optional): The reduction method.
                Defaults to "mean".
        """
        super().__init__()

        self.reduction = reduction

    def forward_single_size(self, y_pred: torch.Tensor,
                            y_true: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Calculate the loss for a single size."""

        loss_out = torch.mean((y_true - y_pred.view(*y_true.shape))**2)
        n_elements = y_true.numel()

        return loss_out, n_elements

    def forward(
        self,
        y_pred: list[torch.Tensor] | torch.Tensor,
        y_true: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for the predicted and true values."""

        if isinstance(y_pred, torch.Tensor) and isinstance(y_true, torch.Tensor):
            y_pred = [y_pred]
            y_true = [y_true]
        elif isinstance(y_pred, list) and isinstance(y_true, list):
            pass
        else:
            raise ValueError(
                "y_pred and y_true must be of the same type. "
                f"Type y_pred: {type(y_pred)}, type y_true: {type(y_true)}")

        losses, size_n_elements = zip(*[
            self.forward_single_size(y_pred_, y_true_)
            for y_pred_, y_true_ in zip(y_pred, y_true)
        ])

        if self.reduction == "size_mean":
            loss: torch.Tensor = cast(torch.Tensor, sum(losses)) / len(losses)
        elif self.reduction == "mean":
            loss = sum(_loss * _n_elements for _loss, _n_elements in zip(
                losses, size_n_elements)) / sum(size_n_elements)
        else:
            raise ValueError(f"Reduction {self.reduction} not supported.")

        return loss


class MSLELoss(torch.nn.Module):
    """Mean Squared Logarithmic Error loss function."""

    def __init__(
        self,
        *args: Any,
        reduction: Literal["mean"] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.reduction = reduction

    def forward_single_size(self, y_pred: torch.Tensor,
                            y_true: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = torch.log((y_true + 1) / (y_pred + 1))**2
        valid_mask = loss.isfinite()

        loss_out: torch.Tensor = sum(loss[valid_mask]) / torch.sum(valid_mask)
        number_of_elements = int(torch.sum(valid_mask).item())

        return loss_out, number_of_elements

    def forward(
        self,
        y_pred: list[torch.Tensor] | torch.Tensor,
        y_true: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for the predicted and true values."""

        if isinstance(y_pred, torch.Tensor) and isinstance(y_true, torch.Tensor):
            y_pred = [y_pred]
            y_true = [y_true]
        elif isinstance(y_pred, list) and isinstance(y_true, list):
            pass
        else:
            raise ValueError(
                "y_pred and y_true must be of the same type. "
                f"Type y_pred: {type(y_pred)}, type y_true: {type(y_true)}")

        losses, size_n_elements = zip(*[
            self.forward_single_size(y_pred_, y_true_)
            for y_pred_, y_true_ in zip(y_pred, y_true)
        ])

        if self.reduction == "mean":
            loss: torch.Tensor = cast(
                torch.Tensor,
                sum(_loss * _n_elements
                    for _loss, _n_elements in zip(losses, size_n_elements)) /
                sum(size_n_elements),
            )
        else:
            raise ValueError(f"Reduction {self.reduction} not supported.")

        return loss
