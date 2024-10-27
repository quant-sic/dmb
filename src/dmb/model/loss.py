from __future__ import annotations

from typing import Any, Literal, cast

import torch

from dmb.logging import create_logger

log = create_logger(__name__)


class MSELoss(torch.nn.Module):

    def __init__(
        self,
        reduction: Literal["mean"] | None = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.reduction = reduction

    def forward_impl(self, y_pred: torch.Tensor,
                     y_true: torch.Tensor) -> torch.Tensor:
        loss = (y_true - y_pred.view(*y_true.shape))**2

        if self.reduction == "mean":
            loss_out = torch.mean(loss)

        return loss_out

    def forward(
        self,
        y_pred: list[torch.Tensor] | torch.Tensor,
        y_true: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(y_pred, torch.Tensor) and isinstance(
                y_true, torch.Tensor):
            y_pred = [y_pred]
            y_true = [y_true]
        else:
            raise ValueError("y_pred and y_true must be of the same type")

        loss: torch.Tensor = cast(
            torch.Tensor,
            sum(
                self.forward_impl(y_pred_, y_true_)
                for y_pred_, y_true_ in zip(y_pred, y_true)))

        return loss


class MSLELoss(torch.nn.Module):

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.reduction = reduction

    def forward_impl(self, y_pred: torch.Tensor,
                     y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = torch.log((y_true + 1) / (y_pred + 1))**2

        valid_mask = loss.isfinite()

        if self.reduction == "mean":
            loss_out: torch.Tensor = sum(
                loss[valid_mask]) / torch.sum(valid_mask)

        return loss_out

    def forward(
        self,
        y_pred: list[torch.Tensor] | torch.Tensor,
        y_true: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(y_pred, torch.Tensor) and isinstance(
                y_true, torch.Tensor):
            y_pred = [y_pred]
            y_true = [y_true]
        else:
            raise ValueError("y_pred and y_true must be of the same type")

        loss: torch.Tensor = cast(
            torch.Tensor,
            sum(
                self.forward_impl(y_pred_, y_true_)
                for y_pred_, y_true_ in zip(y_pred, y_true)))

        return loss
