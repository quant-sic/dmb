"""Loss functions for training DMB models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import torch
from attrs import define, frozen

from dmb.data.collate import MultipleSizesBatch
from dmb.data.transforms import GroupElement
from dmb.logging import create_logger

log = create_logger(__name__)


@frozen
class LossOutput:
    """Output of a loss function."""

    loss: torch.Tensor
    loggables: dict[str, torch.Tensor] = {}


class Loss(ABC, torch.nn.Module):
    """Base class for loss functions."""

    @abstractmethod
    def forward(self, model_output: list[torch.Tensor],
                batch: MultipleSizesBatch) -> LossOutput:
        """Calculate the loss for the predicted and true values."""


@define(hash=False, eq=False)
class WeightedLoss(Loss):
    """Weighted loss function."""

    constituent_losses: dict[str, Loss]
    weights: dict[str, float] | None = None

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        if self.weights is None:
            self.weights = {name: 1.0 for name in self.losses.keys()}

    def forward(self, model_output: list[torch.Tensor],
                batch: MultipleSizesBatch) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        constituent_losses_values = {
            name: loss(model_output, batch).loss
            for name, loss in self.constituent_losses.items()
        }
        total_loss = sum(
            cast(dict[str, float], self.weights)[name] * loss_value
            for name, loss_value in constituent_losses_values.items())

        return LossOutput(loss=total_loss, loggables=constituent_losses_values)


class EquivarianceErrorLoss(Loss):
    """Equivariance loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the loss function."""

        super().__init__()

    def forward_single_size(
            self, y_pred: torch.Tensor, sample_ids: list[str],
            group_elements: list[list[GroupElement]]) -> tuple[torch.Tensor, int]:
        """Calculate the loss for a single size."""

        # transform back to original sample
        y_pred_original = torch.stack([
            GroupElement.from_group_elements(group_elements).inverse_transform(
                y_pred_sample)
            for y_pred_sample, group_elements in zip(y_pred, group_elements)
        ])
        losses = []

        # get indices of sample groups with same id
        for sample_id in sample_ids:
            sample_indices = [
                idx for idx, id in enumerate(sample_ids) if id == sample_id
            ]

            #variance
            losses.append(
                torch.var(
                    y_pred_original[sample_indices], dim=0, correction=1))

        loss = torch.mean(torch.stack(losses))

        return loss, len(set(sample_ids))

    def forward(self, model_output: list[torch.Tensor],
                batch: MultipleSizesBatch) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        losses_out, n_samples_losses = zip(*[
            self.forward_single_size(y_pred, sample_ids, group_elements)
            for y_pred, sample_ids, group_elements in zip(
                model_output, batch.sample_ids, batch.group_elements)
        ])

        loss = sum(loss * n_samples_loss for loss, n_samples_loss in zip(
            losses_out, n_samples_losses)) / sum(n_samples_losses)

        return LossOutput(loss=loss)


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
        model_output: list[torch.Tensor],
        batch: MultipleSizesBatch,
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        losses, size_n_elements = zip(*[
            self.forward_single_size(y_pred, y_true)
            for y_pred, y_true in zip(model_output, batch.outputs)
        ])

        if self.reduction == "size_mean":
            loss: torch.Tensor = cast(torch.Tensor, sum(losses)) / len(losses)
        elif self.reduction == "mean":
            loss = sum(_loss * _n_elements for _loss, _n_elements in zip(
                losses, size_n_elements)) / sum(size_n_elements)
        else:
            raise ValueError(f"Reduction {self.reduction} not supported.")

        return LossOutput(loss=loss)


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
        model_output: list[torch.Tensor],
        batch: MultipleSizesBatch,
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        losses, size_n_elements = zip(*[
            self.forward_single_size(y_pred, y_true)
            for y_pred, y_true in zip(model_output, batch.outputs)
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

        return LossOutput(loss=loss)
