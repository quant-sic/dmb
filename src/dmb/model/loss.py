"""Loss functions for training DMB models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import torch
from attrs import frozen

from dmb.data.collate import MultipleSizesBatch
from dmb.data.transforms import GroupElement
from dmb.logging import create_logger
from dmb.model.dmb_model import DMBModelOutput

log = create_logger(__name__)


@frozen
class LossOutput:
    """Output of a loss function."""

    loss: torch.Tensor
    loggables: dict[str, torch.Tensor] = {}


class Loss(ABC, torch.nn.Module):
    """Base class for loss functions."""

    def __init__(self, label_modification: torch.nn.Module | None = None) -> None:
        """Initialize the loss function.

        Args:
            label_modification (nn.Module, optional): Module to modify labels.
                Defaults to None.
        """
        super().__init__()
        self.label_modification = (
            label_modification if label_modification else torch.nn.Identity()
        )

    @abstractmethod
    def forward(
        self, model_output: DMBModelOutput, batch: MultipleSizesBatch
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""


class WeightedLoss(Loss):
    """Weighted loss function."""

    def __init__(
        self,
        constituent_losses: dict[str, Loss],
        weights: dict[str, float] | None = None,
        label_modification: torch.nn.Module | None = None,
    ) -> None:
        """Initialize the loss function.

        Args:
            constituent_losses (dict[str, Loss]): Dictionary of constituent losses.
            weights (dict[str, float], optional): Dictionary of weights for each
                constituent loss. Defaults to None.
        """
        super().__init__(label_modification=label_modification)
        self.constituent_losses = constituent_losses
        self.weights = weights

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        if self.weights is None:
            self.weights = {name: 1.0 for name in self.losses.keys()}

    def forward(
        self, model_output: DMBModelOutput, batch: MultipleSizesBatch
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        constituent_losses_values = {
            name: loss(model_output, batch).loss
            for name, loss in self.constituent_losses.items()
        }
        total_loss = sum(
            cast(dict[str, float], self.weights)[name] * loss_value
            for name, loss_value in constituent_losses_values.items()
        )

        return LossOutput(loss=total_loss, loggables=constituent_losses_values)


class EquivarianceErrorLoss(Loss):
    """Equivariance loss."""

    def __init__(self, label_modification: torch.nn.Module | None = None) -> None:
        """Initialize the loss function."""

        super().__init__(label_modification=label_modification)

    def forward_single_size(
        self,
        y_pred: torch.Tensor,
        sample_ids: list[str],
        group_elements: list[list[GroupElement]],
    ) -> tuple[torch.Tensor, int]:
        """Calculate the loss for a single size."""
        # get group elements
        accumulated_group_elements = [
            GroupElement.from_group_elements(group_elements)
            for group_elements in group_elements
        ]

        # all close does not capture variations in training well. Eg. due to Dropout
        acts_like_identity = [
            group_element.acts_like_identity(y_pred_sample)
            for y_pred_sample, group_element in zip(y_pred, accumulated_group_elements)
        ]

        # transform back to original sample
        y_pred_original = torch.stack(
            [
                group_element.inverse_transform(y_pred_sample)
                for y_pred_sample, group_element in zip(
                    y_pred, accumulated_group_elements
                )
            ]
        )
        losses = []

        # get indices of sample groups with same id
        for sample_id in set(sample_ids):
            sample_indices = [
                idx for idx, id in enumerate(sample_ids) if id == sample_id
            ]

            # if all group elements are identity, skip
            if all(acts_like_identity[idx] for idx in sample_indices):
                continue

            # variance
            if len(sample_indices) > 1:
                losses.append(torch.var(y_pred_original[sample_indices], dim=0))

        if len(losses) == 0:
            loss = torch.tensor(0.0)
        else:
            loss = torch.mean(torch.stack(losses))

        return loss, len(set(sample_ids))

    def forward(
        self, model_output: DMBModelOutput, batch: MultipleSizesBatch
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""
        losses_out, n_samples_losses = zip(
            *[
                self.forward_single_size(y_pred, sample_ids, group_elements)
                for y_pred, sample_ids, group_elements in zip(
                    model_output.inference_output,
                    batch.sample_ids,
                    batch.group_elements,
                )
            ]
        )

        loss = sum(
            loss * n_samples_loss
            for loss, n_samples_loss in zip(losses_out, n_samples_losses)
        ) / sum(n_samples_losses)

        return LossOutput(loss=loss)


class MSELoss(Loss):
    """Mean Squared Error loss function."""

    def __init__(
        self,
        *args: Any,
        reduction: Literal["mean", "size_mean"] = "mean",
        label_modification: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the loss function.

        Args:
            reduction (Literal["mean"], optional): The reduction method.
                Defaults to "mean".
        """
        super().__init__(label_modification=label_modification)

        self.mse_loss = torch.nn.MSELoss(reduction=reduction)
        self.reduction = reduction

    def forward_single_size(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Calculate the loss for a single size."""
        y_true_modified = self.label_modification(y_true)

        loss_out = self.mse_loss(y_pred.view(*y_true_modified.shape), y_true_modified)

        n_elements = y_true_modified.numel()

        return loss_out, n_elements

    def forward(
        self,
        model_output: DMBModelOutput,
        batch: MultipleSizesBatch,
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        losses, size_n_elements = zip(
            *[
                self.forward_single_size(y_pred, y_true)
                for y_pred, y_true in zip(model_output.loss_input, batch.outputs)
            ]
        )

        if self.reduction == "size_mean":
            loss: torch.Tensor = cast(torch.Tensor, sum(losses)) / len(losses)
        elif self.reduction == "mean":
            loss = sum(
                _loss * _n_elements
                for _loss, _n_elements in zip(losses, size_n_elements)
            ) / sum(size_n_elements)
        else:
            raise ValueError(f"Reduction {self.reduction} not supported.")

        return LossOutput(loss=loss)


class MSLELoss(Loss):
    """Mean Squared Logarithmic Error loss function."""

    def __init__(
        self,
        *args: Any,
        reduction: Literal["mean"] = "mean",
        label_modification: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(label_modification=label_modification)

        self.reduction = reduction

    def forward_single_size(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """
        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: Loss value.
        """
        y_true_modified = self.label_modification(y_true)
        loss = (
            torch.log1p(y_pred.view(*y_true_modified.shape))
            - torch.log1p(y_true_modified)
        ) ** 2

        valid_mask = loss.isfinite()

        loss_out: torch.Tensor = sum(loss[valid_mask]) / torch.sum(valid_mask)
        number_of_elements = int(torch.sum(valid_mask).item())

        return loss_out, number_of_elements

    def forward(
        self,
        model_output: DMBModelOutput,
        batch: MultipleSizesBatch,
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        losses, size_n_elements = zip(
            *[
                self.forward_single_size(y_pred, y_true)
                for y_pred, y_true in zip(model_output.loss_input, batch.outputs)
            ]
        )

        if self.reduction == "mean":
            loss: torch.Tensor = cast(
                torch.Tensor,
                sum(
                    _loss * _n_elements
                    for _loss, _n_elements in zip(losses, size_n_elements)
                )
                / sum(size_n_elements),
            )
        else:
            raise ValueError(f"Reduction {self.reduction} not supported.")

        return LossOutput(loss=loss)


class MAPELoss(Loss):
    """Mean Absolute Percentage Error loss function."""

    def __init__(self, label_modification: torch.nn.Module | None = None) -> None:
        """Initialize the loss function."""
        super().__init__(label_modification=label_modification)

    def forward_single_size(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Calculate the loss for a single size."""
        y_true_modified = self.label_modification(y_true)

        y_pred_v = y_pred.view(*y_true_modified.shape)
        denominator = torch.abs(y_true_modified) + torch.finfo(y_pred.dtype).eps
        loss = torch.abs((y_true_modified - y_pred_v) / denominator)

        valid_mask = loss.isfinite()

        loss_out = sum(loss[valid_mask]) / torch.sum(valid_mask)
        number_of_elements = int(torch.sum(valid_mask).item())

        return loss_out, number_of_elements

    def forward(
        self,
        model_output: DMBModelOutput,
        batch: MultipleSizesBatch,
    ) -> LossOutput:
        """Calculate the loss for the predicted and true values."""

        losses, size_n_elements = zip(
            *[
                self.forward_single_size(y_pred, y_true)
                for y_pred, y_true in zip(model_output.loss_input, batch.outputs)
            ]
        )

        loss = sum(
            _loss * _n_elements for _loss, _n_elements in zip(losses, size_n_elements)
        ) / sum(size_n_elements)

        return LossOutput(loss=loss)
