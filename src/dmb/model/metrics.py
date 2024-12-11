"""Metrics for model evaluation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torchmetrics
from torchmetrics.functional.regression.mse import \
    _mean_squared_error_compute, _mean_squared_error_update

from dmb.logging import create_logger

log = create_logger(__name__)


class MinMSE(torchmetrics.Metric):
    """Computes the minimum Mean Squared Error (MSE) metric over the batch dimension."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    sum_squared_error: torch.Tensor
    total: torch.Tensor

    def __init__(self,
                 *args: Any,
                 squared: bool = True,
                 num_outputs: int = 1,
                 **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Initialize the Mean Squared Error (MSE) metric."""

        super().__init__(**kwargs)

        if not isinstance(squared, bool):
            raise ValueError(
                f"Expected argument `squared` to be a boolean but got {squared}")
        self.squared = squared

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(
                f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("sum_squared_error",
                       default=torch.zeros(self.num_outputs),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self) -> torch.Tensor:
        """Compute the Mean Squared Error (MSE) metric."""
        return _mean_squared_error_compute(self.sum_squared_error,
                                           self.total,
                                           squared=self.squared)

    def update_single_size(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the Mean Squared Error (MSE) metric."""

        mse_samples = ((preds - target)**2).sum(dim=tuple(range(1, preds.ndim)))
        min_mse_sample_idx = mse_samples.argmin()

        sum_squared_error, num_obs = mse_samples[min_mse_sample_idx], np.prod(
            preds.shape[1:])

        self.sum_squared_error = self.sum_squared_error.clone() + sum_squared_error
        self.total = self.total.clone() + num_obs

    def update(
        self,
        preds: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor],
    ) -> None:
        """
        Update the Mean Squared Error (MSE) metric.

        Args:
            preds (torch.Tensor): Predicted values.
            target (torch.Tensor): True values.
        """

        if isinstance(preds, (list, tuple)):
            for _preds, _target in zip(preds, target):
                self.update_single_size(_preds, _target)
        else:
            self.update_single_size(preds, cast(torch.Tensor, target))


class MSE(torchmetrics.Metric):
    """Mean Squared Error (MSE) metric."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    sum_squared_error: torch.Tensor
    total: torch.Tensor

    def __init__(self,
                 *args: Any,
                 squared: bool = True,
                 num_outputs: int = 1,
                 **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Initialize the Mean Squared Error (MSE) metric."""

        super().__init__(**kwargs)

        if not isinstance(squared, bool):
            raise ValueError(
                f"Expected argument `squared` to be a boolean but got {squared}")
        self.squared = squared

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(
                f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("sum_squared_error",
                       default=torch.zeros(self.num_outputs),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute(self) -> torch.Tensor:
        """Compute the Mean Squared Error (MSE) metric."""
        return _mean_squared_error_compute(self.sum_squared_error,
                                           self.total,
                                           squared=self.squared)

    def update_single_size(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the Mean Squared Error (MSE) metric."""
        sum_squared_error, num_obs = _mean_squared_error_update(
            preds.reshape(-1), target.reshape(-1), num_outputs=self.num_outputs)

        self.sum_squared_error = self.sum_squared_error.clone() + sum_squared_error
        self.total = self.total.clone() + num_obs

    def update(
        self,
        preds: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor],
    ) -> None:
        """
        Update the Mean Squared Error (MSE) metric.

        Args:
            preds (torch.Tensor): Predicted values.
            target (torch.Tensor): True values.
        """

        if isinstance(preds, (list, tuple)):
            for _preds, _target in zip(preds, target):
                self.update_single_size(_preds, _target)
        else:
            self.update_single_size(preds, cast(torch.Tensor, target))

    def reset(self) -> None:
        """Reset the Mean Squared Error (MSE) metric."""
        self.sum_squared_error = torch.zeros(self.num_outputs,
                                             device=self.sum_squared_error.device)
        self.total = torch.tensor(0, device=self.total.device)
