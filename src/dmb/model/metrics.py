from __future__ import annotations

from typing import Any, cast

import torch
import torchmetrics

from dmb.logging import create_logger

log = create_logger(__name__)


class MSE(torchmetrics.Metric):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.mse: torchmetrics.Metric = torchmetrics.MeanSquaredError()

    def compute(self) -> torch.Tensor:
        mse: torch.Tensor = self.mse.compute()
        return mse

    def update_impl(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.mse.update(preds.reshape(-1), target.reshape(-1))

    def update(
        self,
        preds: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor],
    ) -> None:
        """
        Args:
            preds (torch.Tensor): Predicted values.
            target (torch.Tensor): True values.
        """

        if isinstance(preds, (list, tuple)):
            for _preds, _target in zip(preds, target):
                self.update_impl(_preds, _target)
        else:
            self.update_impl(preds, cast(torch.Tensor, target))

    def to(self, *args: Any, **kwargs: Any) -> MSE:
        self.mse = self.mse.to(*args, **kwargs)

        return self

    def set_dtype(self, dtype: str | torch.dtype) -> MSE:
        self.mse = self.mse.set_dtype(dtype)

        return self

    def reset(self) -> None:
        self.mse.reset()
