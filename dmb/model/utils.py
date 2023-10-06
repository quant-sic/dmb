from __future__ import annotations

import torch
import torchmetrics
from typing import Any, Optional, Literal,Union,List

from dmb.utils import create_logger

log = create_logger(__name__)

class MaskedMSE(torchmetrics.Metric):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.mse = torchmetrics.MeanSquaredError()

    def compute(self) -> torch.Tensor:
        return self.mse.compute()

    def update_impl(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> None:

        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)

        if weight is None:
            weight = torch.ones_like(target)

        valid_mask = mask

        # log shapes
        log.debug(
            f"preds: {preds.shape}, target: {target.shape}, mask: {mask.shape}, {mask.flatten().shape}"
        )

        self.mse.update(preds[valid_mask], target[valid_mask])

    def update(
        self, preds: Union[torch.Tensor,List[torch.Tensor]], target: Union[torch.Tensor,List[torch.Tensor]], mask: Union[torch.Tensor,List[torch.Tensor]]=None, weight: Union[torch.Tensor,List[torch.Tensor]]=None
    ) -> None:
        """
        Args:
            preds (torch.Tensor): Predicted values.
            target (torch.Tensor): True values.
        """

        if isinstance(preds, (list,tuple)):
                
            if mask is None:
                mask = [None]*len(preds)
            
            if weight is None:
                weight = [None]*len(preds)

            for _, (preds_, target_, mask_, weight_) in enumerate(zip(preds, target, mask, weight)):
                self.update_impl(preds_, target_, mask_, weight_)
        else:
            self.update_impl(preds, target, mask, weight)


    def to(self, dst) -> MaskedMSE:
        self.mse = self.mse.to(dst)

        return self

    def set_dtype(self, dtype: torch.dtype) -> MaskedMSE:
        self.mse = self.mse.set_dtype(dtype)

        return self

    def reset(self) -> None:
        self.mse.reset()


class MaskedMSELoss(torch.nn.Module):
    def __init__(
        self,
        reduction: Optional[Literal["mean"]] = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__()

        self.reduction = reduction

    def forward_impl(self,y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        if mask is None:
            mask = torch.ones_like(y_true, dtype=torch.bool)

        loss = (y_true - y_pred.view(*y_true.shape)) ** 2

        if self.reduction == "mean":
            try:
                loss_out = torch.sum(loss.view(mask.shape) * mask) / torch.sum(mask)
            except RuntimeError:
                # log shapes
                log.info(
                    f"y_pred: {y_pred.shape}, y_true: {y_true.shape}, mask: {mask.shape}, {mask.flatten().shape}"
                )
                raise
        elif self.reduction is None:
            loss_out = loss * mask

        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return loss_out

    def forward(
        self, y_pred: Union[List[torch.Tensor],torch.Tensor], y_true: Union[List[torch.Tensor],torch.Tensor], mask: Union[List[torch.Tensor],torch.Tensor] = None
    ) -> torch.Tensor:

        if isinstance(y_pred, (list, tuple)):
            
            if mask is None:
                mask = [None]*len(y_pred)

            for idx, (y_pred_, y_true_, mask_) in enumerate(zip(y_pred, y_true, mask)):
                if idx == 0:
                    loss = self.forward_impl(y_pred_, y_true_, mask_)
                else:
                    loss += self.forward_impl(y_pred_, y_true_, mask_)

        else:
            loss = self.forward_impl(y_pred, y_true, mask)

        return loss
    

class MSLELoss(torch.nn.Module):
    def __init__(
        self,
        reduction: Optional[Literal["mean"]] = "mean",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.reduction = reduction

    def forward_impl(self,y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        """
        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: Loss value.
        """
        # log shapes
        # log.info(f"y_pred: {y_pred.shape}, y_true: {y_true.shape}, mask: {mask.shape}")

        y_pred = y_pred[mask].clone()
        y_true = y_true[mask].clone()

        loss = torch.log((y_true + 1 )/ (y_pred + 1)) ** 2

        valid_mask = loss.isfinite()

        if self.reduction == "mean":
            loss_out = sum(loss[valid_mask]) / torch.sum(valid_mask)

        elif self.reduction is None:
            loss_out = loss * valid_mask
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        if not torch.isfinite(loss_out).all():
            raise RuntimeError(
                f"Loss is NaN. y_pred: {y_pred}, y_true: {y_true}, mask: {valid_mask}"
            )

        # log.info(f"loss_out: {loss_out}. MASK: {mask}. y_pred: {y_pred}, y_true: {y_true}")

        return loss_out

    def forward(
            self, y_pred: Union[List[torch.Tensor],torch.Tensor], y_true: Union[List[torch.Tensor],torch.Tensor], mask: Union[List[torch.Tensor],torch.Tensor] = None
        ) -> torch.Tensor:
        
        if isinstance(y_pred, (list, tuple)):
            
            if mask is None:
                mask = [None]*len(y_pred)

            for idx, (y_pred_, y_true_, mask_) in enumerate(zip(y_pred, y_true, mask)):
                if idx == 0:
                    loss = self.forward_impl(y_pred_, y_true_, mask_)
                else:
                    loss += self.forward_impl(y_pred_, y_true_, mask_)

        else:
            loss = self.forward_impl(y_pred, y_true, mask)

        return loss
