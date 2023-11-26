from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, cast

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics
from torchmetrics import MetricCollection

from dmb.model.utils import MaskedMSE, MaskedMSELoss
from dmb.utils import create_logger

log = create_logger(__name__)


class LitModelMixin:
    @property
    def metrics(self) -> MetricCollection:
        metric_collection = MetricCollection(
            {
                "mse": MaskedMSE(),
            }
        )

        module_device: torch.device = (
            self.model.parameters().__next__().device
        )  # self.device does not work due to mypy error
        metric_collection.to(module_device)

        return metric_collection

    @cached_property
    def val_metrics(self) -> MetricCollection:
        return self.metrics.clone(prefix="val/")

    @cached_property
    def train_metrics(self) -> MetricCollection:
        return self.metrics.clone(prefix="train/")

    @cached_property
    def test_metrics(self) -> MetricCollection:
        return self.metrics.clone(prefix="test/")

    @cached_property
    def loss(self) -> torch.nn.Module:
        _loss: torch.nn.Module = hydra.utils.instantiate(self.hparams["loss"])

        log.info("Using {} for {} task.".format(_loss.__class__.__name__))

        return _loss

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        # filter required for fine-tuning
        _optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.hparams["optimizer"],
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
        )

        if self.hparams["scheduler"] is None:
            return {"optimizer": _optimizer}
        else:
            _scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(
                self.hparams["scheduler"], optimizer=_optimizer
            )

            return {
                "optimizer": _optimizer,
                "lr_scheduler": {
                    "scheduler": _scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

    def compute_and_log_metrics(
        self,
        model_out: torch.Tensor,
        _label: torch.Tensor,
        stage: Literal["train", "val", "test"],
        loss: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute and log metrics.

        Args:
            model_out (torch.Tensor): Model output.
            _label (torch.Tensor): Label.
            stage (Literal["train", "val", "test"]): Current stage.
        """

        # log loss
        if loss is not None:
            self.log_dict(
                {f"{stage}/loss": loss},
                on_epoch=True,
                on_step=(stage == "train"),
            )

        # get metrics for the current stage
        metrics_collection = getattr(self, f"{stage}_metrics")
        for metric_name, metric in metrics_collection.items():
            # if metric update takes mask
            metric.update(model_out, _label, mask=mask)

            # log metric
            computed_metric = metric.compute()
            log_dict = (
                {metric_name: computed_metric}
                if not isinstance(computed_metric, dict)
                else {
                    f"{metric_name}/{key}": value
                    for key, value in computed_metric.items()
                }
            )

            # log on step only for training
            log_on_step = stage == "train"

            self.log_dict(
                log_dict,
                on_epoch=True,
                on_step=log_on_step,
            )


class DMBLitModel(pl.LightningModule, LitModelMixin):
    def __init__(
        self,
        model: Dict[str, Any],
        optimizer: Dict[str, Any],
        scheduler: Dict[str, Any],
        loss: Dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        # instantiate the decoder
        self.model = self.load_model(model)

        # serves as a dummy input for the model to get the output shape with lightning
        # self.example_input_array = torch.zeros(
        #     1,
        #     getattr(_encoder, "window_size", 1),
        #     getattr(_encoder, "window_size", 1),
        #     _encoder.input_dim,
        # )

    def load_model(model_dict):
        return hydra.utils.instantiate(model_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch_in, batch_label, batch_mask = batch

        # forward pass
        model_out = self(batch_in)

        # compute loss
        loss = self.loss(model_out, batch_label, mask=batch_mask)

        # log metrics
        self.compute_and_log_metrics(
            model_out, batch_in, "train", loss, mask=batch_mask
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch_in, batch_label, batch_mask = batch

        # forward pass
        model_out = self(batch_in)

        # compute loss
        loss = self.loss(model_out, batch_label, mask=batch_mask)

        # log metrics
        self.compute_and_log_metrics(
            model_out, batch_label, "val", loss, mask=batch_mask
        )

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch_in, batch_label, batch_mask = batch

        # forward pass
        model_out = self(batch_in)

        # compute loss
        loss = self.loss(model_out, batch_label, mask=batch_mask)

        # log metrics
        self.compute_and_log_metrics(
            model_out, batch_label, "test", loss, mask=batch_mask
        )

        return loss

    def reset_metrics(self, stage: Literal["train", "val", "test"]) -> None:
        """Reset metrics."""
        # get metrics for the current stage
        metrics_dict = getattr(self, f"{stage}_metrics")
        for metric_name, (metric, lit_module_attribute) in metrics_dict.items():
            metric.reset()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """Reset metrics at the end of the epoch."""
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Reset metrics at the end of the epoch."""
        self.val_metrics.reset()
