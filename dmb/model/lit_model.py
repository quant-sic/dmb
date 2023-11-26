from functools import cached_property
from typing import Any, Dict, List, Literal

import hydra
import lightning.pytorch as pl
import torch

from dmb.model.mixins import LitModelMixin
from dmb.utils import create_logger

log = create_logger(__name__)


class DMBLitModel(pl.LightningModule, LitModelMixin):
    def __init__(
        self,
        model: Dict[str, Any],
        optimizer: Dict[str, Any],
        scheduler: Dict[str, Any],
        loss: Dict[str, Any],
        observables: List[str],
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        # instantiate the decoder
        self.model = self.load_model(model)

        # serves as a dummy input for the model to get the output shape with lightning
        self.example_input_array = torch.zeros(
            1, self.hparams["model"]["in_channels"], 10, 10
        )

    @staticmethod
    def load_model(model_dict):
        return hydra.utils.instantiate(model_dict, _recursive_=False, _convert_="all")

    @property
    def observables(self):
        return self.hparams["observables"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch_in, batch_label = batch

        # forward pass
        model_out = self(batch_in)

        # compute loss
        loss = self.loss(model_out, batch_label)

        # log metrics
        self.compute_and_log_metrics(model_out, batch_label, "train", loss)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch_in, batch_label = batch

        # forward pass
        model_out = self(batch_in)

        # compute loss
        loss = self.loss(model_out, batch_label)

        # log metrics
        self.compute_and_log_metrics(model_out, batch_label, "val", loss)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch_in, batch_label = batch

        # forward pass
        model_out = self(batch_in)

        # compute loss
        loss = self.loss(model_out, batch_label)

        # log metrics
        self.compute_and_log_metrics(model_out, batch_label, "test", loss)

        return loss

    def on_train_epoch_end(self) -> None:
        """Reset metrics at the end of the epoch."""
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Reset metrics at the end of the epoch."""
        self.val_metrics.reset()

    def reset_metrics(self, stage: Literal["train", "val", "test"]) -> None:
        """Reset metrics."""
        # get metrics for the current stage
        metrics_dict = getattr(self, f"{stage}_metrics")
        for metric_name, (metric, lit_module_attribute) in metrics_dict.items():
            metric.reset()

    @cached_property
    def loss(self) -> torch.nn.Module:
        _loss: torch.nn.Module = hydra.utils.instantiate(self.hparams["loss"])

        log.info("Using {} as loss".format(_loss.__class__.__name__))

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
