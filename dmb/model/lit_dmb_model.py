from __future__ import annotations
import functools
from collections.abc import Mapping
from typing import Any, Literal, cast

import lightning.pytorch as pl
import torch
import torchmetrics
from attrs import define
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torchmetrics
import itertools
import yaml
from omegaconf import DictConfig
import hydra

from dmb.logging import create_logger

log = create_logger(__name__)


@define(hash=False, eq=False)
class LitDMBModel(pl.LightningModule):

    model: torch.nn.Module
    optimizer: functools.partial[Optimizer]
    lr_scheduler: functools.partial[dict[str, functools.partial[_LRScheduler | Any]]]
    loss: torch.nn.Module
    metrics: torchmetrics.MetricCollection

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.example_input_array = torch.zeros(1, 4, 10, 10)

    @classmethod
    def load_from_logged_checkpoint(
        cls, log_dir: Path, checkpoint_path: Path
    ) -> LitDMBModel:
        """Load a model from a checkpoint.

        Args:
            log_dir: The directory of the hydra log.
            checkpoint_path: The path to the checkpoint file.
                Contains the state_dict of the model.

        Returns:
            The loaded model.
        """
        with open(log_dir / ".hydra/config.yaml", encoding="utf-8") as file:
            config = DictConfig(yaml.load(file, Loader=yaml.FullLoader))

        # keep "lit_model" key, but remove "_target_" key, such that config is
        # resolvable but lit_model is not instantiated
        config_without_target = {
            "lit_model": {k: v for k, v in config.lit_model.items() if k != "_target_"}
        }
        model: LitDMBModel = cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **hydra.utils.instantiate(config_without_target)["lit_model"],
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _calculate_loss(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch_in, batch_label = batch

        model_out = self(batch_in)
        loss = self.loss(model_out, batch_label)

        return model_out, loss

    def _evaluate_metrics(self, batch: Any, model_out: torch.Tensor) -> None:
        batch_in, batch_label = batch
        self.metrics.update(preds=model_out, target=batch_label)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        model_out, loss = self._calculate_loss(batch)
        self._evaluate_metrics(batch, model_out)

        # log metrics
        batch_size = sum(len(b) for b in batch)
        self.log_metrics(
            stage="train",
            metric_collection=dict(
                itertools.chain(self.metrics.items(), {"loss": loss}.items())
            ),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        model_out, loss = self._calculate_loss(batch)
        self._evaluate_metrics(batch, model_out)

        # log metrics
        batch_size = sum(len(b) for b in batch)
        self.log_metrics(
            stage="val",
            metric_collection=dict(
                itertools.chain(self.metrics.items(), {"loss": loss}.items())
            ),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        model_out, loss = self._calculate_loss(batch)
        self._evaluate_metrics(batch, model_out)

        # log metrics
        batch_size = sum(len(b) for b in batch)
        self.log_metrics(
            stage="test",
            metric_collection=dict(
                itertools.chain(self.metrics.items(), {"loss": loss}.items())
            ),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and scheduler."""
        optimizer: torch.optim.Optimizer = self.optimizer(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
        )

        configuration: dict[str, Any] = {"optimizer": optimizer}

        if self.lr_scheduler is not None:
            scheduler: _LRScheduler = self.lr_scheduler["scheduler"](
                optimizer=optimizer
            )
            configuration["lr_scheduler"] = {
                **self.lr_scheduler,
                "scheduler": scheduler,
            }

        return cast(OptimizerLRScheduler, configuration)

    def log_metrics(
        self,
        stage: Literal["train", "val", "test"],
        metric_collection: Mapping[str, torch.Tensor | torchmetrics.Metric],
        on_step: bool,
        on_epoch: bool,
        batch_size: int,
    ) -> None:
        """Log metrics."""
        for metric_name, metric in metric_collection.items():

            computed_metric = (
                metric.compute() if isinstance(metric, torchmetrics.Metric) else metric
            )

            if isinstance(computed_metric, dict):
                loggable = {
                    f"{stage}/{metric_name}/{key}": value
                    for key, value in computed_metric.items()
                }
            else:
                loggable = {f"{stage}/{metric_name}": computed_metric}

            self.log_dict(
                loggable,
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=batch_size,
            )

    def on_train_epoch_end(self) -> None:
        """Execute at the end of each training epoch."""
        self.metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Execute at the end of each validation epoch."""
        self.metrics.reset()
