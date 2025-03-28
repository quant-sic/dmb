"""Lightning module for DMB models."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import lightning.pytorch as pl
import torch
import torchmetrics
import yaml
from attrs import define, field, frozen
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from dmb.data.collate import MultipleSizesBatch
from dmb.logging import create_logger
from dmb.model.dmb_model import DMBModel
from dmb.model.loss import Loss, LossOutput
from dmb.paths import REPO_LOGS_ROOT

log = create_logger(__name__)


@frozen
class WeightsCheckpoint:
    """Weights checkpoint path configuration.

    Attributes:
        path: The path to the checkpoint file.
        state_dict: The state_dict of the model.
    """

    path: Path
    state_dict: dict[str, Any] = field(init=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_dict",
            torch.load(
                REPO_LOGS_ROOT / self.path,
                weights_only=True,
                map_location=torch.device("cpu"),
            )["state_dict"],
        )


@define(hash=False, eq=False)
class LitDMBModel(pl.LightningModule):
    """Lightning module for DMB models."""

    model: DMBModel
    optimizer: functools.partial[Optimizer]
    lr_scheduler: dict[str, functools.partial[_LRScheduler | Any]]
    loss: Loss
    metrics: torchmetrics.MetricCollection
    weights_checkpoint: WeightsCheckpoint | None = None

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        self.example_input_array = torch.zeros(1, 4, 10, 10)

        if self.weights_checkpoint is not None:
            self.load_state_dict(self.weights_checkpoint.state_dict, strict=True)

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
        with open(
            REPO_LOGS_ROOT / log_dir / ".hydra/config.yaml", encoding="utf-8"
        ) as file:
            config = DictConfig(yaml.load(file, Loader=yaml.FullLoader))

        # keep "lit_model" key, but remove "_target_" key, such that config is
        # resolvable but lit_model is not instantiated
        config_without_target = {
            "lit_model": {
                k: v
                for k, v in config.lit_model.items()
                if k not in ("_target_", "weights_checkpoint")
            }
        }
        model: LitDMBModel = cls.load_from_checkpoint(
            checkpoint_path=REPO_LOGS_ROOT / checkpoint_path,
            **hydra.utils.instantiate(config_without_target)["lit_model"],
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(x)
        return out

    def _calculate_loss(
        self, batch: MultipleSizesBatch
    ) -> tuple[torch.Tensor, LossOutput]:
        model_out = self(batch.inputs)
        loss_output = self.loss(model_out, batch)

        return model_out, loss_output

    def _evaluate_metrics(
        self, batch: MultipleSizesBatch, model_out: torch.Tensor
    ) -> None:
        self.metrics.update(preds=model_out, target=batch.outputs)

    def training_step(
        self,
        batch: MultipleSizesBatch,
    ) -> torch.Tensor:
        model_out, loss_output = self._calculate_loss(batch)
        self._evaluate_metrics(batch, model_out)

        # log metrics
        self.log_metrics(
            stage="train",
            metric_collection=dict(
                itertools.chain(
                    self.metrics.items(),
                    {"loss": loss_output.loss, **loss_output.loggables}.items(),
                )
            ),
            on_step=True,
            on_epoch=True,
            batch_size=batch.size,
        )

        return loss_output.loss

    def validation_step(self, batch: MultipleSizesBatch) -> None:
        model_out, loss_output = self._calculate_loss(batch)
        self._evaluate_metrics(batch, model_out)

        # log metrics
        self.log_metrics(
            stage="val",
            metric_collection=dict(
                itertools.chain(
                    self.metrics.items(),
                    {"loss": loss_output.loss, **loss_output.loggables}.items(),
                )
            ),
            on_step=False,
            on_epoch=True,
            batch_size=batch.size,
        )

    def test_step(self, batch: MultipleSizesBatch) -> None:
        model_out, loss_output = self._calculate_loss(batch)
        self._evaluate_metrics(batch, model_out)

        # log metrics
        self.log_metrics(
            stage="test",
            metric_collection=dict(
                itertools.chain(
                    self.metrics.items(),
                    {"loss": loss_output.loss, **loss_output.loggables}.items(),
                )
            ),
            on_step=False,
            on_epoch=True,
            batch_size=batch.size,
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

    def on_train_start(self) -> None:
        """Execute at the start of training."""
        self.model.train()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Execute when loading a checkpoint."""
        super().on_load_checkpoint(checkpoint)

        # # print optimizer state
        # optimizer = self.trainer.optimizers[0]
        # # Access the exp_avg and exp_avg_sq for each parameter
        # for i, param in enumerate(optimizer.param_groups[0]):
        #     if param in optimizer.state:
        #         state = optimizer.state[param]

        #         #lr
        #         print("lr:\n{}".format(
        #         # print("exp_avg:\n{}".format(exp_avg))
        #         # print("exp_avg_sq:\n{}".format(exp_avg_sq))

    def on_train_epoch_end(self) -> None:
        """Execute at the end of each training epoch."""
        self.metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Execute at the end of each validation epoch."""
        self.metrics.reset()
