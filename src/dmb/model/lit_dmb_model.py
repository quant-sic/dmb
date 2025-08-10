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
from attrs import field, frozen
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from dmb.data.collate import MultipleSizesBatch
from dmb.logging import create_logger
from dmb.model.dmb_model import DMBModel, DMBModelOutput
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


class LitDMBModel(pl.LightningModule):
    """Lightning module for DMB models."""

    def __init__(
        self,
        model: DMBModel,
        optimizer: functools.partial[Optimizer],
        lr_scheduler: dict[str, functools.partial[_LRScheduler | Any]],
        loss: Loss,
        metrics: torchmetrics.MetricCollection,
        evaluation_data_metrics: torchmetrics.MetricCollection | None = None,
        weights_checkpoint: WeightsCheckpoint | None = None,
    ) -> None:
        """Initialize the LitDMBModel.

        Args:
            model: The DMB model.
            optimizer: The optimizer for the model.
            lr_scheduler: The learning rate scheduler for the model.
            loss: The loss function for the model.
            metrics: The metrics for the model.
            weights_checkpoint: The weights checkpoint to load from.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.metrics = metrics
        self.evaluation_data_metrics = evaluation_data_metrics
        self.weights_checkpoint = weights_checkpoint

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

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> DMBModelOutput:
        out: DMBModelOutput = self.model(x)
        return out

    def _calculate_loss(
        self, batch: MultipleSizesBatch
    ) -> tuple[DMBModelOutput, LossOutput]:
        model_out = self(batch.inputs)
        loss_output = self.loss(model_out, batch)

        return model_out, loss_output

    def _evaluate_metrics(
        self, batch: MultipleSizesBatch, model_out: DMBModelOutput
    ) -> None:
        self.metrics.update(preds=model_out.inference_output, target=batch.outputs)

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

    def validation_step(
        self, batch: MultipleSizesBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Validation step for the model.

        Args:
            dataloader_idx: The index of the dataloader.
            batch: The batch of data.
        """

        dataloader_name = self.trainer.datamodule.val_dataloader_names[dataloader_idx]

        if dataloader_name == "val":
            model_out, loss_output = self._calculate_loss(batch)
            self._evaluate_metrics(batch, model_out)

            _metric_collection = dict(
                itertools.chain(
                    self.metrics.items(),
                    {"loss": loss_output.loss, **loss_output.loggables}.items(),
                )
            )

        elif dataloader_name == "val/data":
            if self.evaluation_data_metrics is not None:
                self.evaluation_data_metrics.update(
                    preds=[
                        mapped[:, 0].unsqueeze(1)
                        for mapped in self(batch.inputs).inference_output
                    ],
                    target=[output.unsqueeze(1) for output in batch.outputs],
                )

                _metric_collection = dict(
                    itertools.chain(self.evaluation_data_metrics.items())
                )
            else:
                _metric_collection = {}

        # log metrics
        self.log_metrics(
            stage=dataloader_name,
            metric_collection=_metric_collection,
            on_step=False,
            on_epoch=True,
            batch_size=batch.size,
        )

    def test_step(
        self, batch: MultipleSizesBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
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

        # some schedulers require the initial learning rate to be set
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

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
        stage: Literal["train", "val", "val/data", "test"],
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
                add_dataloader_idx=False,
            )

    def on_train_start(self) -> None:
        """Execute at the start of training."""
        self.model.train()

    def on_train_epoch_end(self) -> None:
        """Execute at the end of each training epoch."""
        self.metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Execute at the end of each validation epoch."""
        self.metrics.reset()
