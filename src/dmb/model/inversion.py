import functools
import itertools
from typing import Any, Literal, Mapping, cast

import torch
import torchmetrics
from attrs import define, field
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import numpy as np
from dmb.data.collate import MultipleSizesBatch
from dmb.model.dmb_model import DMBModel
from dmb.model.loss import Loss, LossOutput
from dmb.data.bose_hubbard_2d.nn_input import get_nn_input_dimless_const_parameters
from dmb.model.lit_dmb_model import LitDMBModel
from pathlib import Path
from dmb.paths import REPO_DATA_ROOT
import matplotlib.pyplot as plt

class InversionResult(torch.nn.Module):

    def __init__(self, shape:tuple[int,...],input_parameters:dict[str,float|torch.Tensor|np.ndarray]
    ) -> None:
        super().__init__()

        self.input_parameters = input_parameters
        self.inversion_result = torch.nn.Parameter(torch.empty(*shape),
                                                   requires_grad=True)
        torch.nn.init.xavier_uniform_(self.inversion_result)

    def forward(self) -> torch.Tensor:
        nn_input = get_nn_input_dimless_const_parameters(
            muU=self.inversion_result,
            **self.input_parameters)
        return nn_input

class InversionFakeDataLoader:
    def __init__(self) -> None:
        self.output = "fake"

    def __iter__(self) -> torch.Tensor:
        return iter([self.output])

    def __len__(self) -> int:
        return 1

def output_from_npy(npy_path: Path) -> torch.Tensor:
    return torch.tensor(np.load(REPO_DATA_ROOT/npy_path).astype(np.float32))

@define(hash=False, eq=False)
class InversionResultLitModel(LightningModule):

    output: torch.Tensor
    lit_dmb_model: LitDMBModel
    optimizer: functools.partial[Optimizer]
    lr_scheduler: dict[str, functools.partial[_LRScheduler | Any]]
    loss: Loss
    metrics: torchmetrics.MetricCollection
    inversion_result: InversionResult

    dmb_model: DMBModel = field(init=False)


    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        # freeze output
        self.output.requires_grad = False

        self.dmb_model = self.lit_dmb_model.model

        # freeze the dmb model
        for param in self.dmb_model.parameters():
            param.requires_grad = False

        # set dmb model to eval
        self.dmb_model.eval()

    def forward(self) -> torch.Tensor:
        dmb_model_out: torch.Tensor = self.dmb_model(self.inversion_result().unsqueeze(0))
        return dmb_model_out

    def _calculate_loss(self) -> tuple[torch.Tensor, LossOutput]:
        model_out = self()
        density_feature_dim = self.dmb_model.observables.index("density")
        model_out = model_out[...,density_feature_dim,:,:]

        batch = MultipleSizesBatch(
            inputs=[],
            outputs=[self.output.unsqueeze(0).expand_as(model_out)],
            sample_ids=[],
            group_elements=[],
            size=1).to(model_out.device)

        return model_out, self.loss(model_out, batch)

    def _evaluate_metrics(self, model_out: torch.Tensor) -> None:
        self.metrics.update(preds=model_out, target=self.output.to(model_out.device))

    def training_step(self, *args: Any, **kwargs: dict) -> torch.Tensor:
        model_out, loss_out = self._calculate_loss()
        self._evaluate_metrics(model_out)

        # log metrics
        self.log_metrics(
            stage="train",
            metric_collection=dict(
                itertools.chain(self.metrics.items(), {
                    "loss": loss_out.loss,
                    **loss_out.loggables
                }.items())),
            on_step=True,
            on_epoch=True,
        )

        return loss_out.loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and scheduler."""
        optimizer: torch.optim.Optimizer = self.optimizer(
            params=self.inversion_result.parameters())

        configuration: dict[str, Any] = {"optimizer": optimizer}

        if self.lr_scheduler is not None:
            scheduler: _LRScheduler = self.lr_scheduler["scheduler"](
                optimizer=optimizer)
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
    ) -> None:
        """Log metrics."""
        for metric_name, metric in metric_collection.items():

            computed_metric = (metric.compute() if isinstance(
                metric, torchmetrics.Metric) else metric)

            if isinstance(computed_metric, dict):
                loggable = {
                    f"{stage}/{metric_name}/{key}": value
                    for key, value in computed_metric.items()
                }
            else:
                loggable = {f"{stage}/{metric_name}": computed_metric}

            self.log_dict(loggable, on_step=on_step, on_epoch=on_epoch, batch_size=1)

    def on_train_epoch_end(self) -> None:
        """Execute at the end of each training epoch."""
        self.metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Execute at the end of each validation epoch."""
        self.metrics.reset()
