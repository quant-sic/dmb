import functools
import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torchmetrics
from attrs import define
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from dmb.data.bose_hubbard_2d.phase_diagram import (
    plot_phase_diagram,
    plot_phase_diagram_mu_cut,
)
from dmb.data.bose_hubbard_2d.plots import (
    create_box_cuts_plot,
    create_box_plot,
    create_wedding_cake_plot,
)
from dmb.logging import create_logger

log = create_logger(__name__)


@define(hash=False, eq=False)
class LitDMBModel(pl.LightningModule):

    model: torch.nn.Module
    optimizer: functools.partial[Optimizer]
    lr_scheduler: functools.partial[dict[str, functools.partial[_LRScheduler | Any]]]
    loss: torch.nn.Module

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.example_input_array = torch.zeros(1, self.model.in_channels, 10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _batch_evaluation(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch_in, batch_label = batch

        model_out = self(batch_in)
        loss = self.loss(model_out, batch_label)

        return model_out, loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        model_out, loss = self._batch_evaluation(batch)

        # log metrics
        batch_size = sum(len(b) for b in batch)
        self.log_metrics(
            stage="train",
            metric_collection={"loss": loss},
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        model_out, loss = self._batch_evaluation(batch)
        # log metrics
        batch_size = sum(len(b) for b in batch)
        self.log_metrics(
            stage="val",
            metric_collection={"loss": loss},
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        model_out, loss = self._batch_evaluation(batch)
        # log metrics
        batch_size = sum(len(b) for b in batch)
        self.log_metrics(
            stage="test",
            metric_collection={"loss": loss},
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

    def plot_model(
        self,
        save_dir: Path,
        file_name_stem: str,
        resolution: int = 300,
        check: list[tuple[str, ...]] = [
            ("density", "max-min"),
            ("density_variance", "mean"),
            ("mu_cut",),
            ("wedding_cake", "2.67"),
            ("wedding_cake", "1.33"),
            ("wedding_cake", "2.0"),
            ("box", "1.71"),
            ("box_cuts",),
        ],
        zVUs: list[float] = (1.0, 1.5),
        ztUs: list[float] = (0.1, 0.25),
    ) -> None:
        # mu,ztU,out = model_predict(net,batch_size=512)
        for zVU, ztU in itertools.product(zVUs, ztUs):
            for figures in (
                create_box_cuts_plot(self, zVU=zVU, ztU=ztU),
                create_box_plot(self, zVU=zVU, ztU=ztU),
                create_wedding_cake_plot(self, zVU=zVU, ztU=ztU),
                plot_phase_diagram(self, n_samples=resolution, zVU=zVU),
                plot_phase_diagram_mu_cut(self, zVU=zVU, ztU=ztU),
                plot_phase_diagram_mu_cut(self, zVU=zVU, ztU=ztU),
            ):

                def recursive_iter(path, obj):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            yield from recursive_iter(path + (key,), value)
                    elif isinstance(obj, list):
                        for idx, value in enumerate(obj):
                            yield from recursive_iter(path + (idx,), value)
                    else:
                        yield path, obj

                # recursively visit all figures
                for path, figure in recursive_iter((), figures):

                    # * is a wildcard
                    if not any(
                        all(
                            a == b or a == "*" or b == "*" for a, b in zip(check_, path)
                        )
                        and len(check_) == len(path)
                        for check_ in check
                    ):
                        continue

                    if isinstance(figure, plt.Figure):
                        save_path = Path(save_dir) / (
                            file_name_stem
                            + "_"
                            + str(zVU).replace(".", "_")
                            + "_"
                            + str(ztU).replace(".", "_")
                            + "_"
                            + "_".join(path)
                            + ".png"
                        )
                        save_path.parent.mkdir(exist_ok=True, parents=True)
                        figure.savefig(save_path)
