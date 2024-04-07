import functools
import itertools
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, cast

import hydra
import lightning
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torchmetrics
import transformers
from attrs import define, field
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer

from dmb.data.bose_hubbard_2d.phase_diagram import plot_phase_diagram, \
    plot_phase_diagram_mu_cut
from dmb.data.bose_hubbard_2d.plots import create_box_cuts_plot, \
    create_box_plot, create_wedding_cake_plot
from dmb.logging import create_logger
from dmb.model.mixins import LitModelMixin

log = create_logger(__name__)


@define
class DMBLitModel(pl.LightningModule, LitModelMixin):

    model: torch.nn.Module
    optimizer: functools.partial[Optimizer]
    scheduler: functools.partial[dict[str,
                                      functools.partial[_LRScheduler | Any]]]
    loss: torch.nn.Module

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
            1, self.hparams["model"]["in_channels"], 10, 10)

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
        for metric_name, (metric,
                          lit_module_attribute) in metrics_dict.items():
            metric.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and scheduler."""
        optimizer: torch.optim.Optimizer = self.optimizer(params=filter(
            lambda p: p.requires_grad, self.trocr_model.parameters()), )

        configuration: dict[str, Any] = {"optimizer": optimizer}

        if self.lr_scheduler is not None:
            scheduler: _LRScheduler = self.lr_scheduler["scheduler"](
                optimizer=optimizer)
            configuration["lr_scheduler"] = {
                **self.lr_scheduler,
                "scheduler": scheduler,
            }

        return cast(OptimizerLRScheduler, configuration)

    def plot_model(
            self,
            save_dir: Path,
            file_name_stem: str,
            resolution: int = 300,
            check: List[Tuple[str, ...]] = [
                ("density", "max-min"),
                ("density_variance", "mean"),
                ("mu_cut", ),
                ("wedding_cake", "2.67"),
                ("wedding_cake", "1.33"),
                ("wedding_cake", "2.0"),
                ("box", "1.71"),
                ("box_cuts", ),
            ],
            zVUs: List[float] = (1.0, 1.5),
            ztUs: List[float] = (0.1, 0.25),
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
                            yield from recursive_iter(path + (key, ), value)
                    elif isinstance(obj, list):
                        for idx, value in enumerate(obj):
                            yield from recursive_iter(path + (idx, ), value)
                    else:
                        yield path, obj

                # recursively visit all figures
                for path, figure in recursive_iter((), figures):

                    # * is a wildcard
                    if not any(
                            all(a == b or a == "*" or b == "*"
                                for a, b in zip(check_, path))
                            and len(check_) == len(path) for check_ in check):
                        continue

                    if isinstance(figure, plt.Figure):
                        save_path = Path(save_dir) / (
                            file_name_stem + "_" + str(zVU).replace(".", "_") +
                            "_" + str(ztU).replace(".", "_") + "_" +
                            "_".join(path) + ".png")
                        save_path.parent.mkdir(exist_ok=True, parents=True)
                        figure.savefig(save_path)
