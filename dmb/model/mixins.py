import pytorch_lightning as pl
import torch
from typing import Any, Dict, Optional, Literal, cast,List
import hydra
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torchmetrics import MetricCollection

from dmb.utils import create_logger
from dmb.model.utils import MaskedMSE,MaskedMSELoss
from functools import cached_property

log = create_logger(__name__)

class LitModelMixin(pl.LightningModule):

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

        log.info(
            "Using {} as loss".format(
                _loss.__class__.__name__
            )
        )

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

        if isinstance(model_out, (list,tuple)):
            batch_size = sum([x.shape[0] for x in model_out])
        else:
            batch_size = model_out.shape[0]

        # log loss
        if loss is not None:
            self.log_dict(
                {f"{stage}/loss": loss},
                on_epoch=True,
                on_step=(stage == "train"),
                batch_size=batch_size,
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
            log_on_step = (stage == "train")

            self.log_dict(
                log_dict,
                on_epoch=True,
                on_step=log_on_step,
                batch_size=batch_size,
            )