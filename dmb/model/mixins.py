from functools import cached_property
from typing import Literal, Optional

import torch
from torchmetrics import MetricCollection

from dmb.model.utils import MaskedMSE
from dmb.utils import create_logger

log = create_logger(__name__)


class LitModelMixin:

    @property
    def metrics(self) -> MetricCollection:
        metric_collection = MetricCollection({
            "mse": MaskedMSE(),
        })

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

        if isinstance(model_out, (list, tuple)):
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
            log_dict = ({
                metric_name: computed_metric
            } if not isinstance(computed_metric, dict) else {
                f"{metric_name}/{key}": value
                for key, value in computed_metric.items()
            })

            # log on step only for training
            log_on_step = stage == "train"

            self.log_dict(
                log_dict,
                on_epoch=True,
                on_step=log_on_step,
                batch_size=batch_size,
            )
