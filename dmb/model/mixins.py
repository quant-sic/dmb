from functools import cached_property
from typing import Literal, Optional

import torch
from torchmetrics import MetricCollection

from dmb.logging import create_logger
from dmb.model.utils import MaskedMSE

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
