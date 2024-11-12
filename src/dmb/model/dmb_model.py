"""DMBModel class and utility functions for model prediction."""

import itertools
from typing import Iterable, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DMBModel(nn.Module):
    """DMB model class."""

    def __init__(
            self,
            observables: list[str],
            module_list: Iterable[nn.Module],
            output_modification: Iterable[nn.Module] = (),
    ) -> None:
        """Initialize DMB model.

        Args:
            observables: List of observables.
            module_list: List of modules.
            output_modification: List of output modification modules.
        """
        super().__init__()

        self.modules_list = torch.nn.ModuleList(modules=module_list)
        self.output_modification = torch.nn.ModuleList(modules=output_modification)

        self.observables = observables

    def forward_single_size(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single input size."""
        for module in itertools.chain(self.modules_list, self.output_modification):
            x = module(x)

        return x

    def forward(
            self,
            x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Compute forward pass."""
        if isinstance(x, (tuple, list)):
            out: list[torch.Tensor] | torch.Tensor = [
                self.forward_single_size(_x) for _x in x
            ]
        else:
            out = self.forward_single_size(x)

        return out

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device


class PredictionMapping:
    """Prediction mapping class."""

    def __init__(self, model: DMBModel, batch_size: int = 512) -> None:
        """Initialize prediction mapping.

        Args:
            model: DMB model.
        """
        self.model = model
        self.batch_size = batch_size

    def __call__(
            self, inputs: list[torch.Tensor] | Dataset | torch.Tensor
    ) -> dict[str, np.ndarray]:
        """Predict with DMB model."""

        dataloader = DataLoader(
            cast(Dataset, inputs),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()
        with torch.no_grad():
            outputs = []
            for batch in dataloader:
                batch = batch.to(self.model.device).float()
                outputs.append(self.model(batch))

            outputs_tensor = torch.cat(outputs, dim=0).to("cpu").detach()

        return {
            obs: outputs_tensor[:, idx].numpy()
            for idx, obs in enumerate(self.model.observables)
        }
