"""DMBModel class and utility functions for model prediction."""

import itertools
from typing import Iterable, cast

import numpy as np
import torch
from attrs import define, field
from torch import nn
from torch.utils.data import DataLoader, Dataset


@define
class DMBModelOutput:
    """DMB model output class."""

    loss_input: list[torch.Tensor] = field()
    inference_output: list[torch.Tensor] = field()


@define
class OutputModifications:
    """Output modifications for DMB model."""

    loss_input: nn.Module = field(default=nn.Identity())
    inference_output: nn.Module = field(default=nn.Identity())


class DMBModel(nn.Module):
    """DMB model class."""

    def __init__(
        self,
        observables: list[str],
        module_list: Iterable[nn.Module],
        modifications: OutputModifications | None = None,
    ) -> None:
        """Initialize DMB model.

        Args:
            observables: List of observables.
            module_list: List of modules.
            output_modification: List of output modification modules.
        """
        super().__init__()

        self.modules_list = torch.nn.ModuleList(modules=module_list)

        self.modifications = (
            modifications if modifications is not None else OutputModifications()
        )

        self.observables = observables

    def forward_single_size(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single input size."""
        for module in itertools.chain(self.modules_list):
            x = module(x)

        loss_input = self.modifications.loss_input(x)
        inference_output = self.modifications.inference_output(x)

        return loss_input, inference_output

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> DMBModelOutput:
        """Forward pass of the DMB model."""
        if not isinstance(x, (tuple, list)):
            x = [x]

        out: DMBModelOutput = DMBModelOutput(loss_input=[], inference_output=[])
        for _x in x:
            loss_input, inference_output = self.forward_single_size(_x)
            out.loss_input.append(loss_input)
            out.inference_output.append(inference_output)

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
                outputs.append(self.model(batch).inference_output[0])

            outputs_tensor = torch.cat(outputs, dim=0).to("cpu").detach()

        return {
            obs: outputs_tensor[:, idx].numpy()
            for idx, obs in enumerate(self.model.observables)
        }
