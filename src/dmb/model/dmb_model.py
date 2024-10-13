"""DMBModel class and utility functions for model prediction."""

import itertools
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader


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
        self.output_modification = torch.nn.ModuleList(
            modules=output_modification)

        self.observables = observables

    def forward_single_size(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single input size."""
        for module in itertools.chain(self.modules_list,
                                      self.output_modification):
            x = module(x)

        return x

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor]:
        if isinstance(x, (tuple, list)):
            out = tuple(self.forward_single_size(_x) for _x in x)
        else:
            out = self.forward_single_size(x)

        return out

    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device


def dmb_model_predict(
    model: DMBModel,
    inputs: torch.Tensor | list[torch.Tensor],
    batch_size: int = 512,
) -> dict[str, torch.Tensor]:
    """Predict with DMB model."""
    dl = DataLoader(inputs,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0)

    model.eval()
    with torch.no_grad():
        outputs = []
        for inputs in dl:
            inputs = inputs.to(model.device).float()
            outputs.append(model(inputs))

        outputs = torch.cat(outputs, dim=0).to("cpu").detach()

    return {
        obs: outputs[:, idx].numpy()
        for idx, obs in enumerate(model.observables)
    }
