import torch
import lightning.pytorch as pl
from torch import nn


class Ellipsoid(pl.LightningModule):
    def __init__(
        self,
        center: torch.Tensor,
        params: torch.Tensor,
        min_value: float,
        max_value: float,
        separation_exponent: int = 1,
    ):
        super().__init__()

        self.register_parameter("center", nn.Parameter(center))
        self.register_parameter("params", nn.Parameter(params))
        self.min_value = min_value
        self.max_value = max_value
        self.separation_exponent = separation_exponent

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(((x - self.center) / self.params) ** 2, dim=-1) < 1

    def reparametrized_distance_to_center(self, x: torch.Tensor) -> float:
        return torch.sqrt((((x - self.center) / self.params) ** 2).sum(dim=-1))


class Gradient(pl.LightningModule):
    def __init__(
        self,
        anchor: torch.Tensor,
        anchor_value: float,
        direction: torch.Tensor,
        direction_value: float,
    ) -> None:
        super().__init__()

        self.register_parameter("anchor", nn.Parameter(anchor))
        self.register_parameter("direction", nn.Parameter(direction))
        self.anchor_value = anchor_value
        self.direction_value = direction_value

    def get_value_at_x(self, x: torch.tensor) -> float:
        # print datatypes of tensors
        return abs(
            self.anchor_value
            + self.direction_value * ((x - self.anchor) @ self.direction)
        )


BOSE_HUBBARD_FAKE_ELLIPSOIDS = [
    Ellipsoid(
        center=torch.tensor((0.5, 0, 1), dtype=torch.float32),
        params=torch.tensor((0.5, 0.35, 0.2), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=1.0,
    ),
    Ellipsoid(
        center=torch.tensor((1.5, 0, 1), dtype=torch.float32),
        params=torch.tensor((0.5, 0.2, 0.15), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=1.0,
        max_value=1.0,
    ),
    Ellipsoid(
        center=torch.tensor((2.5, 0, 1), dtype=torch.float32),
        params=torch.tensor((0.5, 0.2, 0.25), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=1.0,
        max_value=2.0,
    ),
    Ellipsoid(
        center=torch.tensor((2.5, 0, 1.5), dtype=torch.float32),
        params=torch.tensor((0.5, 0.7, 0.25), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=3.0,
    ),
    Ellipsoid(
        center=torch.tensor((1.5, 0, 1.5), dtype=torch.float32),
        params=torch.tensor((0.5, 0.5, 0.15), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=2.0,
    ),
    Ellipsoid(
        center=torch.tensor((0.5, 0, 1.5), dtype=torch.float32),
        params=torch.tensor((0.5, 0.3, 0.15), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=1.0,
    ),
]

BOSE_HUBBARD_FAKE_GRADIENTS = [
    Gradient(
        anchor=torch.tensor((0, 0, 0), dtype=torch.float32),
        anchor_value=0.0,
        direction=torch.tensor((1, 0, 0), dtype=torch.float32),
        direction_value=1,
    )
]
