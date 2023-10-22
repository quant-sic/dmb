from torch.utils.data import Dataset
from dmb.data.bose_hubbard_2d.network_input import (
    net_input,
    dimless_from_net_input,
)
from dmb.data.bose_hubbard_2d.potential import get_random_trapping_potential
import torch

from typing import Any, Tuple, List
from torch import nn
import numpy as np
from torch import nn
import lightning


class Ellipsoid(lightning.LightningModule):
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


class Gradient(lightning.LightningModule):
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


class PhaseDiagram3d(lightning.LightningModule):
    def __init__(
        self,
        ellipsoids: List[Ellipsoid],
        gradients: List[Gradient],
        muU_range: Tuple[float, float] = (-0.5, 3.0),
        ztU_range: Tuple[float, float] = (0.05, 1.0),
        zVU_range: Tuple[float, float] = (0.75, 1.75),
    ):
        super().__init__()

        self.register_module("ellipsoids", nn.ModuleList(ellipsoids))
        self.register_module("gradients", nn.ModuleList(gradients))

        self.register_parameter(
            "ellipsoid_min_values",
            nn.Parameter(torch.tensor([e.min_value for e in ellipsoids])),
        )
        self.register_parameter(
            "ellipsoid_max_values",
            nn.Parameter(torch.tensor([e.max_value for e in ellipsoids])),
        )
        self.register_parameter(
            "ellipsoid_separation_exponents",
            nn.Parameter(torch.tensor([e.separation_exponent for e in ellipsoids])),
        )

        self.size = [muU_range, ztU_range, zVU_range]

    def volume(self):
        return np.prod([max - min for min, max in self.size])

    def __contains__(self, x: torch.Tensor) -> bool:
        return all(
            [x[i] > self.size[i][0] and x[i] < self.size[i][1] for i in range(3)]
        )

    def get_value_at_x(self, x: torch.Tensor) -> Tuple[float, float]:
        base_value = sum([g.get_value_at_x(x) for g in self.gradients])

        ellipsoids_contain = torch.stack(
            [e.contains(x) for e in self.ellipsoids], dim=-1
        )

        distances = torch.stack(
            [e.reparametrized_distance_to_center(x) for e in self.ellipsoids], dim=-1
        )

        total_min_value = (
            self.ellipsoid_min_values - base_value[..., None]
        ) * torch.exp(-(distances**self.ellipsoid_separation_exponents)) + base_value[
            ..., None
        ]
        total_max_value = (
            self.ellipsoid_max_values - base_value[..., None]
        ) * torch.exp(-(distances**self.ellipsoid_separation_exponents)) + base_value[
            ..., None
        ]

        # trick
        total_min_value = torch.where(
            ~ellipsoids_contain,
            total_min_value.max(dim=-1)
            .values[..., None]
            .expand(*total_min_value.shape),
            total_min_value,
        )
        total_max_value = torch.where(
            ~ellipsoids_contain,
            total_max_value.min(dim=-1)
            .values[..., None]
            .expand(*total_max_value.shape),
            total_max_value,
        )

        # get min and max over last dimension. If no ellipsoid contains x, then the base value is used
        no_ellipsoid_contains = (~ellipsoids_contain).all(dim=-1)

        total_min_value = torch.where(
            no_ellipsoid_contains,
            base_value,
            total_min_value.min(dim=-1).values,
        )
        total_max_value = torch.where(
            no_ellipsoid_contains,
            base_value,
            total_max_value.max(dim=-1).values,
        )

        return total_min_value, total_max_value


class LocalDensityApproximationModel(lightning.LightningModule):
    def __init__(self, phase_diagram: PhaseDiagram3d):
        super().__init__()
        self.register_module("phase_diagram", phase_diagram)

        self.observables = ["Density_Distribution"]

    def __call__(self, x: torch.Tensor, *args: Any, **kwds: Any) -> Any:
        """
        Expects as input a tensor of shape (N,4,L,L).

        Args:
            x (torch.Tensor): Input tensor of shape (N,C,L,L). Channels are mu, U, V, cb
        Returns:
            torch.Tensor: Output tensor of shape (N,1,L,L)
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be of type torch.Tensor")

        if x.shape[1] != 4:
            raise ValueError("Input must have 4 channels")

        if x.shape[2] != x.shape[3]:
            raise ValueError("Input must be square")

        # get muU, ztU, zVU, cb
        muU, cb, ztU, zVU = dimless_from_net_input(x)

        # get base_value, ellipsoid_value
        (
            min_value,
            max_value,
        ) = self.phase_diagram.get_value_at_x(
            torch.concat([muU[..., None], ztU[..., None], zVU[..., None]], dim=-1)
        )

        return min_value * cb + max_value * (1 - cb)


class RandomLDAMSampler:
    def __init__(
        self,
        ellipsoids: List[Ellipsoid],
        gradients: List[Gradient],
        muU_range: Tuple[float, float] = (-0.5, 3.0),
        ztU_range: Tuple[float, float] = (0.05, 1.0),
        zVU_range: Tuple[float, float] = (0.75, 1.75),
        L_range: Tuple[int, int] = (8, 20),
        z=4,
    ):
        self.phase_diagram = PhaseDiagram3d(
            gradients=gradients,
            ellipsoids=ellipsoids,
            muU_range=muU_range,
            ztU_range=ztU_range,
            zVU_range=zVU_range,
        )
        self.ldam = LocalDensityApproximationModel(self.phase_diagram)
        self.z = z
        self.muU_range = muU_range
        self.ztU_range = ztU_range
        self.zVU_range = zVU_range
        self.L_range = L_range

    def sample(self):
        L = np.random.randint(low=self.L_range[0] / 2, high=self.L_range[1] / 2) * 2

        muU = np.random.uniform(low=self.muU_range[0], high=self.muU_range[1])
        ztU = np.random.uniform(low=self.ztU_range[0], high=self.ztU_range[1])
        zVU = np.random.uniform(low=self.zVU_range[0], high=self.zVU_range[1])

        U_on = self.z / ztU
        V_nn = zVU * U_on / self.z
        mu_offset = muU * U_on

        _, V_trap = get_random_trapping_potential(
            shape=(L, L), desired_abs_max=abs(mu_offset) / 2
        )

        mu = torch.from_numpy(mu_offset + V_trap).float()
        inputs = net_input(mu=mu, U_on=U_on, V_nn=V_nn, cb_projection=True).unsqueeze(0)
        label = self.ldam(inputs)

        return inputs, label


class SimulatedDataset(Dataset):
    def __init__(self, num_samples, transforms=None):
        self.num_samples = num_samples
        self.transforms = transforms
