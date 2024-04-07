from typing import Any, List, Tuple

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.network_input import dimless_from_net_input, \
    net_input
from dmb.data.bose_hubbard_2d.potential import get_random_trapping_potential
from dmb.data.bose_hubbard_2d.simulated.fake_phase_diagram_objects import \
    BOSE_HUBBARD_FAKE_ELLIPSOIDS, BOSE_HUBBARD_FAKE_GRADIENTS, Ellipsoid, \
    Gradient


class PhaseDiagram3d(pl.LightningModule):

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
            nn.Parameter(
                torch.tensor([e.separation_exponent for e in ellipsoids])),
        )

        self.size = [muU_range, ztU_range, zVU_range]

    @property
    def volume(self):
        return np.prod([max - min for min, max in self.size])

    def __contains__(self, x: torch.Tensor) -> bool:
        return all([
            x[i] > self.size[i][0] and x[i] < self.size[i][1] for i in range(3)
        ])

    def get_value_at_x(self, x: torch.Tensor) -> Tuple[float, float]:
        base_value = sum([g.get_value_at_x(x) for g in self.gradients])

        ellipsoids_contain = torch.stack(
            [e.contains(x) for e in self.ellipsoids], dim=-1)

        distances = torch.stack(
            [e.reparametrized_distance_to_center(x) for e in self.ellipsoids],
            dim=-1)

        total_min_value = (
            self.ellipsoid_min_values - base_value[..., None]) * torch.exp(-(
                distances**self.ellipsoid_separation_exponents)) + base_value[
                    ..., None]
        total_max_value = (
            self.ellipsoid_max_values - base_value[..., None]) * torch.exp(-(
                distances**self.ellipsoid_separation_exponents)) + base_value[
                    ..., None]

        # trick
        total_min_value = torch.where(
            ~ellipsoids_contain,
            total_min_value.max(dim=-1).values[..., None].expand(
                *total_min_value.shape),
            total_min_value,
        )
        total_max_value = torch.where(
            ~ellipsoids_contain,
            total_max_value.min(dim=-1).values[..., None].expand(
                *total_max_value.shape),
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


class LocalDensityApproximationModel(pl.LightningModule):

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
            torch.concat([muU[..., None], ztU[..., None], zVU[..., None]],
                         dim=-1))

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

    @torch.no_grad()
    def sample(self):
        L = np.random.randint(low=self.L_range[0] / 2,
                              high=self.L_range[1] / 2) * 2

        muU = np.random.uniform(low=self.muU_range[0], high=self.muU_range[1])
        ztU = np.random.uniform(low=self.ztU_range[0], high=self.ztU_range[1])
        zVU = np.random.uniform(low=self.zVU_range[0], high=self.zVU_range[1])

        U_on = self.z / ztU
        V_nn = zVU * U_on / self.z
        mu_offset = muU * U_on

        _, V_trap = get_random_trapping_potential(
            shape=(L, L), desired_abs_max=abs(mu_offset) / 2)

        mu = torch.from_numpy(mu_offset + V_trap).float()
        inputs = net_input(mu=mu, U_on=U_on, V_nn=V_nn,
                           cb_projection=True).unsqueeze(0)
        label = self.ldam(inputs)

        return {
            "mu_offset": mu_offset,
            "U_on": U_on,
            "V_nn": V_nn,
            "L": L,
            "ztU": ztU,
            "zVU": zVU,
            "muU": muU,
            "inputs": inputs.squeeze(0),
            "label": label.squeeze(0),
        }


class SimulatedBoseHubbard2dDataset(Dataset):

    def __init__(
        self,
        num_samples,
        muU_range: Tuple[float, float] = (-0.5, 3.0),
        ztU_range: Tuple[float, float] = (0.05, 1.0),
        zVU_range: Tuple[float, float] = (0.75, 1.75),
        L_range: Tuple[int, int] = (8, 20),
        z=4,
        base_transforms=None,
        train_transforms=None,
    ):
        super().__init__()

        self.num_samples = num_samples

        self.muU_range = muU_range
        self.ztU_range = ztU_range
        self.zVU_range = zVU_range
        self.L_range = L_range
        self.z = z

        self.sampler = RandomLDAMSampler(
            ellipsoids=BOSE_HUBBARD_FAKE_ELLIPSOIDS,
            gradients=BOSE_HUBBARD_FAKE_GRADIENTS,
            muU_range=muU_range,
            ztU_range=ztU_range,
            zVU_range=zVU_range,
            L_range=L_range,
            z=z,
        )

        # transforms
        self.base_transforms = base_transforms
        self.train_transforms = train_transforms

        self.samples = [
            self.sampler.sample() for _ in tqdm(range(num_samples), "Sampling")
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs, outputs = self.samples[idx]["inputs"], self.samples[idx][
            "label"]

        # apply transforms
        if self.base_transforms is not None:
            inputs, outputs = self.base_transforms((inputs, outputs))

        if self.train_transforms is not None and self.apply_train_transforms:
            inputs, outputs = self.train_transforms((inputs, outputs))

        return inputs, outputs

    @property
    def apply_train_transforms(self) -> bool:
        if hasattr(self, "_apply_train_transforms"):
            return self._apply_train_transforms
        else:
            raise AttributeError(
                "apply_transforms is not set. Please set it first.")

    @apply_train_transforms.setter
    def apply_train_transforms(self, value: bool):
        self._apply_train_transforms = value

    @property
    def sample_density(self) -> float:
        return self.num_samples / self.sampler.phase_diagram.volume

    def plot_samples(self):
        figures = []
        linspace = np.linspace(self.zVU_range[0], self.zVU_range[1], 10)

        muU = [sample["muU"] for sample in self.samples]
        ztU = [sample["ztU"] for sample in self.samples]
        zVU = [sample["zVU"] for sample in self.samples]

        for start, end in zip(linspace[:-1], linspace[1:]):
            fig, ax = plt.subplots()

            ax.set_title(f"zVU in [{start:.2f}, {end:.2f}]")
            ax.set_xlabel("muU")
            ax.set_ylabel("ztU")

            ax.set_ylim(self.muU_range)
            ax.set_xlim(self.ztU_range)

            zVU_in_range = [(zVU_i > start and zVU_i < end) for zVU_i in zVU]

            ax.scatter(
                np.array(ztU)[zVU_in_range],
                np.array(muU)[zVU_in_range],
            )

            figures.append(fig)

        return figures

    def get_dataset_ids_from_indices(self, idx: int):
        return idx

    @property
    def observables(self):
        return self.sampler.ldam.observables
