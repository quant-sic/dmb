"""Module for the worm input parameters."""

from __future__ import annotations

__all__ = ["WormInputParameters"]

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from attrs import frozen
import json


@frozen(eq=False, slots=False)
class WormInputParameters:
    """Class for the input parameters of the worm simulation."""

    mu: np.ndarray

    Lx: int
    Ly: int

    t_hop: np.ndarray
    U_on: np.ndarray
    V_nn: np.ndarray

    Lz: int = 1

    model: str = "BoseHubbard"
    runtimelimit: int = 24 * 60 * 60
    sweeps: int = 25000
    thermalization: int = 100

    pbcx: int = 1
    pbcy: int = 1
    pbcz: int = 1
    beta: float = 20.0
    nmax: int = 3
    E_off: float = 1.0
    canonical: int = -1
    seed: int = 30
    Ntest: int = 10000000
    Nsave: int = 100000000
    Nmeasure: int = 1
    Nmeasure2: int = 10
    C_worm: float = 2.0
    p_insertworm: float = 1.0
    p_moveworm: float = 0.3
    p_insertkink: float = 0.2
    p_deletekink: float = 0.2
    p_glueworm: float = 0.3

    mu_power: float | None = None
    mu_offset: float | None = None

    def __eq__(self, other: object) -> bool:
        """Check if two input parameters instances are equal."""
        if other is self:
            return True

        if type(other) is not type(self):
            return False

        for attribute in self.__attrs_attrs__:
            attribute_value = self.__getattribute__(attribute.name)
            if isinstance(attribute_value, np.ndarray):
                if not np.array_equal(
                        self.__getattribute__(attribute.name),
                        other.__getattribute__(attribute.name),
                ):
                    return False

            elif self.__getattribute__(attribute.name) != other.__getattribute__(
                    attribute.name):
                return False

        return True

    @staticmethod
    def get_ini_path(save_dir_path: Path) -> Path:
        return save_dir_path / "parameters.ini"

    @staticmethod
    def get_h5_path(save_dir_path: Path) -> Path:
        return save_dir_path / "parameters.h5"

    @staticmethod
    def get_outputfile_path(save_dir_path: Path) -> Path:
        return save_dir_path / "output.h5"

    @staticmethod
    def get_checkpoint_path(save_dir_path: Path) -> Path:
        return save_dir_path / "checkpoint.h5"

    @classmethod
    def from_dir(cls, save_dir_path: Path) -> WormInputParameters:
        attributes: dict[str, type | None] = {
            attribute.name: eval(attribute.type)
            for attribute in cls.__attrs_attrs__
        }

        # Read ini file
        with open(save_dir_path / "parameters.ini", "r") as f:
            lines = f.readlines()

        # Fill dictionary for ini parameters
        params = {}
        for line in lines:
            if not line.startswith("#"):
                value: Any
                key, value = map(lambda s: s.strip(), line.split("="))

                if key in attributes:
                    if key in ("mu_power", "mu_offset"):
                        value = float(value) if value != "None" else None

                    elif key not in ("mu", "t_hop", "U_on", "V_nn"):
                        if attributes[key] is None:
                            raise ValueError(
                                f"Inconsistent attribute {key} in {cls.__name__}")
                        else:
                            value = attributes[key](value)  # type: ignore

                    # add to dictionary
                    params[key] = value

        # read in h5 site dependent arrays
        with h5py.File(save_dir_path / "parameters.h5", "r") as file:
            for name in ("mu", "t_hop", "U_on", "V_nn"):
                params[name] = file[f"/{name}"][()]

                if name in ("t_hop", "V_nn"):
                    params[name] = params[name].reshape(2, params["Lx"], params["Ly"])
                else:
                    params[name] = params[name].reshape(params["Lx"], params["Ly"])

        # Create input parameters
        return cls(**params)

    def save_h5_file(self, save_dir: Path) -> None:
        h5_file_path = self.get_h5_path(save_dir)

        # create parent directory if it does not exist
        h5_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create h5 file
        with h5py.File(h5_file_path, "w") as file:
            for name, attribute in (
                ("mu", self.mu),
                ("t_hop", self.t_hop),
                ("U_on", self.U_on),
                ("V_nn", self.V_nn),
            ):
                file[f"/{name}"] = attribute.flatten()

    def save_ini_file(self, save_dir: Path) -> None:
        ini_file_path = self.get_ini_path(save_dir)
        h5_file_path = self.get_h5_path(save_dir)

        # create parent directory if it does not exist
        ini_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create ini file
        with open(ini_file_path, "w") as f:
            f.write(f"site_arrays = {h5_file_path}\n")
            f.write(f"outputfile = {self.get_outputfile_path(save_dir)}\n")
            f.write(f"checkpoint = {self.get_checkpoint_path(save_dir)}\n")

            for attribute in self.__attrs_attrs__:
                if not (attribute.name in ("mu", "t_hop", "U_on", "V_nn")
                        and eval(attribute.type) is np.ndarray):
                    f.write(
                        f"{attribute.name} = {self.__getattribute__(attribute.name)}\n")

    def save(self, save_dir: Path) -> None:
        # Create ini file
        self.save_ini_file(save_dir=save_dir)
        self.save_h5_file(save_dir=save_dir)

    def plot_input_parameters(self, plots_dir: Path) -> None:
        """Plot the input parameters of the worm simulation."""
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(self.mu.reshape(self.Lx, self.Ly))
        ax[0].set_title("mu")
        ax[1].imshow(self.t_hop.reshape(2, self.Lx, self.Ly)[0])
        ax[1].set_title("t_hop")
        ax[2].imshow(self.U_on.reshape(self.Lx, self.Ly))
        ax[2].set_title("U_on")
        ax[3].imshow(self.V_nn.reshape(2, self.Lx, self.Ly)[0])
        ax[3].set_title("V_nn")

        # set axes off
        for ax_ in ax:
            ax_.axis("off")

        # set colorbars
        fig.colorbar(
            ax[0].imshow(self.mu.reshape(self.Lx, self.Ly)),
            ax=ax[0],
        )
        fig.colorbar(
            ax[1].imshow(self.t_hop.reshape(2, self.Lx, self.Ly)[0]),
            ax=ax[1],
        )
        fig.colorbar(
            ax[2].imshow(self.U_on.reshape(self.Lx, self.Ly)),
            ax=ax[2],
        )
        fig.colorbar(
            ax[3].imshow(self.V_nn.reshape(2, self.Lx, self.Ly)[0]),
            ax=ax[3],
        )

        plt.savefig(plots_dir / "inputs.png")
        plt.close()

    def plot_phase_diagram_input_parameters(self, plots_dir: Path) -> None:
        """Plot the phase diagram input parameters of the worm simulation."""
        muU = self.mu / self.U_on
        ztU = 4 * self.t_hop / self.U_on[None, ...]
        zVU = 4 * self.V_nn / self.U_on[None, ...]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(muU.reshape(self.Lx, self.Ly))
        ax[0].set_title("muU")

        ax[1].imshow(ztU.reshape(2, self.Lx, self.Ly)[0])
        ax[1].set_title("ztU")

        ax[2].imshow(zVU.reshape(2, self.Lx, self.Ly)[0])
        ax[2].set_title("zVU")

        # set axes off
        for ax_ in ax:
            ax_.axis("off")

        # set colorbars
        fig.colorbar(ax[0].imshow(muU.reshape(self.Lx, self.Ly)), ax=ax[0])
        fig.colorbar(ax[1].imshow(ztU.reshape(2, self.Lx, self.Ly)[0]), ax=ax[1])
        fig.colorbar(ax[2].imshow(zVU.reshape(2, self.Lx, self.Ly)[0]), ax=ax[2])

        plt.savefig(plots_dir / "phase_diagram_inputs.png")
        plt.close()
