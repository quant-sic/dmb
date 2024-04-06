"""Module for the worm input parameters."""

__all__ = [
    "WormInputParameters",
    "plot_worm_input_parameters",
    "plot_phase_diagram_worm_input_parameters",
]

from pathlib import Path
from typing import Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from dmb.utils import create_logger
import os
from attrs import define
from dataclasses import dataclass

log = create_logger(__name__)


@dataclass
class WormInputParameters:
    """Class for the input parameters of the worm simulation."""

    mu: Union[np.ndarray, float]
    t_hop: Union[np.ndarray, float] = 1.0
    U_on: Union[np.ndarray, float] = 4.0
    V_nn: Union[np.ndarray, float] = 0.0
    model: str = "BoseHubbard"
    runtimelimit: int = 24 * 60 * 60
    sweeps: int = 25000
    thermalization: int = 100
    Lx: int = 4
    Ly: int = 4
    Lz: int = 1
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

    h5_path: Optional[Path] = None
    checkpoint: Optional[Path] = None
    outputfile: Optional[Path] = None

    h5_path_relative: Optional[Path] = None
    checkpoint_relative: Optional[Path] = None
    outputfile_relative: Optional[Path] = None

    mu_power: Optional[float] = None
    mu_offset: Optional[float] = None

    @classmethod
    def from_dir(cls, save_dir_path: Path):
        # Read ini file
        with open(save_dir_path / "parameters.ini", "r") as f:
            lines = f.readlines()

        # Fill dictionary for ini parameters
        params = {}
        for line in lines:
            if not line.startswith("#"):
                key, value = map(lambda s: s.strip(), line.split("="))

                if key in cls.__dataclass_fields__.keys():
                    # convert to correct type
                    if key in (
                        "checkpoint",
                        "outputfile",
                        "h5_path",
                        "h5_path_relative",
                        "checkpoint_relative",
                        "outputfile_relative",
                    ):
                        value = Path(value)
                    elif key in ("mu_power", "mu_offset"):
                        value = float(value) if value != "None" else None

                    elif key not in ("mu", "t_hop", "U_on", "V_nn"):
                        value = cls.__dataclass_fields__[key].type(value)

                    # add to dictionary
                    params[key] = value

        # read in h5 site dependent arrays
        with h5py.File(save_dir_path / "parameters.h5", "r") as file:
            for name in ("mu", "t_hop", "U_on", "V_nn"):
                params[name] = file[f"/{name}"][()]

        # Create input parameters
        return cls(**params)

    def save_h5(self):
        if self.h5_path is None:
            raise RuntimeError("h5_path must be set")

        # create parent directory if it does not exist
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)

        # Create h5 file
        with h5py.File(self.h5_path, "w") as file:
            for name, attribute in (
                ("mu", self.mu),
                ("t_hop", self.t_hop),
                ("U_on", self.U_on),
                ("V_nn", self.V_nn),
            ):
                file[f"/{name}"] = (
                    attribute if isinstance(attribute, float) else attribute.flatten()
                )

    @property
    def ini_path(self):
        if self._ini_path is None:
            raise RuntimeError(
                "ini_path must be set. By saving the parameters to a directory, the ini_path is set automatically."
            )
        else:
            return self._ini_path

    @ini_path.setter
    def ini_path(self, ini_path: Path):
        self._ini_path = ini_path

    def to_ini(self, checkpoint, outputfile, save_path: Path):
        # create parent directory if it does not exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create ini file
        with open(save_path, "w") as f:
            for key in self.__dataclass_fields__.keys():
                if not (
                    key in ("mu", "t_hop", "U_on", "V_nn")
                    and isinstance(self.__getattribute__(key), np.ndarray)
                ):
                    f.write(f"{key} = {self.__getattribute__(key)}\n")

            if self.h5_path is None:
                raise RuntimeError("h5_path must be set")
            else:
                f.write(f"site_arrays = {self.h5_path}\n")

    def set_paths(self, save_dir_path: Path, reset_paths: bool = False):
        self.outputfile = save_dir_path / "output.h5"
        self.outputfile_relative = self.outputfile.relative_to(save_dir_path)

        self.checkpoint = save_dir_path / "checkpoint.h5"
        self.checkpoint_relative = self.checkpoint.relative_to(save_dir_path)

        self.h5_path = save_dir_path / "parameters.h5"
        self.ini_path = save_dir_path / "parameters.ini"

        self.h5_path_relative = self.h5_path.relative_to(save_dir_path)

        # if checkpoints already exist, edit the checkpoint path
        if reset_paths:
            try:
                self.update_paths_in_checkpoint()
            except OSError as e:
                log.warning(f"Exception occured during checkpoint editing: {e}")

    def update_paths_in_checkpoint(self):
        def iterfill_path(path, obj, file, contained_str, paths_list):
            if isinstance(obj, h5py.Dataset):
                if isinstance(obj[()], (bytes, str)):
                    out = obj[()].decode("utf-8")
                    if contained_str in out:
                        paths_list.append(path)

        if self.checkpoint.exists():
            for ckpt_path in self.checkpoint.parent.glob("checkpoint.h5*"):
                with h5py.File(ckpt_path, "r+") as file:
                    paths_list = []
                    file.visititems(
                        partial(
                            iterfill_path,
                            file=file,
                            paths_list=paths_list,
                            contained_str="checkpoint.h5",
                        )
                    )
                    for path in paths_list:
                        file[path][...] = str(self.checkpoint).encode("utf-8")

                    paths_list = []
                    file.visititems(
                        partial(
                            iterfill_path,
                            file=file,
                            paths_list=paths_list,
                            contained_str="output.h5",
                        )
                    )
                    for path in paths_list:
                        file[path][...] = str(self.outputfile).encode("utf-8")

                    ini_values = file["parameters"].attrs["ini_values"]
                    ini_keys = list(file["parameters"].attrs["ini_keys"])
                    for ini_key in (
                        "checkpoint",
                        "h5_path",
                        "h5_path_relative",
                        "outputfile",
                        "site_arrays",
                    ):
                        ini_key_idx = ini_keys.index(ini_key)

                        input_parameters_key = (
                            ini_key if ini_key != "site_arrays" else "h5_path"
                        )
                        ini_values[ini_key_idx] = str(
                            self.__getattribute__(input_parameters_key)
                        )

                    file["parameters"].attrs.modify("ini_values", ini_values)
                    file["parameters"].attrs.modify(
                        "origins",
                        [
                            os.environ["WORM_MPI_EXECUTABLE"],
                            str(self.outputfile),
                            str(self.checkpoint),
                        ],
                    )

        if self.outputfile.exists():
            with h5py.File(self.outputfile, "r+") as file:
                paths_list = []
                file.visititems(
                    partial(
                        iterfill_path,
                        file=file,
                        paths_list=paths_list,
                        contained_str="checkpoint.h5",
                    )
                )
                for path in paths_list:
                    file[path][...] = str(self.outputfile).encode("utf-8")

    def save(self, save_dir_path: Path, reset_paths: bool = False):
        # create parent directory if it does not exist
        save_dir_path.parent.mkdir(parents=True, exist_ok=True)

        # set paths
        self.set_paths(save_dir_path, reset_paths=reset_paths)

        # Create ini file
        self.to_ini(
            save_path=self.ini_path,
            checkpoint=self.checkpoint,
            outputfile=self.outputfile,
        )
        self.save_h5()

    def save_parameters(self, save_dir_path: Path):
        self.save(save_dir_path)
        self.save_h5()


def plot_worm_input_parameters(
    parameters: WormInputParameters, plots_dir: Path
) -> None:
    """Plot the input parameters of the worm simulation."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(parameters.mu.reshape(parameters.Lx, parameters.Ly))
    ax[0].set_title("mu")
    ax[1].imshow(parameters.t_hop.reshape(2, parameters.Lx, parameters.Ly)[0])
    ax[1].set_title("t_hop")
    ax[2].imshow(parameters.U_on.reshape(parameters.Lx, parameters.Ly))
    ax[2].set_title("U_on")
    ax[3].imshow(parameters.V_nn.reshape(2, parameters.Lx, parameters.Ly)[0])
    ax[3].set_title("V_nn")

    # set axes off
    for ax_ in ax:
        ax_.axis("off")

    # set colorbars
    fig.colorbar(
        ax[0].imshow(parameters.mu.reshape(parameters.Lx, parameters.Ly)),
        ax=ax[0],
    )
    fig.colorbar(
        ax[1].imshow(parameters.t_hop.reshape(2, parameters.Lx, parameters.Ly)[0]),
        ax=ax[1],
    )
    fig.colorbar(
        ax[2].imshow(parameters.U_on.reshape(parameters.Lx, parameters.Ly)),
        ax=ax[2],
    )
    fig.colorbar(
        ax[3].imshow(parameters.V_nn.reshape(2, parameters.Lx, parameters.Ly)[0]),
        ax=ax[3],
    )

    plt.savefig(plots_dir / "inputs.png")
    plt.close()


def plot_phase_diagram_worm_input_parameters(
    parameters: WormInputParameters, plots_dir: Path
):
    """Plot the phase diagram input parameters of the worm simulation."""
    muU = parameters.mu / parameters.U_on
    ztU = 4 * parameters.t_hop / parameters.U_on[None, ...]
    zVU = 4 * parameters.V_nn / parameters.U_on[None, ...]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(muU.reshape(parameters.Lx, parameters.Ly))
    ax[0].set_title("muU")

    ax[1].imshow(ztU.reshape(2, parameters.Lx, parameters.Ly)[0])
    ax[1].set_title("ztU")

    ax[2].imshow(zVU.reshape(2, parameters.Lx, parameters.Ly)[0])
    ax[2].set_title("zVU")

    # set axes off
    for ax_ in ax:
        ax_.axis("off")

    # set colorbars
    fig.colorbar(ax[0].imshow(muU.reshape(parameters.Lx, parameters.Ly)), ax=ax[0])
    fig.colorbar(
        ax[1].imshow(ztU.reshape(2, parameters.Lx, parameters.Ly)[0]), ax=ax[1]
    )
    fig.colorbar(
        ax[2].imshow(zVU.reshape(2, parameters.Lx, parameters.Ly)[0]), ax=ax[2]
    )

    plt.savefig(plots_dir / "phase_diagram_inputs.png")
    plt.close()
