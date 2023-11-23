from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
from typing import Optional, Union

from dmb.utils import create_logger


log = create_logger(__name__)


@dataclass
class WormInputParameters:
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

                    elif not key in ("mu", "t_hop", "U_on", "V_nn"):
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

    def save(
        self,
        save_dir_path: Path,
        checkpoint: Optional[Path] = None,
        outputfile: Optional[Path] = None,
    ):
        # create parent directory if it does not exist
        save_dir_path.parent.mkdir(parents=True, exist_ok=True)

        self.outputfile = (
            save_dir_path / "output.h5" if outputfile is None else outputfile
        )
        self.outputfile_relative = Path("output.h5")

        self.checkpoint = (
            save_dir_path / "checkpoint.h5" if checkpoint is None else checkpoint
        )
        self.checkpoint_relative = Path("checkpoint.h5")

        self.h5_path = save_dir_path / "parameters.h5"
        self.ini_path = save_dir_path / "parameters.ini"

        self.h5_path_relative = Path("parameters.h5")

        # Create ini file
        self.to_ini(
            save_path=self.ini_path,
            checkpoint=self.checkpoint,
            outputfile=self.outputfile,
        )
        self.save_h5()
