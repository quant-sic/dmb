"""Module to handle output of the worm simulation."""

import abc
from logging import Logger
from pathlib import Path

import h5py
import numpy as np
from attrs import define, field

from dmb.logging import create_logger

from .parameters import WormInputParameters

log = create_logger(__name__)


class Output(metaclass=abc.ABCMeta):
    """Interface for the output of the worm simulation."""

    @property
    @abc.abstractmethod
    def densities(self) -> np.ndarray | None:
        """Return the densities from the output file.

        If the file does not exist, return None. If the file exists but the
        densities dataset is not present, return None. If the file exists and
        the densities dataset is present, return the densities.
        """


@define
class WormOutput(Output):
    """Class to handle output of the worm simulation."""

    out_file_path: Path
    input_parameters: WormInputParameters
    logging_instance: Logger = field(default=log)

    def _reshape_observable(self, observable: np.ndarray) -> np.ndarray:
        """Reshape the observable to (n_samples, Lx, Ly)."""
        if len(observable.shape) == 3:
            return observable

        if len(observable.shape) == 2:
            return observable.reshape(
                observable.shape[0],
                self.input_parameters.Lx,
                self.input_parameters.Ly,
            )

        raise ValueError(f"Observable has invalid shape: {observable.shape}")

    @property
    def densities(self) -> np.ndarray | None:
        """Return the densities from the output file.

        If the file does not exist, return None. If the file exists but the
        densities dataset is not present, return None. If the file exists and
        the densities dataset is present, return the densities.
        """

        if not self.out_file_path.exists():
            self.logging_instance.warning(f"File {self.out_file_path} does not exist.")
            return None

        try:
            with h5py.File(self.out_file_path, "r") as f:
                densities: np.ndarray = f["simulation"]["densities"][()]
        except (KeyError, OSError) as e:
            self.logging_instance.error(
                f"Exception occured during of file {self.out_file_path} loading: {e}"
            )
            return None

        # reshape densities to (n_samples, Lx, Ly)
        try:
            densities = self._reshape_observable(densities)
        except (ValueError, TypeError) as e:
            self.logging_instance.error(
                f"Exception occured during reshape: {e} for {self.out_file_path}"
            )
            return None

        return densities
