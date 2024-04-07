import json
from collections import defaultdict
from attrs import define, field
from pathlib import Path

import h5py

from dmb.data.bose_hubbard_2d.worm_qmc.worm.parameters import WormInputParameters
from dmb.utils import create_logger
from dmb.data.bose_hubbard_2d.simulation import SimulationOutput
import numpy as np
from logging import Logger

logger = create_logger(__name__)


@define
class WormOutput(SimulationOutput):

    out_file_path: Path
    input_parameters: WormInputParameters
    logging_instance: Logger = field(default=logger)

    def reshape_observable(self, observable: np.ndarray) -> np.ndarray:
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
                densities = f["simulation"]["densities"][()]
        except OSError as e:
            self.logging_instance.error(
                f"Exception occured during of file {self.out_file_path} loading: {e}"
            )
            return None

        # reshape densities to (n_samples, Lx, Ly)
        try:
            densities = self.reshape_observable(densities)
        except ValueError as e:
            self.logging_instance.error(
                f"Exception occured during reshape: {e} for {self.out_file_path}"
            )
            return None

        return densities

    @property
    def accumulator_observables(
        self,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]] | None:
        """Return the observables from the simulation accumulator.

        If the file does not exist, return None. If the file exists but the
        observables dataset is not present, return None. If the file exists and
        the observables dataset is present, return the observables.
        """

        if not self.out_file_path.exists():
            self.logging_instance.warning(f"File {self.out_file_path} does not exist.")
            return None

        try:
            with h5py.File(self.out_file_path, "r") as f:
                observables = f["simulation"]["results"]
        except OSError as e:
            self.logging_instance.error(
                f"Exception occured during of file {self.out_file_path} loading: {e}"
            )
            return None

        accumulator_observables = defaultdict(dict)
        for obs, obs_dataset in observables.items():
            for measure, value in obs_dataset.items():
                if isinstance(value, h5py.Dataset):
                    accumulator_observables[obs][measure] = self.reshape_observable(
                        value[()]
                    )

                elif isinstance(value, h5py.Group):
                    accumulator_observables[obs][measure] = {}
                    for sub_measure, sub_value in value.items():
                        accumulator_observables[obs][measure][sub_measure] = (
                            self.reshape_observable(sub_value[()])
                        )
