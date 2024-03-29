import json
from collections import defaultdict
from attrs import define
from pathlib import Path

import h5py

from dmb.data.bose_hubbard_2d.cpp_worm.worm.parameters import WormInputParameters
from dmb.utils import create_logger
from dmb.data.bose_hubbard_2d.simulation import SimulationOutput
import numpy as np

log = create_logger(__name__)


@define
class WormOutput(SimulationOutput):
    out_file_path: Path
    input_parameters: WormInputParameters

    @property
    def densities(self) -> np.ndarray | None:

        if not self.out_file_path.exists():
            log.warning(f"File {self.out_file_path} does not exist.")
            return None
        try:
            with h5py.File(self.out_file_path, "r") as f:
                densities = f["simulation"]["densities"][()]
            return densities
        except OSError as e:
            log.error(
                f"Exception occured during of file {self.out_file_path} loading: {e}"
            )
            return None

    @property
    def reshape_densities(self):
        if self.densities is None:
            return None
        else:
            try:
                return self.densities.reshape(
                    self.densities.shape[0],
                    self.input_parameters.Lx,
                    self.input_parameters.Ly,
                )
            except (ValueError, AttributeError) as e:
                log.error(
                    f"Exception occured during reshape: {e} for {self.out_file_path}"
                )
                return None

    @property
    def accumulator_observables(self):
        if not self.out_file_path.exists():
            return None

        h5_file = h5py.File(self.out_file_path, "r")

        observables_dict = {}

        observables_dict = defaultdict(dict)

        for obs, obs_dataset in h5_file["simulation/results"].items():
            for measure, value in obs_dataset.items():
                if isinstance(value, h5py.Dataset):
                    observables_dict[obs][measure] = value[()]

                elif isinstance(value, h5py.Group):
                    observables_dict[obs][measure] = {}
                    for sub_measure, sub_value in value.items():
                        observables_dict[obs][measure][sub_measure] = sub_value[()]

        return observables_dict

    @property
    def accumulator_vector_observables(self):
        observables_dict = self.accumulator_observables
        if observables_dict is None:
            return None

        # filter out non vector observables
        vector_observables = {
            obs: obs_dict
            for obs, obs_dict in observables_dict.items()
            if (
                obs_dict["mean"]["value"].ndim == 1
                and len(obs_dict["mean"]["value"])
                == self.input_parameters.Lx * self.input_parameters.Ly
            )
        }

        return vector_observables
