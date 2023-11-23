import subprocess
from dataclasses import dataclass
from pathlib import Path
import h5py

from dmb.utils import create_logger
from collections import defaultdict
import json
from dmb.data.bose_hubbard_2d.cpp_worm.worm.parameters import WormInputParameters
from dmb.data.bose_hubbard_2d.cpp_worm.worm.ac import PrimaryAnalysis, DerivedAnalysis
from functools import cached_property
import numpy as np

log = create_logger(__name__)


@dataclass
class WormOutput:
    out_file_path: Path
    input_parameters: WormInputParameters

    @cached_property
    def densities(self):
        with h5py.File(self.out_file_path, "r") as f:
            densities = f["simulation"]["densities"][()]
        return densities

    @cached_property
    def density_errors(self):
        analysis = PrimaryAnalysis(
            self.densities.reshape(1, *self.densities.shape),
            rep_sizes=[len(self.densities)],
            name=[
                f"{int(idx/self.input_parameters.Lx)}{idx%self.input_parameters.Lx}"
                for idx in range(self.input_parameters.Lx**2)
            ],
        )
        analysis.mean()
        results = analysis.errors()
        return results

    @cached_property
    def density_variance(self):
        return np.var(self.densities, axis=0)

    @property
    def observables(self):
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
    def vector_observables(self):
        observables_dict = self.observables

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


class SimulationRecord(object):
    def __init__(self, record_dir: Path):
        self.record_file_path = record_dir / "record.json"

        if self.record_file_path.exists():
            with open(self.record_file_path, "r") as f:
                self.record = json.load(f)
        else:
            self.record = {}

    def save(self):
        with open(self.record_file_path, "w") as f:
            json.dump(self.record, f)

    def update(self, record: dict):
        self.record.update(record)

        self.save()

    def __getitem__(self, key: str):
        return self.record.get(key, None)

    def __setitem__(self, key: str, value):
        self.record[key] = value

        self.save()
