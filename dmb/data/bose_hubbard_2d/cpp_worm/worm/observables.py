import numpy as np

from dmb.data.bose_hubbard_2d.cpp_worm.worm.outputs import WormOutput
from dmb.utils import create_logger

log = create_logger(__name__)


class SimulationObservables:
    """Class for computing observables from a simulation."""

    @classmethod
    @property
    def observable_name_function_map(cls):
        return {
            "density": cls.get_density_mean,
            "density_variance": cls.get_density_variance,
            "density_density_corr_0": cls.get_density_density_corr_0,
            "density_density_corr_1": cls.get_density_density_corr_1,
            "density_density_corr_2": cls.get_density_density_corr_2,
            "density_density_corr_3": cls.get_density_density_corr_3,
            "density_squared": cls.get_density_squared_mean,
        }

    def __init__(self, output: WormOutput):
        self.output = output

    def __getitem__(self, key) -> float:
        return self.observable_name_function_map[key](self.output)

    def __contains__(self, key) -> bool:
        return key in self.observable_name_function_map

    @classmethod
    @property
    def observables_names(cls):
        return list(cls.observable_name_function_map.keys())

    @staticmethod
    def get_density_distribution(output: WormOutput) -> np.ndarray:
        return output.reshape_densities.mean(axis=0)

    @staticmethod
    def get_density_variance(output: WormOutput) -> np.ndarray:
        return output.reshape_densities.var(axis=0)

    @staticmethod
    def get_density_mean(output: WormOutput) -> np.ndarray:
        return output.reshape_densities.mean(axis=0)

    @staticmethod
    def get_density_density_corr_0(output: WormOutput) -> np.ndarray:
        return (
            np.roll(output.reshape_densities, axis=1, shift=1)
            * output.reshape_densities
        ).mean(axis=0)

    @staticmethod
    def get_density_density_corr_1(output: WormOutput) -> np.ndarray:
        densities = output.reshape_densities
        return (np.roll(densities, axis=2, shift=1) * densities).mean(axis=0)

    @staticmethod
    def get_density_density_corr_2(output: WormOutput) -> np.ndarray:
        densities = output.reshape_densities
        return (
            np.roll(np.roll(densities, axis=1, shift=1), axis=2, shift=1) * densities
        ).mean(axis=0)

    @staticmethod
    def get_density_density_corr_3(output: WormOutput) -> np.ndarray:
        densities = output.reshape_densities
        return (
            np.roll(np.roll(densities, axis=1, shift=-1), axis=2, shift=1) * densities
        ).mean(axis=0)

    @staticmethod
    def get_density_squared_mean(output: WormOutput) -> np.ndarray:
        return (output.reshape_densities**2).mean(axis=0)
