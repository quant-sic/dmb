from typing import Callable, Optional, Tuple

import FyeldGenerator
import numpy as np
from scipy import stats


def periodic_grf(shape: Tuple[int, int], power: float) -> np.ndarray:
    def Pkgen(n):
        def Pk(k):
            return np.power(k, -n)

        return Pk

    def distrib(shape):
        # Build a unit-distribution of complex numbers with random phase
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a + 1j * b

    field = FyeldGenerator.generate_field(distrib, Pkgen(power), shape)

    return field


def get_offset_rescaled_trapping_potential(
    potential: np.ndarray, desired_abs_max: float
) -> np.ndarray:
    abs_max = abs(potential).max()
    return potential * desired_abs_max / abs_max


def get_random_trapping_potential(
    shape: Tuple[int, int], desired_abs_max: float, power: Optional[float] = None
) -> Tuple[float, np.ndarray]:
    if power is None:
        power = stats.loguniform.rvs(0.1, 10)

    scale = float(np.random.uniform(0.3, 1)) * desired_abs_max

    potential = periodic_grf(shape, power)
    return float(power), get_offset_rescaled_trapping_potential(potential, scale)
