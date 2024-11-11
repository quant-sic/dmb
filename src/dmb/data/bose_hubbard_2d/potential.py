"""Module for generating trapping potentials for 2D Bose-Hubbard model"""

from typing import Optional

import FyeldGenerator
import numpy as np
from scipy import stats


def periodic_grf(shape: tuple[int, int], power: float) -> np.ndarray:
    """Generate a periodic Gaussian random field with power-law power spectrum
        in Fourier space."""

    def pk(k: np.ndarray) -> np.ndarray:
        return np.power(k, -power)

    def distribution(shape: tuple[int, int]) -> np.ndarray:
        # Build a unit-distribution of complex numbers with random phase
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a + 1j * b

    field: np.ndarray = FyeldGenerator.generate_field(distribution, pk, shape)

    return field


def get_random_trapping_potential(
        shape: tuple[int, int],
        desired_abs_max: float,
        power: Optional[float] = None) -> tuple[float, np.ndarray]:
    """Generate a random trapping potential.

    Args:
        shape: shape of the potential
        desired_abs_max: desired absolute maximum of the potential
        power: power of the power-law power spectrum

    Returns:
        tuple[float, np.ndarray]: power and potential
    """
    if power is None:
        power = stats.loguniform.rvs(0.1, 10)

    scale = float(np.random.uniform(0.3, 1)) * desired_abs_max

    potential = periodic_grf(shape, power)
    potential_rescaled = potential * scale / abs(potential).max()

    return float(power), potential_rescaled


def get_square_mu_potential(base_mu: float, delta_mu: float, square_size: int,
                            lattice_size: int) -> np.ndarray:
    """Generate a 2D square mu array with a square of higher mu in the center."""

    mu = np.full(shape=(lattice_size, lattice_size), fill_value=base_mu)
    mu[
        int(float(lattice_size) / 2 - float(square_size) /
            2):int(np.ceil(float(lattice_size) / 2 + float(square_size) / 2)),
        int(float(lattice_size) / 2 - float(square_size) /
            2):int(np.ceil(float(lattice_size) / 2 + float(square_size) / 2)),
    ] = base_mu + delta_mu

    return mu


def get_quadratic_mu_potential(
    coeffitients: tuple[float, float],
    lattice_size: int,
    center: tuple[float, float] | None = None,
    offset: float = 0,
) -> np.ndarray:
    """Generate a 2D quadratic mu array

    Args:
        coeffitients: tuple of two floats, quadratic coefficients
        lattice_size: size of the lattice
        center: center of the quadratic mu array
        offset: offset of the quadratic mu array

    Returns:
        np.ndarray: 2D quadratic mu array
    """
    if center is None:
        center = (float(lattice_size) / 2, float(lattice_size) / 2)

    xx, yy = np.meshgrid(np.arange(lattice_size), np.arange(lattice_size))
    mu = (offset + coeffitients[0] * (xx - center[0])**2 /
          ((float(lattice_size) * 0.5)**2) + coeffitients[1] * (yy - center[1])**2 /
          ((float(lattice_size) * 0.5)**2))

    return mu
