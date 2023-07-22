from typing import Callable, Tuple,Optional
import numpy as np
import FyeldGenerator
import numpy as np
from scipy import stats


def grf(shape: Tuple[int, int], cov_func: Callable[[int, int], float]) -> np.ndarray:
    """
    Generate a Gaussian random field with given covariance function. 2D only.

    Args:
        shape (Tuple[int, int]): Shape of the field.
        cov_func (Callable[[int, int], float]): Covariance function.

    Returns:
        np.ndarray: Random field with desired covariance.
    """

    tx = np.arange(shape[0])
    ty = np.arange(shape[1])
    Rows = np.zeros(shape)
    Cols = np.zeros(shape)

    Rows = cov_func(
        tx[:, None] - tx[0], ty[None, :] - ty[0]
    )  # rows of blocks of cov matrix
    Cols = cov_func(
        tx[0] - tx[:, None], ty[None, :] - ty[0]
    )  # columns of blocks of cov matrix

    # create the first row of the block circulant matrix with circular blocks and store it as a matrix suitable for fft2;
    BlkCirc_row = np.concatenate(
        (
            np.concatenate((Rows, Cols[:, :0:-1]), axis=1),
            np.concatenate((Cols[:0:-1, :], Rows[:0:-1, :0:-1]), axis=1),
        ),
        axis=0,
    )

    # compute eigen-values
    lam = np.real(np.fft.fft2(BlkCirc_row)) / (2 * shape[0] - 1) / (2 * shape[1] - 1)
    if (lam < 0).any() and np.abs(np.min(lam[lam < 0])) > 10**-15:
        raise ValueError("Could not find positive definite embedding!")
    else:
        lam[lam < 0] = 0
        lam = np.sqrt(lam)

    # #generate field with covariance given by block circular matrix
    F = np.fft.fft2(
        lam
        * (
            np.random.randn(2 * shape[0] - 1, 2 * shape[1] - 1)
            + 1j * np.random.randn(2 * shape[0] - 1, 2 * shape[1] - 1)
        )
    )
    F = F[: shape[0], : shape[1]]  # extract subblock with desired covariance

    return np.real(F)


# def coloured_noise():

#     import matplotlib.pyplot as plt

#     sz = 24
#     dim = 2
#     exponent = -2

#     import numpy as np

#     # white noise signal
#     white_noise_signal = np.random.normal(0, 1, size=(sz,) * dim)

#     # create Fourier filter
#     # first create a line with squared distances from mid-point
#     # np.ones(sz)
#     # for i in range(sz):
#     #     dist[i] = np.linalg.norm(i - sz/2 - 1)**2

#     distance_to_center = (np.arange(sz) - sz/2 - 1)**2
#     squared_distances_added = np.meshgrid(distance_to_center,distance_to_center)
#     dist_tot = np.sqrt(sum(squared_distances_added))

#     dist_tot[dist_tot==0] = 1
#     filt = dist_tot**(exponent*dim/2)

#     # # Fourier transform white noise, then fftshift to shift 0-frequency
#     # # to the center of the array, to align with filter whose
#     # # 0-frequency is also at the center. Otherwise multiplying
#     # # them together will not multiply corresponding elements.
#     wnf = np.fft.fftshift(np.fft.fftn(white_noise_signal))

#     # # multiply with frequency filter
#     wnf_filt = wnf * filt

#     # # ifftshift to first shift back the Fourier transform
#     # # to have 0-frequency at the start again. This lets
#     # # ifftn do inverse Fourier transform correctly
#     colored_noise_signal = np.real(np.fft.ifftn(np.fft.ifftshift(wnf_filt)))

#     plt.imshow(colored_noise_signal)


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
        power = stats.loguniform.rvs(0.1, 10, size=1)

    scale = float(np.random.uniform(.3, 1, size=1)) * desired_abs_max

    potential = periodic_grf(shape, power)
    return float(power), get_offset_rescaled_trapping_potential(potential, scale)
