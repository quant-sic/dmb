"""Plotting functions for the phase diagram of the Bose-Hubbard model in 2D."""

from collections import defaultdict
from typing import Callable, Generator

import matplotlib.pyplot as plt
import numpy as np
import torch

from dmb.data.bose_hubbard_2d.nn_input import \
    get_nn_input_dimless_const_parameters


def phase_diagram_uniform_inputs_iter(
    n_samples: int,
    zVU: float = 1.0,
    muU_range: tuple[float, float] = (-0.1, 3.1),
    ztU_range: tuple[float, float] = (0.05, 0.85),
) -> Generator[tuple[float, float, torch.Tensor], None, None]:
    """Generate inputs for the phase diagram with uniform sampling in muU and ztU.

    Args:
        n_samples: Number of samples in each dimension.
        zVU: Value of zVU.
        muU_range: Range of muU values.
        ztU_range: Range of ztU values.

    Yields:
        Tuple of muU, ztU, and inputs.
    """

    muU = np.linspace(*muU_range, n_samples)
    ztU = np.linspace(*ztU_range, n_samples)

    MUU, ZTU = np.meshgrid(muU, ztU)

    MUU = MUU.flatten()
    ZTU = ZTU.flatten()

    # cb version 1
    fake_target_density = np.zeros((16, 16))
    fake_target_density[::2, ::2] = 1.0

    for i in range(n_samples * n_samples):
        yield MUU[i], ZTU[i], get_nn_input_dimless_const_parameters(
            muU=np.full((16, 16), fill_value=MUU[i]),
            ztU=ZTU[i],
            zVU=zVU,
            cb_projection=True,
            target_density=fake_target_density,
        )


def phase_diagram_uniform_inputs(
        n_samples: int,
        zVU: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate inputs for the phase diagram with uniform sampling in muU and ztU.

    Args:
        n_samples: Number of samples in each dimension.
        zVU: Value of zVU.

    Returns:
        Tuple of muU, ztU, and inputs.
    """
    MUU, ZTU, inputs = zip(
        *list(phase_diagram_uniform_inputs_iter(n_samples, zVU=zVU)))

    MUU = torch.from_numpy(np.array(MUU)).float()
    ZTU = torch.from_numpy(np.array(ZTU)).float()
    inputs = torch.stack(inputs, dim=0)

    return MUU, ZTU, inputs


def add_phase_boundaries(ax: plt.Axes) -> None:
    """Add phase boundaries to an axis."""
    red = [
        (0, 0),
        (2, 62),
        (3, 124),
        (3, 183),
        (3, 245),
        (3.5, 306),
        (4, 368),
        (5, 430),
        (6, 478),
        (6.5, 491),
        (7, 500),
        (8, 506),
        (8, 502),
        (9, 494),
        (10, 484),
        (13.5, 439),
        (18, 386),
        (24, 334),
        (32, 287),
        (42, 247),
        (55, 214),
        (72, 198),
        (90, 199),
        (73, 216),
        (60, 249),
        (51, 295),
        (45, 345),
        (42, 381),
        (42, 394),
        (42, 408),
        (43, 415),
        (45, 415),
        (47, 413),
        (53, 402),
        (62, 389),
        (71, 383),
        (81, 387),
        (90, 398),
        (81, 416),
        (74.5, 442),
        (71, 462),
        (68, 483),
        (65, 508),
        (64, 522),
        (64, 529),
        (65, 542),
        (66, 544),
        (67, 548),
        (71, 556),
        (74, 561),
        (78, 567),
        (84, 580),
        (90, 597),
    ]
    distances = (199, 230 / 0.15)

    red_dots = []
    for point in red:
        x, y = (
            np.cos(point[0] * np.pi / 180) * point[1] / distances[1],
            np.sin(point[0] * np.pi / 180) * point[1] / distances[0],
        )
        red_dots.append((x, y))

    ax.plot(*zip(*red_dots), marker="o", c="red")

    blue_1 = [
        (90, 199),
        (73, 210),
        (58, 235),
        (47, 271),
        (38, 313),
        (31, 358),
        (25, 403),
        (18.5, 450),
        (12.5, 499),
        (11, 509),
        (10, 513),
        (8, 515),
        (7, 508),
        (6.5, 503),
    ]
    blue_dots_1 = []
    for point in blue_1:
        x, y = (
            np.cos(point[0] * np.pi / 180) * point[1] / distances[1],
            np.sin(point[0] * np.pi / 180) * point[1] / distances[0],
        )
        blue_dots_1.append((x, y))

    ax.plot(*zip(*blue_dots_1), marker="o", c="blue")

    blue_2 = [
        (90, 398),
        (81, 406),
        (73, 425),
        (66, 453),
        (61, 495),
        (59, 527),
        (58, 555),
        (58, 568),
        (59, 581),
        (60, 590),
        (63, 605),
        (67, 616),
        (73, 618),
        (78, 610),
        (84, 602),
        (90, 597),
    ]
    blue_dots_2 = []
    for point in blue_2:
        x, y = (
            np.cos(point[0] * np.pi / 180) * point[1] / distances[1],
            np.sin(point[0] * np.pi / 180) * point[1] / distances[0],
        )
        blue_dots_2.append((x, y))

    ax.plot(*zip(*blue_dots_2), marker="o", c="blue")


def plot_phase_diagram(
    mapping: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    n_samples: int = 250,
    zVU: int = 1.0,
) -> dict[str, dict[str, plt.Figure]]:
    """Plot the phase diagram of the Bose-Hubbard model.

    Args:
        mapping: Model to use for prediction. Returns a dictionary of observables.
        n_samples: Number of samples in each dimension.
        zVU: Value of zVU.

    Returns:
        Dictionary of figures.
    """
    MUU, ZTU, inputs = phase_diagram_uniform_inputs(n_samples=n_samples,
                                                    zVU=zVU)
    outputs = mapping(inputs=inputs)

    reductions = {
        "mean": lambda x: x.mean(axis=(-1, -2)),
        "std": lambda x: x.std(axis=(-1, -2)),
        "max-min": lambda x: (x.max(axis=(-1, -2)) - x.min(axis=(-1, -2))),
        "max": lambda x: x.max(axis=(-1, -2)),
        "min": lambda x: x.min(axis=(-1, -2)),
    }

    figures_out = defaultdict(dict)

    for obs, output_obs in outputs.items():
        for name, reduction in reductions.items():
            figures_out[obs][name] = plt.figure()

            plt.pcolormesh(
                ZTU.view(n_samples, n_samples).cpu().numpy(),
                MUU.view(n_samples, n_samples).cpu().numpy(),
                reduction(output_obs).reshape(n_samples, n_samples),
            )

            if zVU == 1.0:
                add_phase_boundaries(plt.gca())
                plt.ylim([0, 3])
                plt.xlim([0.1, 0.5])

                if name == "max-min" and obs == "density":
                    plt.clim([0, 1])

            elif zVU == 1.5:
                plt.ylim([0, 3])
                plt.xlim([0.1, 0.8])

                if name == "max-min" and obs == "density":
                    plt.clim([0, 3])

            plt.xlabel(r"$4J/U$")
            plt.ylabel(r"$\mu/{U}$")
            plt.colorbar()
            plt.tight_layout()

            plt.close()

    return figures_out
