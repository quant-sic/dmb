from collections import defaultdict
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dmb.data.bose_hubbard_2d.nn_input import \
    get_nn_input_dimless_const_parameters
from dmb.data.bose_hubbard_2d.potential import get_quadratic_mu_potential, \
    get_square_mu_potential
from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2dTransforms
from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbard2dDataset
from dmb.paths import REPO_DATA_ROOT


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def create_wedding_cake_plot(
    mapping: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 40,
    muU_num_steps: int = 10,
    coefficient: float = -2.0,
) -> dict[str, plt.Figure]:
    """Create wedding cake plot for a given model and parameters."""
    ds = BoseHubbard2dDataset(
        dataset_dir_path=REPO_DATA_ROOT /
        f"datasets/bose_hubbard_2d/wedding_cake/{zVU}/{ztU}/{L}/{coefficient}",
        transforms=BoseHubbard2dTransforms(),
    )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities = [(ds.get_phase_diagram_sample(
        ztU=ztU, zVU=zVU, muU=_muU, L=L)[1][0] if ds.get_phase_diagram_sample(
            ztU=ztU, zVU=zVU, muU=_muU, L=L) is not None else np.ones((L, L)))
                        for _muU in muU]

    inputs = torch.stack(
        [
            get_nn_input_dimless_const_parameters(
                muU=get_quadratic_mu_potential(
                    [coefficient, coefficient],
                    L,
                    offset=_muU,
                ),
                ztU=ztU,
                zVU=zVU,
                cb_projection=True,
                target_density=target_density,
            ) for _muU, target_density in zip(muU, target_densities)
        ],
        dim=0,
    )

    outputs = mapping(inputs=inputs)
    figures_axes = defaultdict(lambda: plt.subplots(1, 1, figsize=(6, 6)))
    figures = {"wedding_cake": {}}

    for _muU, qmc_image, nn_image in zip(muU, target_densities,
                                         outputs["density"]):

        fig, ax = figures_axes[f"{_muU}"]
        ax.set_aspect("equal")

        X, Y = np.meshgrid(np.arange(L), np.arange(L))

        combined = np.concatenate((
            qmc_image[:int(L / 2) + 1],
            nn_image[int(L / 2) + 1:],
        ))
        combined[int(L / 2), :int(L / 2) +
                 1] = nn_image[int(L / 2), :int(L / 2) + 1]

        cm = ax.pcolormesh(X,
                           Y,
                           combined,
                           clim=(0, 2),
                           cmap="viridis",
                           linewidth=0,
                           rasterized=True)

        plt.hlines(
            y=int(L / 2) - 0.5,
            xmin=-0.5,
            xmax=int(L / 2) + 0.5,
            color="white",
        )
        plt.vlines(
            x=int(L / 2) + 0.5,
            ymin=int(L / 2) - 0.5,
            ymax=int(L / 2) + 0.5,
            color="white",
        )
        plt.hlines(
            y=int(L / 2) + 0.5,
            xmin=int(L / 2) + 0.5,
            xmax=L - 0.5,
            color="white",
        )

        plt.axis("off")
        colorbar(cm)

        plt.tight_layout()
        plt.close()

        figures["wedding_cake"]["{:.3}".format(_muU)] = fig

    return figures


def create_box_plot(
    mapping: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 50,
    square_size: int = 22,
) -> dict[str, plt.Figure]:
    """Create box plot for a given model and parameters."""
    ds = BoseHubbard2dDataset(
        dataset_dir_path=REPO_DATA_ROOT /
        f"datasets/bose_hubbard_2d/box/{zVU}/{ztU}/{L}",
        transforms=BoseHubbard2dTransforms(),
    )
    if len(ds) == 0:
        return {}

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities = [(ds.get_phase_diagram_sample(
        ztU=ztU, zVU=zVU, muU=_muU, L=L)[1][0] if ds.get_phase_diagram_sample(
            ztU=ztU, zVU=zVU, muU=_muU, L=L) is not None else np.ones((L, L)))
                        for _muU in muU]

    inputs = torch.stack(
        [
            get_nn_input_dimless_const_parameters(
                muU=get_square_mu_potential(
                    base_mu=0.0,
                    delta_mu=_muU,
                    square_size=square_size,
                    lattice_size=L,
                ),
                ztU=ztU,
                zVU=zVU,
                cb_projection=True,
                target_density=target_density,
            ) for _muU, target_density in zip(muU, target_densities)
        ],
        dim=0,
    )

    outputs = mapping(inputs=inputs)

    figures_axes = defaultdict(lambda: plt.subplots(1, 1, figsize=(6, 6)))

    figures = {"box": {}}

    for _muU, qmc_image, nn_image in zip(muU, target_densities,
                                         outputs["density"]):

        fig, ax = figures_axes[f"{_muU}"]
        ax.set_aspect("equal")

        X, Y = np.meshgrid(np.arange(L), np.arange(L))

        combined = np.concatenate((
            qmc_image[:int(L / 2) + 1],
            nn_image[int(L / 2) + 1:],
        ))
        combined[int(L / 2), :int(L / 2) +
                 1] = nn_image[int(L / 2), :int(L / 2) + 1]

        cm = ax.pcolormesh(X,
                           Y,
                           combined,
                           clim=(0, 2),
                           cmap="viridis",
                           linewidth=0,
                           rasterized=True)

        plt.hlines(
            y=int(L / 2) + 0.5,
            xmin=X[0, 0],
            xmax=X[int(L / 2), -1],
            color="white",
            lw=2,
        )

        plt.axis("off")
        colorbar(cm)

        plt.tight_layout()
        plt.close()

        figures["box"]["{:.3}".format(_muU)] = fig

    return figures


def create_box_cuts_plot(
    mapping: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 50,
    square_size: int = 22,
) -> dict[str, plt.Figure]:

    ds = BoseHubbard2dDataset(
        dataset_dir_path=REPO_DATA_ROOT /
        f"datasets/bose_hubbard_2d/box/{zVU}/{ztU}/{L}",
        transforms=BoseHubbard2dTransforms(),
    )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities_unflipped = torch.stack(
        [(ds.get_phase_diagram_sample(ztU=ztU, zVU=zVU, muU=_muU, L=L)[1][0]
          if ds.get_phase_diagram_sample(ztU=ztU, zVU=zVU, muU=_muU, L=L)
          is not None else torch.ones((L, L))) for _muU in muU],
        dim=0,
    )

    cut_position = int(L / 2)

    # flip qmc image vertically depending on which checkerboard pattern has a higher correlation with the target density
    target_densities = []
    for target_density in target_densities_unflipped:
        cb_1 = torch.ones_like(target_density)
        cb_1[::2, ::2] = 0
        cb_1[1::2, 1::2] = 0

        cb_2 = torch.ones_like(target_density)
        cb_2[1::2, ::2] = 0
        cb_2[::2, 1::2] = 0

        if torch.sum(target_density * cb_1) > torch.sum(target_density * cb_2):
            target_densities.append(target_density)
        else:
            target_densities.append(target_density.roll(1, dims=0))

    target_densities = torch.stack(target_densities, dim=0)

    inputs = torch.stack(
        [
            get_nn_input_dimless_const_parameters(
                muU=get_square_mu_potential(
                    base_mu=0.0,
                    delta_mu=_muU,
                    square_size=square_size,
                    lattice_size=L,
                ),
                ztU=ztU,
                zVU=zVU,
                cb_projection=True,
                target_density=target_density,
            ) for _muU, target_density in zip(muU, target_densities)
        ],
        dim=0,
    )
    outputs = mapping(inputs=inputs)

    nn_cuts = outputs["density"][:, cut_position]
    qmc_cuts = target_densities[:, cut_position]

    MU, X = np.meshgrid(muU, np.arange(L))

    qmc_image, nn_image = map(lambda c: np.stack(c, axis=0).T,
                              (qmc_cuts, nn_cuts))

    combined = np.concatenate((
        qmc_image[:int(L / 2) + 1],
        nn_image[int(L / 2) + 1:],
    ))
    # combined[int(L / 2), : int(L / 2) + 1] = nn_image[int(L / 2), : int(L / 2) + 1]

    fig, ax = plt.subplots(1, 1, figsize=(17.9219 * 0.3, 17.9219 * 0.25))

    mappable = ax.pcolormesh(MU, X, combined, linewidth=0, rasterized=True)

    plt.hlines(
        y=int(L / 2) + 0.5,
        xmin=X[0, 0],
        xmax=X[int(L / 2), -1],
        color="white",
        lw=2,
    )
    plt.xlabel(r"$\Delta V$")
    plt.ylabel(r"Site $j$")
    plt.colorbar(mappable)

    plt.xlim(MU[0, 0], MU[0, -1])
    plt.ylim(X[0, 0], X[-1, 0])
    # plt.xticks([1, 2, 3])

    plt.tight_layout()
    plt.close()

    return {"box_cuts": fig}


def plot_phase_diagram_mu_cut(
    mapping: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    zVU: float = 1.0,
    ztU: float = 0.25,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 16,
    muU_num_steps: int = 50,
) -> dict[str, plt.Figure]:
    """Plot the phase diagram of the Bose-Hubbard model for a given mu cut."""
    ds = BoseHubbard2dDataset(
        dataset_dir_path=REPO_DATA_ROOT / "datasets/bose_hubbard_2d/mu_cut" /
        f"{zVU}/{ztU}/{L}",
        transforms=BoseHubbard2dTransforms(),
    )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    inputs = torch.stack(
        [
            get_nn_input_dimless_const_parameters(
                muU=np.full((L, L), fill_value=_muU),
                ztU=ztU,
                zVU=zVU,
                cb_projection=True,
                target_density=np.ones((L, L)),
            ) for _muU in muU
        ],
        dim=0,
    )
    outputs = mapping(inputs=inputs)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    try:
        muU_qmc, n_qmc = zip(*[(ds.get_phase_diagram_position(i)[1],
                                ds_i[1][0]) for i, ds_i in enumerate(ds)])

        ax.scatter(muU_qmc, [n_qmc[i].max() for i in range(len(n_qmc))],
                   c="black",
                   label="QMC")
        ax.scatter(muU_qmc, [n_qmc[i].min() for i in range(len(n_qmc))],
                   c="black",
                   label="QMC")
    except ValueError:
        pass

    # max
    ax.scatter(
        muU,
        outputs["density"].max(axis=(-1, -2)),
        c="red",
        label="NN",
    )

    ax.scatter(
        muU,
        outputs["density"].min(axis=(-1, -2)),
        c="red",
        label="NN",
    )

    plt.legend()
    plt.xlabel(r"$\mu/U$")
    plt.ylabel(r"$n$")
    plt.tight_layout()
    plt.close()

    return {"mu_cut": fig}
