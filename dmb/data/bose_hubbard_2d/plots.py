import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dmb.data.bose_hubbard_2d.network_input import \
    net_input_dimless_const_parameters
from dmb.data.bose_hubbard_2d.phase_diagram import model_predict
from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbardDataset
from dmb.data.bose_hubbard_2d.worm.scripts.sandbox.box import get_square_mu
from dmb.data.bose_hubbard_2d.worm.scripts.sandbox.wedding_cake import \
    get_quadratic_mu
from dmb.paths import REPO_DATA_ROOT


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def create_wedding_cake_plot(
    model,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 40,
    muU_num_steps: int = 10,
    coefficient: float = -2.0,
):

    path = REPO_DATA_ROOT / f"wedding_cake/{zVU}/{ztU}/{L}/{coefficient}"

    ds = BoseHubbardDataset(
        data_dir=path,
        clean=True,
        observables=[
            "density",
            "density_variance",
            "density_density_corr_0",
            "density_density_corr_1",
            "density_density_corr_2",
            "density_density_corr_3",
            "density_squared",
        ],
        max_density_error=0.015,
        reload=True,
        verbose=False,
    )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities = [(ds.get_phase_diagram_sample(
        ztU=ztU, zVU=zVU, muU=_muU, L=L)[1][0] if ds.get_phase_diagram_sample(
            ztU=ztU, zVU=zVU, muU=_muU, L=L) is not None else np.ones((L, L)))
                        for _muU in muU]

    inputs = torch.stack(
        [
            net_input_dimless_const_parameters(
                muU=get_quadratic_mu(
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

    outputs = model_predict(model, inputs, batch_size=512)
    figures_axes = defaultdict(lambda: plt.subplots(1, 1, figsize=(6, 6)))
    figures = {"wedding_cake": {}}

    for _muU, qmc_image, nn_outputs in zip(muU, target_densities, outputs):

        fig, ax = figures_axes[f"{_muU}"]
        ax.set_aspect("equal")

        X, Y = np.meshgrid(np.arange(L), np.arange(L))

        nn_image = nn_outputs[0]

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
    model,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 50,
    square_size: int = 22,
):

    path = REPO_DATA_ROOT / f"box/{zVU}/{ztU}/{L}"

    ds = BoseHubbardDataset(
        data_dir=path,
        clean=True,
        observables=[
            "density",
            "density_variance",
            "density_density_corr_0",
            "density_density_corr_1",
            "density_density_corr_2",
            "density_density_corr_3",
            "density_squared",
        ],
        max_density_error=0.015,
        reload=True,
        verbose=False,
    )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities = [(ds.get_phase_diagram_sample(
        ztU=ztU, zVU=zVU, muU=_muU, L=L)[1][0] if ds.get_phase_diagram_sample(
            ztU=ztU, zVU=zVU, muU=_muU, L=L) is not None else np.ones((L, L)))
                        for _muU in muU]

    inputs = torch.stack(
        [
            net_input_dimless_const_parameters(
                muU=get_square_mu(
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

    outputs = model_predict(model, inputs, batch_size=512)

    figures_axes = defaultdict(lambda: plt.subplots(1, 1, figsize=(6, 6)))

    figures = {"box": {}}

    for _muU, qmc_image, nn_outputs in zip(muU, target_densities, outputs):

        fig, ax = figures_axes[f"{_muU}"]
        ax.set_aspect("equal")

        X, Y = np.meshgrid(np.arange(L), np.arange(L))

        nn_image = nn_outputs[0]

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
    model,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 10,
    square_size: int = 22,
):

    ds = BoseHubbardDataset(
        data_dir=REPO_DATA_ROOT / f"box/{zVU}/{ztU}/{L}",
        clean=True,
        observables=[
            "density",
            "density_variance",
            "density_density_corr_0",
            "density_density_corr_1",
            "density_density_corr_2",
            "density_density_corr_3",
            "density_squared",
        ],
        max_density_error=0.015,
        reload=True,
        verbose=False,
    )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities = torch.stack(
        [(ds.get_phase_diagram_sample(ztU=ztU, zVU=zVU, muU=_muU, L=L)[1][0]
          if ds.get_phase_diagram_sample(ztU=ztU, zVU=zVU, muU=_muU, L=L)
          is not None else torch.ones((L, L))) for _muU in muU],
        dim=0,
    )

    cut_position = int(L / 2)

    inputs = torch.stack(
        [
            net_input_dimless_const_parameters(
                muU=get_square_mu(
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
    outputs = model_predict(model, inputs, batch_size=512)

    nn_cuts = outputs[:, 0, cut_position]
    qmc_cuts = target_densities[:, cut_position]

    MU, X = np.meshgrid(muU, np.arange(L))

    qmc_image, nn_image = map(lambda c: np.stack(c, axis=0).T,
                              (qmc_cuts, nn_cuts))

    combined = np.concatenate((
        qmc_image[:int(L / 2) + 1],
        nn_image[int(L / 2) + 1:],
    ))
    # combined[int(L / 2), : int(L / 2) + 1] = nn_image[int(L / 2), : int(L / 2) + 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect(1)

    ax.pcolormesh(MU, X, combined, linewidth=0, rasterized=True)

    plt.hlines(
        y=int(L / 2) + 0.5,
        xmin=X[0, 0],
        xmax=X[int(L / 2), -1],
        color="white",
        lw=2,
    )

    plt.xlabel(r"$\mu$")
    plt.ylabel(r"Cut Site")
    # plt.colorbar()

    plt.xlim(MU[0, 0], MU[0, -1])
    plt.ylim(X[0, 0], X[-1, 0])
    # plt.xticks([1, 2, 3])

    plt.close()

    return {"box_cuts": fig}
