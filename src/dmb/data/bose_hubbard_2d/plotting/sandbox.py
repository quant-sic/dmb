from collections import defaultdict
from logging import getLogger

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dmb.data.bose_hubbard_2d.nn_input import get_nn_input_dimless_const_parameters
from dmb.data.bose_hubbard_2d.plotting import PLOT_STYLE, TEXT_WIDTH
from dmb.data.bose_hubbard_2d.potential import (
    get_quadratic_mu_potential,
    get_square_mu_potential,
)
from dmb.data.bose_hubbard_2d.worm.dataset import (
    BoseHubbard2dDataset,
    BoseHubbard2dSampleFilterStrategy,
)
from dmb.model.dmb_model import PredictionMapping
from dmb.paths import REPO_DATA_ROOT

log = getLogger(__name__)


def colorbar(mappable: ScalarMappable) -> Colorbar:
    ax = mappable.axes  # type: ignore
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar: Colorbar = fig.colorbar(mappable, cax=cax)
    return colorbar


def create_wedding_cake_inputs_and_targets(
    bose_hubbard_2d_dataset: BoseHubbard2dDataset | None = None,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 40,
    muU_num_steps: int = 10,
    coefficient: float = -2.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    """Create inputs and targets for the wedding cake plot."""

    if not bose_hubbard_2d_dataset:
        bose_hubbard_2d_dataset = BoseHubbard2dDataset(
            dataset_dir_path=REPO_DATA_ROOT
            / f"bose_hubbard_2d/wedding_cake/{zVU}/{ztU}/{L}/{coefficient}/dataset",
            sample_filter_strategy=BoseHubbard2dSampleFilterStrategy(
                max_density_error=0.1
            ),
        )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities: list[np.ndarray] = [
        (
            bose_hubbard_2d_dataset.get_phase_diagram_sample(
                ztU=ztU, zVU=zVU, muU=_muU, L=L
            ).outputs[0]  # type: ignore
            if bose_hubbard_2d_dataset.get_phase_diagram_sample(
                ztU=ztU, zVU=zVU, muU=_muU, L=L
            )
            is not None
            else np.ones((L, L))
        )
        for _muU in muU
    ]

    inputs = [
        get_nn_input_dimless_const_parameters(
            muU=get_quadratic_mu_potential(
                (coefficient, coefficient),
                L,
                offset=float(_muU),
            ),
            ztU=ztU,
            zVU=zVU,
            cb_projection=True,
            target_density=target_density,
        )
        for _muU, target_density in zip(muU, target_densities)
    ]

    return inputs, target_densities, muU


def create_wedding_cake_plot(
    mapping: PredictionMapping | None = None,
    bose_hubbard_2d_dataset: BoseHubbard2dDataset | None = None,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 40,
    muU_num_steps: int = 10,
    coefficient: float = -2.0,
) -> dict[str, dict[str, plt.Figure]]:
    """Create wedding cake plot for a given model and parameters."""
    inputs, target_densities, muU = create_wedding_cake_inputs_and_targets(
        bose_hubbard_2d_dataset,
        ztU=ztU,
        zVU=zVU,
        muU_min=muU_min,
        muU_max=muU_max,
        L=L,
        muU_num_steps=muU_num_steps,
        coefficient=coefficient,
    )

    outputs = (
        mapping(inputs)
        if mapping is not None
        else {"density": np.zeros_like(target_densities)}
    )
    figures_axes: dict[str, tuple[plt.Figure, plt.Axes]] = defaultdict(
        lambda: plt.subplots(1, 1, figsize=(5, 5))
    )
    figures: dict[str, dict[str, plt.Figure]] = {"wedding_cake": {}}

    for _muU, qmc_image, nn_image in zip(muU, target_densities, outputs["density"]):
        with mpl.rc_context(PLOT_STYLE):
            fig, ax = figures_axes[f"{_muU}"]
            ax.set_aspect("equal")

            X, Y = np.meshgrid(np.arange(L), np.arange(L))

            combined = np.concatenate(
                (
                    qmc_image[: int(L / 2) + 1],
                    nn_image[int(L / 2) + 1 :],
                )
            )

            ax.pcolormesh(
                X,
                Y,
                combined,
                clim=(0, 2),
                cmap="viridis",
                linewidth=0,
                rasterized=True,
            )

            plt.hlines(
                y=int(L / 2) + 0.5,
                xmin=-0.5,
                xmax=L - 0.5,
                color="white",
                lw=2,
            )

            plt.axis("off")

            plt.tight_layout()
            plt.close()

            figures["wedding_cake"]["{:.3}".format(_muU)] = fig

    return figures


def create_box_plot_inputs_and_targets(
    bose_hubbard_2d_dataset: BoseHubbard2dDataset | None = None,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 50,
    square_size: int = 22,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]] | None:
    """Create inputs and targets for the box plot."""
    if not bose_hubbard_2d_dataset:
        bose_hubbard_2d_dataset = BoseHubbard2dDataset(
            dataset_dir_path=REPO_DATA_ROOT
            / f"bose_hubbard_2d/box/{zVU}/{ztU}/{L}/dataset",
            sample_filter_strategy=BoseHubbard2dSampleFilterStrategy(
                max_density_error=0.3
            ),
        )

    if len(bose_hubbard_2d_dataset) == 0:
        return None

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities = [
        (
            bose_hubbard_2d_dataset.get_phase_diagram_sample(
                ztU=ztU, zVU=zVU, muU=_muU, L=L
            ).outputs[0]  # type: ignore
            if bose_hubbard_2d_dataset.get_phase_diagram_sample(
                ztU=ztU, zVU=zVU, muU=_muU, L=L
            )
            is not None
            else np.ones((L, L))
        )
        for _muU in muU
    ]
    inputs = [
        get_nn_input_dimless_const_parameters(
            muU=get_square_mu_potential(
                base_mu=0.0,
                delta_mu=float(_muU),
                square_size=square_size,
                lattice_size=L,
            ),
            ztU=ztU,
            zVU=zVU,
            cb_projection=True,
            target_density=target_density,
        )
        for _muU, target_density in zip(muU, target_densities)
    ]

    return inputs, target_densities, muU


def create_box_plot(
    mapping: PredictionMapping,
    bose_hubbard_2d_dataset: BoseHubbard2dDataset | None = None,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 50,
    square_size: int = 22,
) -> dict[str, dict[str, plt.Figure]]:
    """Create box plot for a given model and parameters."""
    inputs_targets = create_box_plot_inputs_and_targets(
        bose_hubbard_2d_dataset,
        ztU=ztU,
        zVU=zVU,
        muU_min=muU_min,
        muU_max=muU_max,
        L=L,
        muU_num_steps=muU_num_steps,
        square_size=square_size,
    )
    if inputs_targets is None:
        log.warning("No data available for box plot.")
        return {}

    inputs, target_densities, muU = inputs_targets

    outputs = mapping(inputs)

    figures_axes: dict[str, tuple[plt.Figure, plt.Axes]] = defaultdict(
        lambda: plt.subplots(1, 1, figsize=(TEXT_WIDTH * 0.3, TEXT_WIDTH * 0.25))
    )

    figures: dict[str, dict[str, plt.Figure]] = {"box": {}}

    for _muU, qmc_image, nn_image in zip(muU, target_densities, outputs["density"]):
        with mpl.rc_context(PLOT_STYLE):
            fig, ax = figures_axes[f"{_muU}"]
            ax.set_aspect("equal")

            X, Y = np.meshgrid(np.arange(L), np.arange(L))

            combined = np.concatenate(
                (
                    qmc_image[: int(L / 2) + 1],
                    nn_image[int(L / 2) + 1 :],
                )
            )
            combined[int(L / 2), : int(L / 2) + 1] = nn_image[
                int(L / 2), : int(L / 2) + 1
            ]

            ax.pcolormesh(
                X,
                Y,
                combined,
                clim=(0, 2),
                cmap="viridis",
                linewidth=0,
                rasterized=True,
            )

            plt.hlines(
                y=int(L / 2) + 0.5,
                xmin=X[0, 0],
                xmax=X[int(L / 2), -1],
                color="white",
                lw=2,
            )

            # plt.axis("off")
            # colorbar(cm)

            if L == 41:
                # add x,y ticks at 20,40
                ax.set_xticks([20, 40])
                ax.set_yticks([20, 40])

            plt.tight_layout()
            plt.close()

            figures["box"]["{:.3}".format(_muU)] = fig

    return figures


def create_box_cuts_inputs_and_targets(
    bose_hubbard_2d_dataset: BoseHubbard2dDataset | None = None,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 50,
    square_size: int = 22,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], int]:
    if not bose_hubbard_2d_dataset:
        bose_hubbard_2d_dataset = BoseHubbard2dDataset(
            dataset_dir_path=REPO_DATA_ROOT
            / f"bose_hubbard_2d/box/{zVU}/{ztU}/{L}/dataset",
            sample_filter_strategy=BoseHubbard2dSampleFilterStrategy(
                max_density_error=0.3
            ),
        )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    target_densities_unflipped = torch.stack(
        [
            (
                bose_hubbard_2d_dataset.get_phase_diagram_sample(
                    ztU=ztU, zVU=zVU, muU=_muU, L=L
                ).outputs[0]  # type: ignore
                if bose_hubbard_2d_dataset.get_phase_diagram_sample(
                    ztU=ztU, zVU=zVU, muU=_muU, L=L
                )
                is not None
                else torch.ones((L, L))
            )
            for _muU in muU
        ],
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

    inputs = [
        get_nn_input_dimless_const_parameters(
            muU=get_square_mu_potential(
                base_mu=0.0,
                delta_mu=float(_muU),
                square_size=square_size,
                lattice_size=L,
            ),
            ztU=ztU,
            zVU=zVU,
            cb_projection=True,
            target_density=target_density,
        )
        for _muU, target_density in zip(muU, target_densities)
    ]

    return inputs, target_densities, muU, cut_position


def create_box_cuts_plot(
    mapping: PredictionMapping,
    bose_hubbard_2d_dataset: BoseHubbard2dDataset | None = None,
    ztU: float = 0.1,
    zVU: float = 1.0,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 41,
    muU_num_steps: int = 50,
    square_size: int = 22,
) -> dict[str, plt.Figure]:
    """Create box cuts plot for a given model and parameters."""
    inputs, target_densities, muU, cut_position = create_box_cuts_inputs_and_targets(
        bose_hubbard_2d_dataset,
        ztU=ztU,
        zVU=zVU,
        muU_min=muU_min,
        muU_max=muU_max,
        L=L,
        muU_num_steps=muU_num_steps,
        square_size=square_size,
    )
    _target_densities = torch.stack(target_densities, dim=0)
    outputs = mapping(inputs)

    nn_cuts: np.ndarray = outputs["density"][:, cut_position]
    qmc_cuts: torch.Tensor = _target_densities[:, cut_position]

    MU, X = np.meshgrid(muU[2:], np.arange(L))

    qmc_image = np.stack(qmc_cuts.cpu().numpy()[2:], axis=0).T  # type: ignore
    nn_image = np.stack(nn_cuts[2:], axis=0).T  # type: ignore

    combined = np.concatenate(
        (
            qmc_image[: int(L / 2) + 1],
            nn_image[int(L / 2) + 1 :],
        )
    )
    # combined[int(L / 2), : int(L / 2) + 1] = nn_image[int(L / 2), : int(L / 2) + 1]

    with mpl.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(TEXT_WIDTH * 0.3, TEXT_WIDTH * 0.25))

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
        cbar = plt.colorbar(mappable)
        cbar.set_ticks([0, 1, 2])

        plt.xlim(MU[0, 0], MU[0, -1])
        plt.ylim(X[0, 0], X[-1, 0])
        plt.xticks([1, 2, 3])

        plt.tight_layout()
        plt.close()

    return {"box_cuts": fig}


def plot_phase_diagram_mu_cut(
    mapping: PredictionMapping,
    bose_hubbard_2d_dataset: BoseHubbard2dDataset | None = None,
    zVU: float = 1.0,
    ztU: float = 0.25,
    muU_min: float = 0.0,
    muU_max: float = 3.0,
    L: int = 16,
    muU_num_steps: int = 50,
) -> dict[str, plt.Figure]:
    """Plot the phase diagram of the Bose-Hubbard model for a given mu cut."""

    if not bose_hubbard_2d_dataset:
        bose_hubbard_2d_dataset = BoseHubbard2dDataset(
            dataset_dir_path=REPO_DATA_ROOT
            / "bose_hubbard_2d/mu_cut"
            / f"{zVU}/{ztU}/{L}/dataset",
            sample_filter_strategy=BoseHubbard2dSampleFilterStrategy(
                max_density_error=0.015
            ),
        )

    muU = np.linspace(muU_min, muU_max, muU_num_steps)
    inputs = [
        get_nn_input_dimless_const_parameters(
            muU=np.full((L, L), fill_value=_muU),
            ztU=ztU,
            zVU=zVU,
            cb_projection=True,
            target_density=np.ones((L, L)),
        )
        for _muU in muU
    ]
    outputs: dict[str, np.ndarray] = mapping(inputs)

    with mpl.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(TEXT_WIDTH * 0.25, TEXT_WIDTH * 0.2))

        try:
            muU_qmc, n_qmc = zip(
                *[  # type: ignore
                    (
                        bose_hubbard_2d_dataset.get_phase_diagram_position(i)[1],
                        bose_hubbard_2d_dataset_i.outputs[0],
                    )
                    for i, bose_hubbard_2d_dataset_i in enumerate(
                        bose_hubbard_2d_dataset  # type: ignore
                    )
                ]
            )

            ax.scatter(
                muU_qmc,
                [n_qmc[i].max() for i in range(len(n_qmc))],
                s=20,
                c="gold",
                label=r"$QMC_{max}$",
            )
            ax.scatter(
                muU_qmc,
                [n_qmc[i].min() for i in range(len(n_qmc))],
                s=20,
                c="darkorange",
                label=r"$QMC_{min}$",
            )
        except ValueError:
            pass

        # max
        ax.scatter(
            muU,
            outputs["density"].max(axis=(-1, -2)),
            s=20,
            c="darkblue",
            label=r"$NN_{max}$",
        )

        ax.scatter(
            muU,
            outputs["density"].min(axis=(-1, -2)),
            s=20,
            c="cornflowerblue",
            label=r"$NN_{min}$",
        )

        plt.xticks([0, 1, 2])
        plt.yticks([1, 2])

        # plt.legend()
        plt.xlabel(r"$\mu/U$")
        plt.ylabel(r"$n$")
        plt.tight_layout()
        plt.close()

    return {"mu_cut": fig}
