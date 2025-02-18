from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

from dmb.data.bose_hubbard_2d.plotting.phase_diagram import plot_phase_diagram
from dmb.data.bose_hubbard_2d.plotting.sandbox import (
    create_box_cuts_plot,
    create_box_plot,
    create_wedding_cake_plot,
    plot_phase_diagram_mu_cut,
)
from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulation
from dmb.model.dmb_model import PredictionMapping
from dmb.model.lit_dmb_model import LitDMBModel
from dmb.paths import REPO_DATA_ROOT, REPO_LOGS_ROOT

app = typer.Typer()


@cache
def load_model(log_dir: Path, checkpoint: Path | None = None):
    if checkpoint is None:
        # get latest step in log_dir/checkpoints/best
        checkpoint = sorted(
            (log_dir / "checkpoints/best_train").glob("*.ckpt"),
            key=lambda x: int(x.stem.split("=")[-1]),
        )[-1]

    model = LitDMBModel.load_from_logged_checkpoint(log_dir, checkpoint)

    return model


@app.command()
def plot_wedding_cake(
    log_dir: Path = typer.Option(
        default=REPO_LOGS_ROOT
        / "train/bose_hubbard_2d/worm/all_obs/se_resnet/mse/softplus/runs/2025-01-03_13-19-37",
        help="Path to the log directory",
    ),
    checkpoint: Path = typer.Option(default=None, help="Path to the checkpoint file"),
) -> None:
    model = load_model(log_dir, checkpoint)
    output = create_wedding_cake_plot(mapping=PredictionMapping(model=model.model))

    figures_dir = REPO_DATA_ROOT / "figures" / log_dir.name / "wedding_cake"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for key, value in output["wedding_cake"].items():
        value.savefig(
            figures_dir / f"{key}.pdf", transparent=True, bbox_inches="tight", dpi=600
        )


@app.command()
def plot_box_cuts(
    log_dir: Path = typer.Option(
        default=REPO_LOGS_ROOT
        / "train/bose_hubbard_2d/worm/all_obs/se_resnet/mse/softplus/runs/2025-01-03_13-19-37",
        help="Path to the log directory",
    ),
    checkpoint: Path = typer.Option(default=None, help="Path to the checkpoint file"),
) -> None:
    model = load_model(log_dir, checkpoint)
    output = create_box_cuts_plot(mapping=PredictionMapping(model=model.model))

    figures_dir = REPO_DATA_ROOT / "figures" / log_dir.name
    figures_dir.mkdir(parents=True, exist_ok=True)
    output["box_cuts"].savefig(
        figures_dir / "box_cuts.pdf", transparent=True, bbox_inches="tight", dpi=600
    )


@app.command()
def plot_box(
    log_dir: Path = typer.Option(
        default=REPO_LOGS_ROOT
        / "train/bose_hubbard_2d/worm/all_obs/se_resnet/mse/softplus/runs/2025-01-03_13-19-37",
        help="Path to the log directory",
    ),
    checkpoint: Path = typer.Option(default=None, help="Path to the checkpoint file"),
) -> None:
    model = load_model(log_dir, checkpoint)
    output = create_box_plot(mapping=PredictionMapping(model=model.model))

    figures_dir = REPO_DATA_ROOT / "figures" / log_dir.name / "box"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for key, value in output["box"].items():
        value.savefig(
            figures_dir / f"{key}.pdf", transparent=True, bbox_inches="tight", dpi=600
        )


@app.command()
def plot_phase_diagram_for_model(
    log_dir: Path = typer.Option(
        default=REPO_LOGS_ROOT
        / "train/bose_hubbard_2d/worm/all_obs/se_resnet/mse/softplus/runs/2025-01-03_13-19-37",
        help="Path to the log directory",
    ),
    checkpoint: Path = typer.Option(default=None, help="Path to the checkpoint file"),
) -> None:
    model = load_model(log_dir, checkpoint)

    for zVU in (1.0, 1.5):
        figures = plot_phase_diagram(
            mapping=PredictionMapping(model=model.model), zVU=zVU
        )

        for key, value in figures.items():
            figures_dir = (
                REPO_DATA_ROOT
                / "figures"
                / log_dir.name
                / "phase_diagram"
                / f"zVU={zVU}"
                / key
            )
            figures_dir.mkdir(parents=True, exist_ok=True)

            for title, fig in value.items():
                fig.savefig(
                    figures_dir / f"{title}.pdf",
                    transparent=True,
                    bbox_inches="tight",
                    dpi=600,
                )


@app.command()
def plot_phase_diagram_mu_cut_for_model(
    log_dir: Path = typer.Option(
        default=REPO_LOGS_ROOT
        / "train/bose_hubbard_2d/worm/all_obs/se_resnet/mse/softplus/runs/2025-01-03_13-19-37",
        help="Path to the log directory",
    ),
    checkpoint: Path = typer.Option(default=None, help="Path to the checkpoint file"),
) -> None:
    model = load_model(log_dir, checkpoint)

    for zVU in (1.0, 1.5):
        for ztU in (0.1, 0.25):
            figures = plot_phase_diagram_mu_cut(
                mapping=PredictionMapping(model=model.model), zVU=zVU, ztU=ztU
            )
            figures_dir = (
                REPO_DATA_ROOT
                / "figures"
                / log_dir.name
                / "phase_diagram_mu_cut"
                / f"zVU={zVU}/ztU={ztU}"
            )
            figures_dir.mkdir(parents=True, exist_ok=True)

            for title, fig in figures.items():
                fig.savefig(
                    figures_dir / f"{title}.pdf",
                    transparent=True,
                    bbox_inches="tight",
                    dpi=600,
                )


@app.command()
def plot_all(
    log_dir: Path = typer.Option(
        default=REPO_LOGS_ROOT
        / "train/bose_hubbard_2d/worm/all_obs/se_resnet/mse/softplus/runs/2025-01-03_13-19-37",
        help="Path to the log directory",
    ),
    checkpoint: Path = typer.Option(default=None, help="Path to the checkpoint file"),
) -> None:
    plot_wedding_cake(log_dir, checkpoint)
    plot_box_cuts(log_dir, checkpoint)
    plot_box(log_dir, checkpoint)
    plot_phase_diagram_mu_cut_for_model(log_dir, checkpoint)
    plot_phase_diagram_for_model(log_dir, checkpoint)


@app.command()
def plot_inversion(
    inversion_log_dir: Path = typer.Option(
        default=REPO_LOGS_ROOT
        / "invert/minerva/runs/2025-02-18_22-28-47/minerva_2025-02-18_22-28-47/version_0",
        help="Path to the log directory",
    ),
    qmc_simulation_dir: Path = typer.Option(
        default=REPO_DATA_ROOT
        / "bose_hubbard_2d/from_potential/simulations/2025-01-19_22-47-26_sample_8273749_minerva_test_2025-01-19-22-47-26/tune",
        help="Path to the qmc simulation directory",
    ),
) -> None:
    target = torch.load(inversion_log_dir / "results/target.pt", weights_only=False)
    inverted = torch.load(inversion_log_dir / "results/inverted.pt", weights_only=False)
    remapped = torch.load(inversion_log_dir / "results/remapped.pt", weights_only=False)

    qmc_simulation_result = WormSimulation.from_dir(
        qmc_simulation_dir
    ).observables.get_error_analysis("primary", "density")["expectation_value"]

    figures_dir = REPO_DATA_ROOT / "figures/inversion" / inversion_log_dir.parent.name
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, plottable in (
        ("target", target),
        ("inverted", inverted),
        ("remapped", remapped),
        ("qmc_simulation_result", qmc_simulation_result),
    ):
        plottable = plottable.squeeze()
        plt.figure(figsize=(5, 5))
        plt.imshow(plottable)
        plt.axis("off")

        plt.clim(-1.5, 5.0)

        plt.tight_layout()
        plt.savefig(
            figures_dir / f"{name}.pdf", transparent=True, bbox_inches="tight", dpi=600
        )


if __name__ == "__main__":
    app()
