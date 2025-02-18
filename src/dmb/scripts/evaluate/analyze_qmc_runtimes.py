from collections import defaultdict
from logging import getLogger
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulation
from dmb.paths import REPO_DATA_ROOT

log = getLogger(__name__)

app = typer.Typer()


@app.command()
def plot_qmc_runtimes(
    simulations_dir: Path = REPO_DATA_ROOT / "bose_hubbard_2d/random/simulations",
    figures_dir: Path = REPO_DATA_ROOT / "figures",
    max_num_simulations: int = 1000,
) -> None:
    """Plot the QMC runtimes."""

    figures_dir.mkdir(exist_ok=True, parents=True)

    all_simulation_directories = list(simulations_dir.glob("*"))

    elapsed_times = defaultdict(list)
    for simulation_dir in tqdm(all_simulation_directories[:max_num_simulations]):
        if simulation_dir.is_dir():
            try:
                simulation = WormSimulation.from_dir(simulation_dir)
                elapsed_times[simulation.input_parameters.Lx].append(
                    sum(step["elapsed_time"] for step in simulation.record["steps"])
                )
            except (OSError, KeyError, ValueError):
                pass

    # violin plot
    # remove large outliers (>95%)
    most_elapsed_times = {}
    for L, times in elapsed_times.items():
        most_elapsed_times[L] = [
            time for time in times if time <= np.percentile(times, 95)
        ]

    matplotlib.rcParams.update({"font.size": 20})
    matplotlib.rcParams.update({"figure.autolayout": True})
    # use latex
    matplotlib.rcParams["text.usetex"] = True

    fig, ax = plt.subplots(figsize=(3, 3))

    # plot means
    sns.violinplot(
        x=[L for L in most_elapsed_times.keys() for _ in most_elapsed_times[L]],
        y=[
            time / 3600
            for L in most_elapsed_times.keys()
            for time in most_elapsed_times[L]
        ],
        inner=None,
        linewidth=0,
        color="lightblue",
        ax=ax,
        density_norm="count",
    )

    # plot means
    plt.scatter(
        np.arange(len(most_elapsed_times)),
        [np.mean(most_elapsed_times[L]) / 3600 for L in sorted(most_elapsed_times)],
        label="mean",
        zorder=10,
    )

    plt.ylim(
        0,
        max(
            [
                time / 3600
                for L in most_elapsed_times.keys()
                for time in most_elapsed_times[L]
            ]
        )
        * 1.1,
    )

    # yticks only 5, 15
    plt.yticks([5, 15])

    plt.xticks(
        range(len(most_elapsed_times)),
        [f"${L}$" if L in [10, 18] else "" for L in sorted(most_elapsed_times)],
    )

    # plt.xlabel("L")
    # plt.ylabel("Runtime [h]")

    plt.savefig(figures_dir / "qmc_runtimes_violin.pdf")


if __name__ == "__main__":
    app()
