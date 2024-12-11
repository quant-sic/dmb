from collections import defaultdict
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulation
from dmb.paths import REPO_DATA_ROOT

log = getLogger(__name__)

app = typer.Typer()


@app.command()
def plot_qmc_runtimes(simulations_dir: Path,
                      figures_dir: Path = REPO_DATA_ROOT / "figures") -> None:
    """Plot the QMC runtimes."""

    figures_dir.mkdir(exist_ok=True, parents=True)

    all_simulation_directories = list(simulations_dir.glob("*"))

    elapsed_times = defaultdict(list)
    for simulation_dir in tqdm(all_simulation_directories):

        if simulation_dir.is_dir():
            try:
                simulation = WormSimulation.from_dir(simulation_dir)
                elapsed_times[simulation.record["L"]].append(
                    simulation.record["elapsed_time"])
            except (OSError, KeyError, ValueError):
                log.error(f"Could not load simulation from {simulation_dir}")

    mean = np.array([[L, np.mean(times)] for L, times in elapsed_times.items()])
    std = np.array([[L, np.std(times)] for L, times in elapsed_times.items()])

    order = np.argsort(mean[:, 0])
    plt.plot(mean[order, 0], mean[order, 1], ls=(0, (3, 1, 1, 1)), lw=2, marker="o")
    plt.fill_between(mean[order, 0],
                     mean[order, 1] + std[order, 1] * 3,
                     mean[order, 1] - std[order, 1] * 3,
                     alpha=0.2)
