import asyncio
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import typer

from dmb.data.bose_hubbard_2d.worm.simulate import simulate
from dmb.paths import REPO_DATA_ROOT

app = typer.Typer()


@app.command()
def from_potential(
    potential_file_path: Path,
    run_name: str,
    t_hop: float = 1.0,
    ztU: float = 0.25,
    zVU: float = 1.0,
    max_density_error: float = 0.015,
    simulations_dir: Path = REPO_DATA_ROOT
    / "bose_hubbard_2d/from_potential/simulations",
    number_of_concurrent_jobs: int = 1,
) -> None:
    """Run a simulation from a potential file."""

    if not number_of_concurrent_jobs == 1:
        raise ValueError("Only 1 concurrent job is supported.")

    simulations_dir.mkdir(parents=True, exist_ok=True)

    potential = (
        torch.load(potential_file_path, map_location="cpu", weights_only=True)
        .cpu()
        .detach()
        .numpy()
    )
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    U_on = 4.0 * t_hop / ztU

    asyncio.run(
        simulate(
            parent_dir=simulations_dir,
            simulation_name=f"{run_name}_{now}",
            L=potential.shape[0],
            mu=potential,
            t_hop_array=np.full((2, potential.shape[0], potential.shape[1]), t_hop),
            U_on_array=np.full((potential.shape[0], potential.shape[1]), U_on),
            V_nn_array=np.full(
                (2, potential.shape[0], potential.shape[1]), zVU * U_on / 4
            ),
            power=1.0,
            mu_offset=0.0,
        )
    )


if __name__ == "__main__":
    app()
