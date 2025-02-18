import argparse
import asyncio

import numpy as np
import typer

from dmb.data.bose_hubbard_2d.potential import get_square_mu_potential
from dmb.data.bose_hubbard_2d.worm.simulate import get_missing_samples, \
    simulate
from dmb.paths import REPO_DATA_ROOT

app = typer.Typer()


@app.command(help="Run worm simulation for 2D BH model")
def simulate_box(
    muU_offset: float = typer.Option(0.0, help="mu/U offset"),
    muU_delta_min: float = typer.Option(0.0, help="minimum mu/U delta"),
    muU_delta_max: float = typer.Option(3.0, help="maximum mu/U delta"),
    muU_delta_num_steps: int = typer.Option(50, help="Number of mu/U delta steps"),
    ztU: float = typer.Option(0.1, help="nearest neighbor interaction"),
    zVU: float = typer.Option(1.0, help="nearest neighbor interaction"),
    L: int = typer.Option(41, help="lattice size"),
    number_of_concurrent_jobs: int = typer.Option(1, help="number of concurrent jobs"),
    max_density_error: float = typer.Option(0.015, help="max density error"),
    tau_max_threshold: int = typer.Option(10, help="tau max threshold"),
) -> None:
    target_dir = REPO_DATA_ROOT / f"bose_hubbard_2d/box/{zVU}/{ztU}/{L}/simulations"
    dataset_dir = REPO_DATA_ROOT / f"bose_hubbard_2d/box/{zVU}/{ztU}/{L}/dataset"
    target_dir.mkdir(parents=True, exist_ok=True)

    L_out, ztU_out, zVU_out, muU_out = get_missing_samples(
        dataset_dir=dataset_dir,
        L=[L] * muU_delta_num_steps,
        ztU=[ztU] * muU_delta_num_steps,
        zVU=[zVU] * muU_delta_num_steps,
        muU=list(np.linspace(muU_delta_min, muU_delta_max, muU_delta_num_steps)),
        tolerance_ztU=0,
        tolerance_zVU=0,
        tolerance_muU=0,
        max_density_error=max_density_error,
    )

    semaphore = asyncio.Semaphore(number_of_concurrent_jobs)

    async def run_sample(sample_id: int) -> None:
        U_on = 4 / ztU
        async with semaphore:
            await simulate(
                parent_dir=target_dir,
                simulation_name="box_{}_{:.3f}_{}".format(
                    zVU, muU_out[sample_id], sample_id
                ),
                L=L,
                mu=get_square_mu_potential(
                    base_mu=0.0,
                    delta_mu=muU_out[sample_id],
                    square_size=22,
                    lattice_size=L,
                )
                * U_on,
                t_hop_array=np.ones((2, L, L)),
                U_on_array=np.ones((L, L)) * U_on,
                V_nn_array=np.ones((2, L, L)) * zVU * U_on / 4,
                power=1.0,
                mu_offset=muU_out[sample_id] * U_on,
                tune_tau_max_threshold=tau_max_threshold,
                run_max_density_error=max_density_error,
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        asyncio.gather(*[run_sample(sample_id) for sample_id in range(len(muU_out))])
    )
    loop.close()


if __name__ == "__main__":
    app()
