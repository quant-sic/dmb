import asyncio
import os

import numpy as np
import typer
from dotenv import load_dotenv

from dmb.data.bose_hubbard_2d.potential import get_quadratic_mu_potential
from dmb.data.bose_hubbard_2d.worm.simulate import get_missing_samples, simulate
from dmb.paths import REPO_DATA_ROOT

app = typer.Typer()


@app.command(help="Run worm simulation for 2D BH model")
def simulate_wedding_cake(
    muU_min: float = typer.Option(0.0, help="Minimum mu offset"),
    muU_max: float = typer.Option(3.0, help="Maximum mu offset"),
    muU_num_steps: int = typer.Option(10, help="Number of mu offset steps"),
    coefficient: float = typer.Option(-2.0, help="Quadratic mu coefficients"),
    ztU: float = typer.Option(0.1, help="Nearest neighbor interaction"),
    zVU: float = typer.Option(1.0, help="Nearest neighbor interaction"),
    L: int = typer.Option(40, help="Lattice size"),
    number_of_concurrent_jobs: int = typer.Option(1, help="Number of concurrent jobs"),
) -> None:
    load_dotenv()

    os.environ["WORM_JOB_NAME"] = "wedding_cake"

    target_dir = (
        REPO_DATA_ROOT
        / f"bose_hubbard_2d/wedding_cake/{zVU}/{ztU}/{L}/{coefficient}/simulations"
    )
    dataset_dir = (
        REPO_DATA_ROOT
        / f"bose_hubbard_2d/wedding_cake/{zVU}/{ztU}/{L}/{coefficient}/dataset"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    L_out, ztU_out, zVU_out, muU_out = get_missing_samples(
        dataset_dir=dataset_dir,
        L=[L] * muU_num_steps,
        ztU=[ztU] * muU_num_steps,
        zVU=[zVU] * muU_num_steps,
        muU=list(np.linspace(muU_min, muU_max, muU_num_steps)),
        tolerance_ztU=0,
        tolerance_zVU=0,
        tolerance_muU=0,
    )

    semaphore = asyncio.Semaphore(number_of_concurrent_jobs)

    U_on = 4 / ztU

    async def run_sample(sample_id: int) -> None:
        async with semaphore:
            await simulate(
                parent_dir=target_dir,
                simulation_name="wedding_cake_{}_{:.3f}_{}".format(
                    zVU, muU_out[sample_id], sample_id
                ),
                L=L,
                mu=get_quadratic_mu_potential(
                    (coefficient, coefficient),
                    L,
                    offset=muU_out[sample_id],
                )
                * U_on,
                t_hop_array=np.ones((2, L, L)),
                U_on_array=np.ones((L, L)) * U_on,
                V_nn_array=np.ones((2, L, L)) * zVU * U_on / 4,
                power=1.0,
                mu_offset=muU_out[sample_id] * U_on,
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        asyncio.gather(*[run_sample(sample_id) for sample_id in range(len(muU_out))])
    )
    loop.close()


if __name__ == "__main__":
    app()
