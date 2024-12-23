import argparse
import asyncio
import os

import numpy as np
from dotenv import load_dotenv

from dmb.data.bose_hubbard_2d.potential import get_quadratic_mu_potential
from dmb.data.bose_hubbard_2d.worm.scripts.simulate import \
    get_missing_samples, simulate
from dmb.paths import REPO_DATA_ROOT

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run worm simulation for 2D BH model")
    parser.add_argument(
        "--muU_min",
        type=float,
        default=0.0,
        help="minimum mu offset",
    )
    parser.add_argument(
        "--muU_max",
        type=float,
        default=3.0,
        help="maximum mu offset",
    )
    parser.add_argument(
        "--muU_num_steps",
        type=int,
        default=10,
        help="number of mu offset steps",
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        default=-2.0,
        help="quadratic mu coefficients",
    )
    parser.add_argument(
        "--ztU",
        type=float,
        default=0.1,
        help="nearest neighbor interaction",
    )
    parser.add_argument(
        "--zVU",
        type=float,
        default=1.0,
        help="nearest neighbor interaction",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=40,
        help="lattice size",
    )
    parser.add_argument(
        "--number_of_concurrent_jobs",
        type=int,
        default=1,
        help="number of concurrent jobs",
    )

    args = parser.parse_args()

    os.environ["WORM_JOB_NAME"] = "wedding_cake"

    target_dir = (
        REPO_DATA_ROOT /
        f"simulation/bose_hubbard_2d/wedding_cake/{args.zVU}/{args.ztU}/{args.L}/{args.coefficient}"
    )
    dataset_dir = (
        REPO_DATA_ROOT /
        f"datasets/bose_hubbard_2d/wedding_cake/{args.zVU}/{args.ztU}/{args.L}/{args.coefficient}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    L_out, ztU_out, zVU_out, muU_out = get_missing_samples(
        dataset_dir=dataset_dir,
        L=args.L,
        ztU=args.ztU,
        zVU=args.zVU,
        muU=list(np.linspace(args.muU_min, args.muU_max, args.muU_num_steps)),
        tolerance_ztU=0,
        tolerance_zVU=0,
        tolerance_muU=0,
    )

    semaphore = asyncio.Semaphore(args.number_of_concurrent_jobs)

    async def run_sample(sample_id: int) -> None:
        U_on = 4 / args.ztU
        async with semaphore:
            await simulate(
                parent_dir=target_dir,
                simulation_name="wedding_cake_{}_{:.3f}_{}".format(
                    args.zVU, muU_out[sample_id], sample_id),
                L=args.L,
                mu=get_quadratic_mu_potential(
                    (args.coefficient, args.coefficient),
                    args.L,
                    offset=muU_out[sample_id],
                ) * U_on,
                t_hop_array=np.ones((2, args.L, args.L)),
                U_on_array=np.ones((args.L, args.L)) * U_on,
                V_nn_array=np.ones((2, args.L, args.L)) * args.zVU * U_on / 4,
                power=1.0,
                mu_offset=muU_out[sample_id] * U_on,
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        asyncio.gather(*[run_sample(sample_id) for sample_id in range(len(muU_out))]))
    loop.close()
