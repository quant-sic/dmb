import argparse
import asyncio

import numpy as np

from dmb.data.bose_hubbard_2d.potential import get_square_mu_potential
from dmb.data.bose_hubbard_2d.worm.scripts.simulate import \
    get_missing_samples, simulate
from dmb.paths import REPO_DATA_ROOT

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run worm simulation for 2D BH model")
    parser.add_argument(
        "--muU_offset",
        type=float,
        default=0.0,
        help="mu/U offset",
    )
    parser.add_argument(
        "--muU_delta_min",
        type=float,
        default=0.0,
        help="minimum mu/U delta",
    )
    parser.add_argument(
        "--muU_delta_max",
        type=float,
        default=3.0,
        help="maximum mu/U delta",
    )
    parser.add_argument(
        "--muU_delta_num_steps",
        type=int,
        default=50,
        help="Number of mu/U delta steps",
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
        default=41,
        help="lattice size",
    )
    parser.add_argument(
        "--number_of_concurrent_jobs",
        type=int,
        default=1,
        help="number of concurrent jobs",
    )
    parser.add_argument(
        "--max_density_error",
        type=float,
        default=0.015,
        help="max density error",
    )
    parser.add_argument(
        "--tau_max_threshold",
        type=int,
        default=10,
        help="tau max threshold",
    )

    args = parser.parse_args()

    target_dir = (
        REPO_DATA_ROOT /
        f"simulation/bose_hubbard_2d/box/{args.zVU}/{args.ztU}/{args.L}")
    dataset_dir = (
        REPO_DATA_ROOT /
        f"datasets/bose_hubbard_2d/box/{args.zVU}/{args.ztU}/{args.L}")
    target_dir.mkdir(parents=True, exist_ok=True)

    L_out, ztU_out, zVU_out, muU_out = get_missing_samples(
        dataset_dir=dataset_dir,
        L=args.L,
        ztU=args.ztU,
        zVU=args.zVU,
        muU=list(
            np.linspace(args.muU_delta_min, args.muU_delta_max,
                        args.muU_delta_num_steps)),
        tolerance_ztU=0,
        tolerance_zVU=0,
        tolerance_muU=0,
    )

    semaphore = asyncio.Semaphore(args.number_of_concurrent_jobs)

    async def run_sample(sample_id):
        U_on = 4 / args.ztU
        async with semaphore:
            await simulate(
                parent_dir=target_dir,
                simulation_name="box_{}_{:.3f}_{}".format(
                    args.zVU, muU_out[sample_id], sample_id),
                L=args.L,
                mu=get_square_mu_potential(
                    base_mu=0.0,
                    delta_mu=muU_out[sample_id],
                    square_size=22,
                    lattice_size=args.L,
                ) * U_on,
                t_hop_array=np.ones((2, args.L, args.L)),
                U_on_array=np.ones((args.L, args.L)) * U_on,
                V_nn_array=np.ones((2, args.L, args.L)) * args.zVU * U_on / 4,
                power=1.0,
                mu_offset=muU_out[sample_id] * U_on,
                tune_tau_max_threshold=args.tau_max_threshold,
                run_max_density_error=args.max_density_error,
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        asyncio.gather(
            *[run_sample(sample_id) for sample_id in range(len(muU_out))]))
    loop.close()
