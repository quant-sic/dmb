import argparse
import asyncio
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.cpp_worm.dataset import BoseHubbardDataset
from dmb.data.bose_hubbard_2d.cpp_worm.scripts.simulate import simulate
from dmb.utils import REPO_DATA_ROOT


def get_required_inputs(
    target_dir: Path,
    L: int,
    U_on_inv_min: float,
    U_on_inv_max: float,
    zVU: float,
    muU_min: float,
    muU_max: float,
    min_distance_between_samples_U_inv: float,
    min_distance_between_samples_mu: float,
):
    bh_dataset = BoseHubbardDataset(
        data_dir=target_dir,
        observables=["density"],
        clean=True,
        reload=True,
        verbose=False,
        max_density_error=0.015,
        include_tune_dirs=False,
    )
    # bh_dataset.reload_samples()

    if len(bh_dataset) == 0:
        zVU, muU, ztU = [], [], []
    else:
        zVU, muU, ztU = zip(*[
            bh_dataset.phase_diagram_position(idx)
            for idx in tqdm(range(len(bh_dataset)))
        ])

    rest_filter = [
        bh_dataset.get_parameters(idx).Lx == L and zVU[idx] == zVU
        for idx in range(len(bh_dataset))
    ]
    zVU, muU, ztU = map(
        lambda x: np.array(x)[rest_filter],
        (zVU, muU, ztU),
    )

    # get meshgrid of U and mu
    U_on_array = np.arange(U_on_inv_min, U_on_inv_max,
                           min_distance_between_samples_U_inv)
    mu_offset_array = np.arange(muU_min, muU_max,
                                min_distance_between_samples_mu)

    U_on_inv_mesh, muU_mesh = np.meshgrid(U_on_array, mu_offset_array)

    # mask out points that are already close to existing samples. ie absolute difference is less than min_distance_between_samples
    mask = np.full(shape=U_on_inv_mesh.shape, fill_value=True)
    for muU_, ztU_ in zip(muU, ztU):
        mask = np.logical_and(
            mask,
            np.logical_or(
                np.abs(U_on_inv_mesh - ztU_)
                > min_distance_between_samples_U_inv,
                np.abs(muU_mesh - muU_) > min_distance_between_samples_mu,
            ),
        )

    # get the remaining points
    U_on_inv_required = U_on_inv_mesh[mask]
    muU_required = muU_mesh[mask]

    U_on_required = 4 / U_on_inv_required
    mu_required = muU_required * U_on_required

    return U_on_required, mu_required


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run worm simulation for 2D BH model")
    parser.add_argument(
        "--min_distance_between_samples_U_inv",
        type=float,
        default=0.01,
        help="minimum distance between samples",
    )
    parser.add_argument(
        "--min_distance_between_samples_mu",
        type=float,
        default=0.1,
        help="minimum distance between samples",
    )
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
        "--min_U_on_inv",
        type=float,
        default=0.05,
        help="minimum U_on",
    )
    parser.add_argument(
        "--max_U_on_inv",
        type=float,
        default=1.0,
        help="maximum U_on",
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
        default=16,
        help="lattice size",
    )
    parser.add_argument(
        "--number_of_concurrent_jobs",
        type=int,
        default=1,
        help="number of concurrent jobs",
    )

    args = parser.parse_args()

    os.environ["WORM_JOB_NAME"] = "phase_diagram"

    target_dir = REPO_DATA_ROOT / f"phase_diagram/{args.V_nn_z}/{args.L}"
    target_dir.mkdir(parents=True, exist_ok=True)

    U_on_required, mu_required = get_required_inputs(
        target_dir=target_dir,
        L=args.L,
        U_on_inv_min=args.min_U_on_inv,
        U_on_inv_max=args.max_U_on_inv,
        V_nn_z=args.V_nn_z,
        muU_min=args.muU_min,
        muU_max=args.muU_max,
        min_distance_between_samples_U_inv=args.
        min_distance_between_samples_U_inv,
        min_distance_between_samples_mu=args.min_distance_between_samples_mu,
    )

    semaphore = asyncio.Semaphore(args.number_of_concurrent_jobs)

    async def run_sample(sample_id):
        async with semaphore:
            await simulate(
                parent_dir=target_dir,
                simulation_name=
                f"phase_diagram_{args.zVU}_{args.L}_{sample_id}",
                L=args.L,
                U_on=U_on_required[sample_id],
                V_nn=args.zVU * U_on_required[sample_id] / 4,
                mu=np.ones((args.L, args.L)) * mu_required[sample_id],
                t_hop_array=np.ones((2, args.L, args.L)),
                U_on_array=np.ones(
                    (args.L, args.L)) * U_on_required[sample_id],
                V_nn_array=np.ones((2, args.L, args.L)) * args.V_nn_z,
                power=1.0,
                mu_offset=mu_required[sample_id],
            )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.gather(
            *
            [run_sample(sample_id)
             for sample_id in range(len(U_on_required))]))
    loop.close()
