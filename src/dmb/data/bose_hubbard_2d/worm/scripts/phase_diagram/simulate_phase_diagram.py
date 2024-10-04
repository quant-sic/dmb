import argparse
import datetime
import gc
import shutil
from functools import partial
from pathlib import Path

import joblib

from dmb.data.bose_hubbard_2d.worm.scripts.create_random import \
    draw_random_config, draw_uniform_config
from dmb.data.bose_hubbard_2d.worm.worm.parameters import WormInputParameters
from dmb.data.bose_hubbard_2d.worm.worm.sim import WormSimulation
from dmb.io import ProgressParallel
from dmb.logging import create_logger

log = create_logger(__name__)


def simulate(sample_id, type="random"):
    if type == "random":
        (
            L,
            U_on,
            V_nn,
            mu,
            t_hop_array,
            U_on_array,
            V_nn_array,
            power,
            mu_offset,
        ) = draw_random_config()
    elif type == "uniform":
        (
            L,
            U_on,
            V_nn,
            mu,
            t_hop_array,
            U_on_array,
            V_nn_array,
            power,
            mu_offset,
        ) = draw_uniform_config()
    else:
        raise ValueError(f"Unknown type {type}")

    thermalization = 10000
    sweeps = 100000

    p = WormInputParameters(
        Lx=L,
        Ly=L,
        Nmeasure2=100,
        t_hop=t_hop_array,
        U_on=U_on_array,
        V_nn=V_nn_array,
        thermalization=thermalization,
        mu=mu,
        sweeps=sweeps,
        mu_power=power,
        mu_offset=mu_offset,
    )

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    # save_dir=Path(REPO_ROOT/f"data/bose_hubbard_2d/{now}_sample_{sample_id}")
    save_dir = Path("/ptmp/bale/data/bose_hubbard_2d_phase_diagram/")

    shutil.rmtree(save_dir, ignore_errors=True)

    sim = WormSimulation(p, save_dir=save_dir)

    sim.save_parameters()
    try:
        sim.run_until_convergence(
            executable="/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi")
        sim.plot_result()
    except Exception as e:
        log.error("Simulation failed with exception: %s", e)
        return
    # sim.run_until_convergence(executable="/Users/fabian/paper/worm/build_non_uniform/qmc_worm_mpi")

    finally:
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run worm simulation for 2D BH model")
    parser.add_argument(
        "--number_of_samples_per_dim",
        type=int,
        default=1,
        help="number of samples to run",
    )
    parser.add_argument(
        "--number_of_jobs",
        type=int,
        default=1,
        help="number of jobs to run in parallel",
    )

    args = parser.parse_args()

    # run jobs in parallel
    ProgressParallel(
        n_jobs=args.number_of_jobs,
        total=args.number_of_samples,
        desc="Running Simulations",
        timeout=99999,
    )(joblib.delayed(partial(simulate, type=args.type))(sample_id)
      for sample_id in range(args.number_of_samples))
