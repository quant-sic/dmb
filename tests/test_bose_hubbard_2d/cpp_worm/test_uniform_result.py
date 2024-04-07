import itertools
import os
from pathlib import Path

import numpy as np

from dmb.data.bose_hubbard_2d.worm_qmc.worm.parameters import WormInputParameters
from dmb.data.bose_hubbard_2d.worm_qmc.worm.run import WormSimulationRunner
from dmb.data.bose_hubbard_2d.worm_qmc.worm.sim import WormSimulation

# test function itself


def check_mean_density_validity_assuming_uniform_input(
    mean_density: np.ndarray, num_unique_values: int = 2
):
    if not ((mean_density >= 0).all() and (mean_density <= 1).all()):
        raise ValueError(
            f"mean_density must be between 0 and 1. mean_density: {mean_density}"
        )

    unique_values = []
    for value in mean_density:
        if not any(np.isclose(value, unique_values, atol=0.02)):
            unique_values.append(value)

    if not (len(unique_values) <= num_unique_values):
        raise ValueError(
            f"mean_density must have at most two unique values. mean_density: {mean_density}"
        )


def test_check_mean_density_validity_assuming_uniform_input(tmp_path: Path) -> None:
    for L, muU, ztU in itertools.product(
        [4, 6, 8], [0.5, 1.0, 1.5, 2.0, 2.5], [0.1, 0.4]
    ):
        U_on = np.full(shape=(L, L), fill_value=4 / ztU)

        mu = np.ones((L, L)) * (muU * U_on)

        V_nn = np.expand_dims(U_on / 4, axis=0).repeat(2, axis=0)
        t_hop = np.ones((2, L, L))

        tmp_path = Path("./test")

        p = WormInputParameters(
            Lx=L,
            Ly=L,
            Nmeasure2=1,
            t_hop=t_hop,
            U_on=U_on,
            V_nn=V_nn,
            thermalization=1,
            mu=mu,
            sweeps=1,
        )

        sim = WormSimulation(
            input_parameters=p,
            save_dir=tmp_path / "test_sim",
            worm_executable=os.environ["WORM_MPI_EXECUTABLE"],
        )
        sim_runner = WormSimulationRunner(sim)

        sim_runner.tune_nmeasure2_sync()
        sim_runner.run_iterative_until_converged_sync()

        mean_density = sim.output.densities.mean(axis=0)

        # check that this function does not raise an exception
        check_mean_density_validity_assuming_uniform_input(mean_density)
