import asyncio
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.cpp_worm.worm.sim import WormSimulation
from dmb.utils import create_logger

log = create_logger(__name__)


class WormSimulationRunner:
    def __init__(self, worm_simulation: WormSimulation):
        self.worm_simulation = worm_simulation

    async def run(self, num_restarts: int = 1):
        self.worm_simulation.save_parameters()
        try:
            await self.worm_simulation.execute_worm(num_restarts=num_restarts)
        except RuntimeError as e:
            log.error(e)
            raise e

    async def run_continue(self, num_restarts: int = 1):
        try:
            await self.worm_simulation.execute_worm_continue(num_restarts=num_restarts)
        except RuntimeError as e:
            log.error(e)
            raise e

    async def run_iterative_until_converged(
        self,
        max_num_measurements_per_nmeasure2: int = 150000,
        min_num_measurements_per_nmeasure2: int = 1000,
        num_sweep_increments: int = 25,
        sweeps_to_thermalization_ratio: int = 10,
        max_abs_error_threshold: int = 0.015,
        num_restarts: int = 1,
    ) -> None:
        try:
            expected_required_num_measurements = (
                2
                * (
                    self.worm_simulation.tune_simulation.uncorrected_max_density_error
                    / max_abs_error_threshold
                )
                ** 2
                * self.worm_simulation.tune_simulation.max_tau_int
            )
        except KeyError:
            log.info(
                "No tune simulation found. Will not be able to estimate required number of measurements."
            )
            expected_required_num_measurements = None

        num_sweeps_values = np.array(
            [
                (
                    min_num_measurements_per_nmeasure2
                    + i
                    * int(
                        (
                            max_num_measurements_per_nmeasure2
                            - min_num_measurements_per_nmeasure2
                        )
                        / num_sweep_increments
                    )
                )
                * self.worm_simulation.input_parameters.Nmeasure2
                for i in range(num_sweep_increments)
            ]
        )
        # set thermalization
        self.worm_simulation.input_parameters.thermalization = int(
            num_sweeps_values[0] / sweeps_to_thermalization_ratio
        )
        self.worm_simulation.save_parameters()

        pbar = tqdm(enumerate(num_sweeps_values), total=len(num_sweeps_values))
        self.worm_simulation.record["steps"] = []
        for step_idx, num_sweeps in pbar:
            self.worm_simulation.set_extension_sweeps_in_checkpoints(
                extension_sweeps=num_sweeps
            )

            # execute worm
            start_time = time.perf_counter()
            try:
                if step_idx > 0:
                    await self.run_continue(num_restarts=num_restarts)
                else:
                    await self.run(num_restarts=num_restarts)
            except RuntimeError as e:
                log.error(e)
                continue

            elapsed_time = time.perf_counter() - start_time

            # get current error
            error = self.worm_simulation.max_density_error
            pbar.set_description(
                f"Current error: {error}. Sweeps: {num_sweeps}. Num Nmeasure2: {num_sweeps/self.worm_simulation.input_parameters.Nmeasure2}. At step {step_idx} of {len(num_sweeps_values)}. Expected required number of measurements: {expected_required_num_measurements}"
            )

            self.worm_simulation.plot_observables()
            self.worm_simulation.record["steps"].append(
                {
                    "sweeps": int(num_sweeps),
                    "error": float(error) if error is not None else None,
                    "elapsed_time": float(elapsed_time),
                    "tau_max": float(self.worm_simulation.max_tau_int),
                }
            )

            # if error is below threshold, break
            if error is not None and error < max_abs_error_threshold:
                break

    def run_iterative_until_converged_sync(
        self,
        max_num_measurements_per_nmeasure2: int = 150000,
        min_num_measurements_per_nmeasure2: int = 1000,
        num_sweep_increments: int = 25,
        sweeps_to_thermalization_ratio: int = 10,
        max_abs_error_threshold: int = 0.015,
        num_restarts: int = 1,
    ) -> None:
        asyncio.run(
            self.run_iterative_until_converged(
                max_num_measurements_per_nmeasure2=max_num_measurements_per_nmeasure2,
                min_num_measurements_per_nmeasure2=min_num_measurements_per_nmeasure2,
                num_sweep_increments=num_sweep_increments,
                sweeps_to_thermalization_ratio=sweeps_to_thermalization_ratio,
                max_abs_error_threshold=max_abs_error_threshold,
                num_restarts=num_restarts,
            )
        )

    async def tune_nmeasure2(
        self,
        max_nmeasure2: int = 100000,
        min_nmeasure2: int = 35,
        num_measurements_per_nmeasure2: int = 15000,
        tau_threshold: int = 10,
        step_size_multiplication_factor: float = 1.8,
        sweeps_to_thermalization_ratio: int = 10,
        nmeasure2_to_nmeasure_ratio: int = 10,
        max_nmeasure: int = 100,
        min_nmeasure: int = 1,
    ) -> None:
        tune_simulation = self.worm_simulation.tune_simulation

        tune_simulation.record["steps"] = []

        Nmeasure2_values = np.array([min_nmeasure2])
        while Nmeasure2_values[-1] < max_nmeasure2:
            Nmeasure2_values = np.append(
                Nmeasure2_values, Nmeasure2_values[-1] * step_size_multiplication_factor
            )
        Nmeasure2_values = Nmeasure2_values.astype(int)

        # tune Nmeasure, Nmeasure2, thermalization, sweeps
        skip_next_counter = 0
        for idx, Nmeasure2 in tqdm(
            enumerate(Nmeasure2_values), total=len(Nmeasure2_values)
        ):
            if skip_next_counter > 0:
                skip_next_counter -= 1
                continue

            Nmeasure2 = int(Nmeasure2)
            Nmeasure = int(
                max(
                    min(max_nmeasure, Nmeasure2 / nmeasure2_to_nmeasure_ratio),
                    min_nmeasure,
                )
            )
            sweeps = int(num_measurements_per_nmeasure2 * Nmeasure2)
            thermalization = int(sweeps / sweeps_to_thermalization_ratio)

            tune_simulation.input_parameters.sweeps = sweeps
            tune_simulation.input_parameters.thermalization = thermalization
            tune_simulation.input_parameters.Nmeasure2 = Nmeasure2
            tune_simulation.input_parameters.Nmeasure = Nmeasure

            tune_runner = WormSimulationRunner(worm_simulation=tune_simulation)
            try:
                await tune_runner.run()
            except RuntimeError:
                continue

            tune_simulation.record["steps"].append(
                {
                    "sweeps": sweeps,
                    "thermalization": thermalization,
                    "Nmeasure2": Nmeasure2,
                    "tau_max": float(tune_simulation.max_tau_int),
                    "max_density_error": float(tune_simulation.max_density_error),
                }
            )

            tune_simulation.plot_observables()

            # plot all tau values
            plt.plot([step["tau_max"] for step in tune_simulation.record["steps"]])
            plt.title("tau_max")
            plt.yscale("log")
            plt.savefig(
                tune_simulation.get_plot_dir(tune_simulation.save_dir)
                / "tau_int_max.png"
            )
            plt.close()

            tau_max_values = np.array(
                [step["tau_max"] for step in tune_simulation.record["steps"]]
            )

            if (tau_max_values[-1:] < tau_threshold).all() and (
                tau_max_values[-1:] > 0
            ).all():
                break

            # get biggest smaller 2**x value
            if tau_max_values[-1] > 0 and not tau_max_values[-1] is None:
                skip_next_counter = int(np.log2(tau_max_values[-1])) - 2

        self.worm_simulation.input_parameters.Nmeasure2 = Nmeasure2
        self.worm_simulation.input_parameters.Nmeasure = Nmeasure
        self.worm_simulation.save_parameters()

        # save uncorrected density error for final run
        tune_simulation.record[
            "uncorrected_max_density_error"
        ] = tune_simulation.uncorrected_max_density_error

    def tune_nmeasure2_sync(
        self,
        max_nmeasure2: int = 100000,
        min_nmeasure2: int = 50,
        num_measurements_per_nmeasure2: int = 15000,
        tau_threshold: int = 10,
        step_size_multiplication_factor: float = 1.8,
        sweeps_to_thermalization_ratio: int = 10,
        nmeasure2_to_nmeasure_ratio: int = 10,
        max_nmeasure: int = 100,
        min_nmeasure: int = 1,
    ) -> None:
        asyncio.run(
            self.tune_nmeasure2(
                max_nmeasure2=max_nmeasure2,
                min_nmeasure2=min_nmeasure2,
                num_measurements_per_nmeasure2=num_measurements_per_nmeasure2,
                tau_threshold=tau_threshold,
                step_size_multiplication_factor=step_size_multiplication_factor,
                sweeps_to_thermalization_ratio=sweeps_to_thermalization_ratio,
                nmeasure2_to_nmeasure_ratio=nmeasure2_to_nmeasure_ratio,
                max_nmeasure=max_nmeasure,
                min_nmeasure=min_nmeasure,
            )
        )
