import asyncio
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List, Dict

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

        if not "steps" in self.worm_simulation.record:
            self.worm_simulation.record["steps"] = []

        skip_next_counter = 0
        if len(self.worm_simulation.record["steps"]) > 0:
            # get last sweeps value
            last_sweeps = self.worm_simulation.record["steps"][-1]["sweeps"]
            skip_next_counter = np.argwhere((num_sweeps_values - last_sweeps) > 0).min()
            log.info(
                f"Found existing run. Continuing with sweeps={num_sweeps_values[skip_next_counter]}. Skipping {skip_next_counter} steps."
            )

        pbar = tqdm(enumerate(num_sweeps_values), total=len(num_sweeps_values))
        for step_idx, num_sweeps in pbar:
            if skip_next_counter > 0:
                skip_next_counter -= 1
                continue

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
                    "num_nmeasure2": int(
                        num_sweeps / self.worm_simulation.input_parameters.Nmeasure2
                    ),
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
        def get_tau_max_keys(steps: List[Dict]) -> List[str]:
            # get tau_max key
            tau_max_keys_steps = [
                list(
                    filter(
                        lambda k: "tau_max" in k,
                        step.keys(),
                    )
                )
                for step in steps
            ]
            if any(
                [
                    len(tau_max_keys_step) == 0
                    for tau_max_keys_step in tau_max_keys_steps
                ]
            ):
                raise Exception("No tau_max key found in record.")
            else:
                return [
                    list(tau_max_keys_step)[0]
                    for tau_max_keys_step in tau_max_keys_steps
                ]

        def get_tau_max_values(steps: List[Dict]) -> np.ndarray:
            tau_max_keys = get_tau_max_keys(steps)
            return np.array(
                [step[tau_max_key] for step, tau_max_key in zip(steps, tau_max_keys)]
            )

        def break_condition(tune_sim: WormSimulation) -> bool:
            # get tau_max key
            tau_max_values = get_tau_max_values(tune_simulation.record["steps"])

            if (tau_max_values[-1:] < tau_threshold).all() and (
                tau_max_values[-1:] > 0
            ).all():
                return True

        def finalize_tuning(
            parent_sim: WormSimulation, tune_sim: WormSimulation
        ) -> None:
            parent_sim.input_parameters.sweeps = tune_sim.input_parameters.sweeps
            parent_sim.input_parameters.thermalization = (
                tune_sim.input_parameters.thermalization
            )
            parent_sim.input_parameters.Nmeasure2 = tune_sim.input_parameters.Nmeasure2
            parent_sim.input_parameters.Nmeasure = tune_sim.input_parameters.Nmeasure
            parent_sim.save_parameters()

            # save uncorrected density error for final run
            tune_sim.record[
                "uncorrected_max_density_error"
            ] = tune_sim.uncorrected_max_density_error

        tune_simulation = self.worm_simulation.tune_simulation

        if not "steps" in tune_simulation.record:
            tune_simulation.record["steps"] = []

        Nmeasure2_values = np.array([min_nmeasure2])
        while Nmeasure2_values[-1] < max_nmeasure2:
            Nmeasure2_values = np.append(
                Nmeasure2_values, Nmeasure2_values[-1] * step_size_multiplication_factor
            )
        Nmeasure2_values = Nmeasure2_values.astype(int)
        skip_next_counter = 0

        if len(tune_simulation.record["steps"]) > 0:
            if break_condition(tune_simulation):
                finalize_tuning(self.worm_simulation, tune_simulation)
                log.info("Tuning finished.")
                return
            else:
                # get last Nmeasure2 value
                last_Nmeasure2 = tune_simulation.record["steps"][-1]["Nmeasure2"]
                # take next higher value
                skip_next_counter = np.argwhere(
                    (Nmeasure2_values - last_Nmeasure2) > 0
                ).min()
                log.info(
                    f"Found existing tuning runs. Continuing with Nmeasure2={Nmeasure2_values[skip_next_counter]}. Skipping {skip_next_counter} steps."
                )

        # tune Nmeasure, Nmeasure2, thermalization, sweeps
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

            tau_max_values = get_tau_max_values(tune_simulation.record["steps"])

            # plot all tau values
            plt.plot(tau_max_values)
            plt.title("tau_max")
            plt.yscale("log")
            plt.savefig(
                tune_simulation.get_plot_dir(tune_simulation.save_dir)
                / "tau_int_max.png"
            )
            plt.close()

            if break_condition(tune_simulation):
                break

            # get biggest smaller 2**x value
            if tau_max_values[-1] > 0 and not tau_max_values[-1] is None:
                skip_next_counter = int(np.log2(tau_max_values[-1])) - 2

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
