import asyncio
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .sim import WormSimulation


def sync_async(func):

    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class WormSimulationRunner:
    """Class to run worm simulation."""

    def __init__(self, worm_simulation: WormSimulation):
        self.worm_simulation = worm_simulation

        run_iterative_until_converged_sync = sync_async(
            self.run_iterative_until_converged)
        tune_nmeasure2_sync = sync_async(self.tune_nmeasure2)

    async def _run(self):
        """Run worm simulation."""
        self.worm_simulation.save_parameters()
        try:
            await self.worm_simulation.execute_worm()
        except RuntimeError as e:
            self.worm_simulation.file_logger.error(e)
            raise e

    async def _run_continue(self):
        """Continue worm simulation."""
        self.worm_simulation.save_parameters()
        try:
            await self.worm_simulation.execute_worm_continue()
        except RuntimeError as e:
            self.worm_simulation.file_logger.error(e)
            raise e

    async def run_iterative_until_converged(
        self,
        max_num_measurements_per_nmeasure2: int = 250000,
        min_num_measurements_per_nmeasure2: int = 15000,
        num_sweep_increments: int = 35,
        sweeps_to_thermalization_ratio: int = 10,
        max_abs_error_threshold: int = 0.015,
        restart: bool = True,
    ) -> None:
        """Run worm simulation iteratively until converged.

        Args:
            max_num_measurements_per_nmeasure2: Maximum number of measurements per Nmeasure2.
            min_num_measurements_per_nmeasure2: Minimum number of measurements per Nmeasure2.
            num_sweep_increments: Number of sweep increments.
            sweeps_to_thermalization_ratio: Ratio of sweeps to thermalization.
            max_abs_error_threshold: Maximum absolute error threshold.
            restart: Restart simulation.
        """
        try:
            expected_required_num_measurements = (
                2 *
                (self.worm_simulation.tune_simulation.
                 uncorrected_max_density_error / max_abs_error_threshold)**2 *
                self.worm_simulation.tune_simulation.max_tau_int)
        except KeyError:
            self.worm_simulation.file_logger.info(
                "No tune simulation found. Will not be able to estimate required number of measurements."
            )
            expected_required_num_measurements = None

        # update Nmeasure2 from tune simulation
        try:
            tune_Nmeasure2 = self.worm_simulation.tune_simulation.record[
                "steps"][-1]["Nmeasure2"]
            self.worm_simulation.input_parameters.Nmeasure2 = tune_Nmeasure2
            self.worm_simulation.input_parameters.Nmeasure = int(
                tune_Nmeasure2 / 10)
            self.worm_simulation.save_parameters()
        except KeyError:
            self.worm_simulation.file_logger.info(
                "No tune simulation found. Using default Nmeasure2.")

        num_sweeps_values = np.array([
            (min_num_measurements_per_nmeasure2 + i * int(
                (max_num_measurements_per_nmeasure2 -
                 min_num_measurements_per_nmeasure2) / num_sweep_increments)) *
            self.worm_simulation.input_parameters.Nmeasure2
            for i in range(num_sweep_increments)
        ])
        self.worm_simulation.file_logger.info(
            f"""Will iterate over {num_sweep_increments} steps with sweeps values: {num_sweeps_values} \n\n"""
        )
        self.worm_simulation.input_parameters.sweeps = num_sweeps_values[0]
        self.worm_simulation.save_parameters()

        # set thermalization
        self.worm_simulation.input_parameters.thermalization = int(
            num_sweeps_values[0] / sweeps_to_thermalization_ratio)
        self.worm_simulation.save_parameters()

        if not "steps" in self.worm_simulation.record or restart:
            self.worm_simulation.record["steps"] = []

        skip_next_counter = 0
        checkpoint_produced = False
        if len(self.worm_simulation.record["steps"]) > 0 and not restart:
            # get last sweeps value
            last_sweeps = self.worm_simulation.record["steps"][-1]["sweeps"]
            skip_next_counter = np.argwhere((num_sweeps_values -
                                             last_sweeps) > 0).min()
            self.worm_simulation.file_logger.info(
                f"Found existing run. Continuing with sweeps={num_sweeps_values[skip_next_counter]}. Skipping {skip_next_counter} steps."
            )
            checkpoint_produced = True

        pbar = tqdm(enumerate(num_sweeps_values), total=len(num_sweeps_values))
        for step_idx, num_sweeps in pbar:
            if skip_next_counter > 0:
                skip_next_counter -= 1
                continue

            self.worm_simulation.set_extension_sweeps_in_checkpoints(
                extension_sweeps=num_sweeps)

            # execute worm
            start_time = time.perf_counter()
            try:
                if step_idx > 0 and checkpoint_produced:
                    await self._run_continue()
                else:
                    await self._run()
            except RuntimeError as e:
                self.worm_simulation.file_logger.error(e)
                continue

            elapsed_time = time.perf_counter() - start_time

            # get current error
            error = self.worm_simulation.max_density_error

            if error is not None and error > 0:
                checkpoint_produced = True

            pbar.set_description(
                f"Current error: {error}. Sweeps: {num_sweeps}. Num Nmeasure2: {num_sweeps/self.worm_simulation.input_parameters.Nmeasure2}. At step {step_idx} of {len(num_sweeps_values)}. Expected required number of measurements: {expected_required_num_measurements}"
            )

            self.worm_simulation.plot_observables()
            self.worm_simulation.record["steps"].append({
                "sweeps":
                int(num_sweeps),
                "error":
                error,
                "elapsed_time":
                float(elapsed_time),
                "tau_max":
                self.worm_simulation.max_tau_int,
                "num_nmeasure2":
                int(num_sweeps /
                    self.worm_simulation.input_parameters.Nmeasure2),
            })

            # if error is below threshold, break
            if error is not None and error < max_abs_error_threshold and error > 0:
                break

    async def tune_nmeasure2(
        self,
        max_nmeasure2: int = 250000,
        min_nmeasure2: int = 35,
        num_measurements_per_nmeasure2: int = 15000,
        tau_threshold: int = 10,
        step_size_multiplication_factor: float = 1.8,
        sweeps_to_thermalization_ratio: int = 10,
        nmeasure2_to_nmeasure_ratio: int = 10,
        max_nmeasure: int = 10000,
        min_nmeasure: int = 1,
    ) -> None:
        """Tune Nmeasure2.

        Args:
            max_nmeasure2: Maximum Nmeasure2.
            min_nmeasure2: Minimum Nmeasure2.
            num_measurements_per_nmeasure2: Number of measurements per Nmeasure2.
            tau_threshold: Tau threshold.
            step_size_multiplication_factor: Step size multiplication factor.
            sweeps_to_thermalization_ratio: Ratio of sweeps to thermalization.
            nmeasure2_to_nmeasure_ratio: Ratio of Nmeasure2 to Nmeasure.
            max_nmeasure: Maximum Nmeasure.
            min_nmeasure: Minimum Nmeasure.
        """

        def get_tau_max_keys(steps: list[dict]) -> list[str]:
            # get tau_max key
            tau_max_keys_steps = [
                list(filter(
                    lambda k: "tau_max" in k,
                    step.keys(),
                )) for step in steps
            ]
            if any([
                    len(tau_max_keys_step) == 0
                    for tau_max_keys_step in tau_max_keys_steps
            ]):
                raise Exception("No tau_max key found in record.")
            else:
                return [
                    list(tau_max_keys_step)[0]
                    for tau_max_keys_step in tau_max_keys_steps
                ]

        def get_tau_max_values(steps: list[dict]) -> np.ndarray:
            tau_max_keys = get_tau_max_keys(steps)
            return np.array([
                step[tau_max_key]
                for step, tau_max_key in zip(steps, tau_max_keys)
            ])

        def break_condition(tune_sim: WormSimulation) -> bool:
            # get tau_max key
            tau_max_values = get_tau_max_values(
                tune_simulation.record["steps"])

            if tau_max_values[-1] < tau_threshold and (tau_max_values[-1] > 0):
                return True

        def finalize_tuning(parent_sim: WormSimulation,
                            tune_sim: WormSimulation) -> None:
            parent_sim.input_parameters.sweeps = tune_sim.input_parameters.sweeps
            parent_sim.input_parameters.thermalization = (
                tune_sim.input_parameters.thermalization)
            parent_sim.input_parameters.Nmeasure2 = tune_sim.input_parameters.Nmeasure2
            parent_sim.input_parameters.Nmeasure = tune_sim.input_parameters.Nmeasure
            parent_sim.save_parameters()

            # save uncorrected density error for final run
            tune_sim.record["uncorrected_max_density_error"] = (
                tune_sim.uncorrected_max_density_error)

        tune_simulation = self.worm_simulation.tune_simulation

        if not "steps" in tune_simulation.record:
            tune_simulation.record["steps"] = []

        Nmeasure2_values = np.array([min_nmeasure2])
        while Nmeasure2_values[-1] < max_nmeasure2:
            Nmeasure2_values = np.append(
                Nmeasure2_values,
                Nmeasure2_values[-1] * step_size_multiplication_factor)
        Nmeasure2_values = Nmeasure2_values.astype(int)
        skip_next_counter = 0

        if len(tune_simulation.record["steps"]) > 0:
            if break_condition(tune_simulation):
                finalize_tuning(self.worm_simulation, tune_simulation)
                self.worm_simulation.file_logger.info("Tuning finished.")
                return
            else:
                # get last Nmeasure2 value
                last_Nmeasure2 = tune_simulation.record["steps"][-1][
                    "Nmeasure2"]
                # take next higher value
                skip_next_counter = np.argwhere((Nmeasure2_values -
                                                 last_Nmeasure2) > 0).min()
                self.worm_simulation.file_logger.info(
                    f"Found existing tuning runs. Continuing with Nmeasure2={Nmeasure2_values[skip_next_counter]}. Skipping {skip_next_counter} steps."
                )

        # tune Nmeasure, Nmeasure2, thermalization, sweeps
        for idx, Nmeasure2 in tqdm(enumerate(Nmeasure2_values),
                                   total=len(Nmeasure2_values)):
            if skip_next_counter > 0:
                skip_next_counter -= 1
                continue

            Nmeasure2 = int(Nmeasure2)
            Nmeasure = int(
                max(
                    min(max_nmeasure, Nmeasure2 / nmeasure2_to_nmeasure_ratio),
                    min_nmeasure,
                ))
            sweeps = int(num_measurements_per_nmeasure2 * Nmeasure2)
            thermalization = int(sweeps / sweeps_to_thermalization_ratio)

            tune_simulation.input_parameters.sweeps = sweeps
            tune_simulation.input_parameters.thermalization = thermalization
            tune_simulation.input_parameters.Nmeasure2 = Nmeasure2
            tune_simulation.input_parameters.Nmeasure = Nmeasure

            tune_runner = WormSimulationRunner(worm_simulation=tune_simulation)

            try:
                await tune_runner._run()
            except RuntimeError as e:
                self.worm_simulation.file_logger.error(
                    f"Error {e} running tune simulation.")
                continue

            tune_simulation.record["steps"].append({
                "sweeps":
                sweeps,
                "thermalization":
                thermalization,
                "Nmeasure2":
                Nmeasure2,
                "tau_max":
                tune_simulation.max_tau_int,
                "max_density_error":
                tune_simulation.max_density_error,
            })

            if (tune_simulation.max_density_error is None
                    or tune_simulation.max_density_error < 0):
                # adjust seed
                tune_simulation.input_parameters.seed = np.random.randint(
                    0, 2**16)

            tune_simulation.plot_observables()
            tau_max_values = get_tau_max_values(
                tune_simulation.record["steps"])

            # plot all tau values
            plt.plot(tau_max_values)
            plt.title("tau_max")

            # suppress warning: UserWarning: Data has no positive values, and therefore cannot be log-scaled.
            if (tau_max_values > 0).any():
                plt.yscale("log")

            plt.savefig(
                tune_simulation.get_plot_dir(tune_simulation.save_dir) /
                "tau_int_max.png")
            plt.close()

            if break_condition(tune_simulation):
                finalize_tuning(self.worm_simulation, tune_simulation)
                self.worm_simulation.file_logger.info(
                    "Tuning finished at: \n Nmeasure2: {}, tau_max: {}, max_density_error: {}."
                    .format(Nmeasure2, tau_max_values[-1],
                            tune_simulation.max_density_error))
                break

            # get biggest smaller 2**x value
            if tau_max_values[-1] > 0 and not tau_max_values[-1] is None:
                skip_next_counter = int(np.log2(tau_max_values[-1])) - 2