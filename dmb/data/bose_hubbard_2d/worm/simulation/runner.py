import asyncio
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dmb.data.dispatching import ReturnCode

from .parameters import WormInputParameters
from .sim import WormSimulation


def sync_async(func):

    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def get_expected_required_num_measurements(
        simulation: WormSimulation, max_abs_error_threshold: float) -> int:
    return (2 * (simulation.uncorrected_max_density_error /
                 max_abs_error_threshold)**2 * simulation.max_tau_int)


def set_simulation_parameters(simulation: WormSimulation,
                              Nmeasure2: int,
                              sweeps: int,
                              sweeps_to_thermalization_ratio: int = 10,
                              nmeasure2_to_nmeasure_ratio: int = 10,
                              max_nmeasure: int = 10000,
                              min_nmeasure: int = 1,
                              seed: int | None = None) -> None:

    simulation.input_parameters = WormInputParameters(
        **{
            **simulation.input_parameters.__dict__, "Nmeasure2":
            Nmeasure2,
            "sweeps":
            sweeps,
            "thermalization":
            int(sweeps / sweeps_to_thermalization_ratio),
            "Nmeasure":
            int(
                max(
                    min(max_nmeasure, Nmeasure2 / nmeasure2_to_nmeasure_ratio),
                    min_nmeasure,
                )),
            "seed":
            seed or simulation.input_parameters.seed
        })
    simulation.save_parameters()


def get_tau_max_values_from_simulation_record(steps: list[dict]) -> np.ndarray:

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

    tau_max_keys = get_tau_max_keys(steps)
    return np.array(
        [step[tau_max_key] for step, tau_max_key in zip(steps, tau_max_keys)])


class WormSimulationRunner:
    """Class to run worm simulation."""

    def __init__(self, worm_simulation: WormSimulation):
        self.worm_simulation = worm_simulation

        self.run_iterative_until_converged_sync = sync_async(
            self.run_iterative_until_converged)
        self.tune_nmeasure2_sync = sync_async(self.tune_nmeasure2)

    async def run(self) -> ReturnCode:
        """Run worm simulation."""
        self.worm_simulation.save_parameters()
        return await self.worm_simulation.execute_worm()

    async def run_continue(self) -> ReturnCode:
        """Continue worm simulation."""
        self.worm_simulation.save_parameters()
        return await self.worm_simulation.execute_worm_continue()

    async def run_iterative_until_converged(
        self,
        max_num_measurements_per_nmeasure2: int = 250000,
        min_num_measurements_per_nmeasure2: int = 15000,
        num_sweep_increments: int = 35,
        sweeps_to_thermalization_ratio: int = 10,
        max_abs_error_threshold: int = 0.015,
        Nmeasure2: int | None = None,
    ) -> None:
        """Run worm simulation iteratively until converged.

        Args:
            max_num_measurements_per_nmeasure2: 
                Maximum number of measurements per Nmeasure2.
            min_num_measurements_per_nmeasure2: 
                Minimum number of measurements per Nmeasure2.
            num_sweep_increments: Number of sweep increments.
            sweeps_to_thermalization_ratio: Ratio of sweeps to thermalization.
            max_abs_error_threshold: Maximum absolute error threshold.
        """
        Nmeasure2 = Nmeasure2 or self.worm_simulation.tune_simulation.record[
            "steps"][-1]["Nmeasure2"]

        if Nmeasure2 is None:
            raise ValueError(
                "Nmeasure2 must be set or tuned simulation must exist.")

        num_sweeps_values = np.array([
            (min_num_measurements_per_nmeasure2 + i * int(
                (max_num_measurements_per_nmeasure2 -
                 min_num_measurements_per_nmeasure2) / num_sweep_increments)) *
            self.worm_simulation.input_parameters.Nmeasure2
            for i in range(num_sweep_increments)
        ])

        set_simulation_parameters(
            simulation=self.worm_simulation,
            Nmeasure2=Nmeasure2,
            sweeps=num_sweeps_values[0],
            sweeps_to_thermalization_ratio=sweeps_to_thermalization_ratio)

        pbar = tqdm(enumerate(num_sweeps_values), total=len(num_sweeps_values))
        for step_idx, num_sweeps in pbar:
            self.worm_simulation.set_extension_sweeps_in_checkpoints(
                extension_sweeps=num_sweeps)

            # execute worm
            start_time = time.perf_counter()

            if step_idx > 0 and self.worm_simulation.valid:
                return_code = await self.run_continue()
            else:
                return_code = await self.run()

            if return_code != ReturnCode.SUCCESS:
                self.worm_simulation.file_logger.error(
                    f"Error running simulation. Return code: {return_code}.")

            elapsed_time = time.perf_counter() - start_time

            pbar.set_description(f"""Current error: {
                    self.worm_simulation.max_density_error}.
                . Sweeps: {num_sweeps}.
                Num Nmeasure2: {num_sweeps/
                                self.worm_simulation.input_parameters.Nmeasure2}.
                At step {step_idx} of {len(num_sweeps_values)}.
                Expected required number of measurements:
                 {get_expected_required_num_measurements(self.worm_simulation.tune_simulation,
                                                          max_abs_error_threshold)}"""
                                 )

            self.worm_simulation.plot_observables()
            self.worm_simulation.record["steps"].append({
                "sweeps":
                int(num_sweeps),
                "error":
                self.worm_simulation.max_density_error,
                "elapsed_time":
                float(elapsed_time),
                "tau_max":
                self.worm_simulation.max_tau_int,
                "num_nmeasure2":
                int(num_sweeps /
                    self.worm_simulation.input_parameters.Nmeasure2),
            })

            # if error is below threshold, break
            if self.worm_simulation.max_density_error is not None and \
                self.worm_simulation.max_density_error < max_abs_error_threshold:
                break

    async def tune_nmeasure2(
        self,
        max_nmeasure2: int = 250000,
        min_nmeasure2: int = 35,
        num_measurements_per_nmeasure2: int = 15000,
        tau_threshold: int = 10,
        step_size_multiplication_factor: float = 1.8,
    ) -> None:
        """Tune Nmeasure2.

        Args:
            max_nmeasure2: Maximum Nmeasure2.
            min_nmeasure2: Minimum Nmeasure2.
            num_measurements_per_nmeasure2: Number of measurements per Nmeasure2.
            tau_threshold: Tau threshold.
            step_size_multiplication_factor: Step size multiplication factor.
            nmeasure2_to_nmeasure_ratio: Ratio of Nmeasure2 to Nmeasure.
            max_nmeasure: Maximum Nmeasure.
            min_nmeasure: Minimum Nmeasure.
        """
        tune_simulation = self.worm_simulation.tune_simulation

        if not "steps" in tune_simulation.record:
            tune_simulation.record["steps"] = []

        def break_condition() -> bool:
            # get tau_max key
            tau_max_values = get_tau_max_values_from_simulation_record(
                tune_simulation.record["steps"])

            if tau_max_values[-1] is not None and tau_max_values[
                    -1] < tau_threshold:
                return True

        Nmeasure2_values = np.array([min_nmeasure2])
        while Nmeasure2_values[-1] < max_nmeasure2:
            Nmeasure2_values = np.append(
                Nmeasure2_values,
                Nmeasure2_values[-1] * step_size_multiplication_factor)
        Nmeasure2_values = Nmeasure2_values.astype(int)
        skip_next_counter = 0

        # tune Nmeasure, Nmeasure2, thermalization, sweeps
        for idx, Nmeasure2 in tqdm(enumerate(Nmeasure2_values),
                                   total=len(Nmeasure2_values)):
            if skip_next_counter > 0:
                skip_next_counter -= 1
                continue

            set_simulation_parameters(
                tune_simulation,
                Nmeasure2=int(Nmeasure2),
                sweeps=int(num_measurements_per_nmeasure2 * Nmeasure2),
            )

            tune_runner = WormSimulationRunner(worm_simulation=tune_simulation)

            return_code = await tune_runner.run()

            if return_code != ReturnCode.SUCCESS:
                continue

            tune_simulation.record["steps"].append({
                "sweeps":
                tune_simulation.input_parameters.sweeps,
                "thermalization":
                tune_simulation.input_parameters.thermalization,
                "Nmeasure2":
                Nmeasure2,
                "tau_max":
                tune_simulation.max_tau_int,
                "max_density_error":
                tune_simulation.max_density_error,
            })

            tune_simulation.plot_observables()

            if break_condition():
                self.worm_simulation.file_logger.info(
                    "Tuning finished at: \n Nmeasure2: {}, tau_max: {}, max_density_error: {}."
                    .format(tune_simulation.input_parameters.Nmeasure2,
                            tune_simulation.max_tau_int,
                            tune_simulation.max_density_error))
                break

            # get biggest smaller 2**x value
            if tune_simulation.max_tau_int is not None:
                skip_next_counter = int(np.log2(
                    tune_simulation.max_tau_int)) - 2
