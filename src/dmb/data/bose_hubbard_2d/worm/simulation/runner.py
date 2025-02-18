"""Module for running worm simulation."""

import asyncio
import time
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from dmb.data.dispatching import ReturnCode

from .parameters import WormInputParameters
from .sim import WormSimulationInterface


def sync_async(func: Callable) -> Callable:
    """Decorator to run async function synchronously."""

    def wrapper(*args: tuple, **kwargs: dict) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def get_tune_nmeasure2_values(
    min_nmeasure2: int, max_nmeasure2: int, step_size_multiplication_factor: float
) -> list[int]:
    """Get Nmeasure2 values for tuning.

    Args:
        min_nmeasure2: Minimum Nmeasure2.
        max_nmeasure2: Maximum Nmeasure2.
        step_size_multiplication_factor: Step size multiplication factor,
            for increasing Nmeasure2.
    Returns:
        Nmeasure2 values.
    """
    if step_size_multiplication_factor <= 1:
        raise ValueError("Step size multiplication factor must be greater than 1.")
    if min_nmeasure2 >= max_nmeasure2:
        raise ValueError("Minimum Nmeasure2 must be less than maximum Nmeasure2.")

    max_exponent = int(
        np.emath.logn(step_size_multiplication_factor, max_nmeasure2 / min_nmeasure2)
    )  # floor
    Nmeasure2_values: list[int] = (
        (min_nmeasure2 * step_size_multiplication_factor ** np.arange(max_exponent + 1))
        .round()
        .astype(int)
        .tolist()
    )

    return Nmeasure2_values


def get_run_iteratively_num_sweeps_values(
    Nmeasure2: int,
    min_num_measurements_per_nmeasure2: int,
    max_num_measurements_per_nmeasure2: int,
    num_sweep_increments: int,
) -> np.ndarray:
    """Get number of sweeps values for running iteratively.

    Args:
        Nmeasure2: Nmeasure2.
        min_num_measurements_per_nmeasure2: Minimum number of
            measurements per Nmeasure2.
        max_num_measurements_per_nmeasure2: Maximum number of
            measurements per Nmeasure2.
        num_sweep_increments: Number of sweep increments.
    Returns:
        Number of sweeps values.
    """
    return np.array(
        [
            (
                min_num_measurements_per_nmeasure2
                + increment
                * int(
                    (
                        max_num_measurements_per_nmeasure2
                        - min_num_measurements_per_nmeasure2
                    )
                    / num_sweep_increments
                )
            )
            * Nmeasure2
            for increment in range(num_sweep_increments)
        ]
    )


def set_simulation_parameters(
    simulation: WormSimulationInterface,
    Nmeasure2: int,
    sweeps: int,
    sweeps_to_thermalization_ratio: int = 10,
    nmeasure2_to_nmeasure_ratio: int = 10,
    max_nmeasure: int = 10000,
    min_nmeasure: int = 1,
    seed: int | None = None,
) -> None:
    """Set simulation parameters.

    Args:
        simulation: A WormSimulation instance.
        Nmeasure2: Nmeasure2.
        sweeps: Sweeps.
        sweeps_to_thermalization_ratio: Ratio of sweeps to thermalization.
        nmeasure2_to_nmeasure_ratio: Ratio of Nmeasure2 to Nmeasure.
        max_nmeasure: Maximum Nmeasure.
        min_nmeasure: Minimum Nmeasure.
        seed: Seed.
    """
    simulation.input_parameters = WormInputParameters(
        **{
            **simulation.input_parameters.__dict__,
            "Nmeasure2": Nmeasure2,
            "sweeps": sweeps,
            "thermalization": int(sweeps / sweeps_to_thermalization_ratio),
            "Nmeasure": int(
                max(
                    min(max_nmeasure, Nmeasure2 / nmeasure2_to_nmeasure_ratio),
                    min_nmeasure,
                )
            ),
            "seed": seed or simulation.input_parameters.seed,
        }
    )
    simulation.save_parameters()


def get_tau_max_values_from_simulation_record(steps: list[dict]) -> np.ndarray:
    """Get tau_max values from simulation record.

    Args:
        steps: Steps in simulation record.
    Returns:
        Tau_max values.
    """

    def get_tau_max_keys(steps: list[dict]) -> list[str]:
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
        return [list(tau_max_keys_step)[0] for tau_max_keys_step in tau_max_keys_steps]

    tau_max_keys = get_tau_max_keys(steps)
    return np.array(
        [step[tau_max_key] for step, tau_max_key in zip(steps, tau_max_keys)]
    )


class WormSimulationRunner:
    """Class to run worm simulation."""

    def __init__(self, worm_simulation: WormSimulationInterface):
        self.worm_simulation = worm_simulation

        self.run_iterative_until_converged_sync = sync_async(
            self.run_iterative_until_converged
        )
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
        max_num_measurements_per_nmeasure2: int = 500000,
        min_num_measurements_per_nmeasure2: int = 15000,
        num_sweep_increments: int = 30,
        sweeps_to_thermalization_ratio: int = 10,
        max_abs_error_threshold: float = 0.015,
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
        Nmeasure2 = Nmeasure2 or (
            self.worm_simulation.tune_simulation.record["steps"][-1]["Nmeasure2"]
            if len(self.worm_simulation.tune_simulation.record["steps"])
            else None
        )

        if Nmeasure2 is None:
            raise ValueError("Nmeasure2 must be set or tuned simulation must exist.")

        num_sweeps_values = get_run_iteratively_num_sweeps_values(
            Nmeasure2=Nmeasure2,
            min_num_measurements_per_nmeasure2=min_num_measurements_per_nmeasure2,
            max_num_measurements_per_nmeasure2=max_num_measurements_per_nmeasure2,
            num_sweep_increments=num_sweep_increments,
        )

        for step_idx, num_sweeps in tqdm(
            enumerate(num_sweeps_values), total=len(num_sweeps_values)
        ):
            self.worm_simulation.set_extension_sweeps_in_checkpoints(
                extension_sweeps=num_sweeps
            )

            # execute worm
            start_time = time.perf_counter()

            if step_idx > 0 and self.worm_simulation.valid:
                return_code = await self.run_continue()
            else:
                set_simulation_parameters(
                    simulation=self.worm_simulation,
                    Nmeasure2=Nmeasure2,
                    sweeps=num_sweeps,
                    sweeps_to_thermalization_ratio=sweeps_to_thermalization_ratio,
                )

                return_code = await self.run()

            if return_code != ReturnCode.SUCCESS:
                self.worm_simulation.file_logger.error(
                    f"Error running simulation. Return code: {return_code}."
                )

            elapsed_time = time.perf_counter() - start_time

            self.worm_simulation.file_logger.info(
                (
                    "Running Iteratively:\n"
                    "Current error: {}."
                    "Sweeps: {}. Num Nmeasure2: {}"
                    "At step {} of {}."
                ).format(
                    self.worm_simulation.max_density_error,
                    num_sweeps,
                    int(num_sweeps / self.worm_simulation.input_parameters.Nmeasure2),
                    step_idx + 1,
                    len(num_sweeps_values),
                )
            )

            self.worm_simulation.plot_observables()
            self.worm_simulation.record["steps"].append(
                {
                    "sweeps": int(num_sweeps),
                    "error": self.worm_simulation.max_density_error,
                    "elapsed_time": float(elapsed_time),
                    "tau_max": self.worm_simulation.max_tau_int,
                    "num_nmeasure2": int(
                        num_sweeps / self.worm_simulation.input_parameters.Nmeasure2
                    ),
                }
            )

            # if error is below threshold, break
            if (
                self.worm_simulation.max_density_error is not None
                and self.worm_simulation.max_density_error < max_abs_error_threshold
            ):
                break

    async def tune_nmeasure2(
        self,
        max_nmeasure2: int = 500000,
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

        def break_condition() -> bool:
            # get tau_max key
            tau_max_values = get_tau_max_values_from_simulation_record(
                tune_simulation.record["steps"]
            )

            return bool(
                tau_max_values[-1] is not None and tau_max_values[-1] <= tau_threshold
            )

        nmeasure2_values = get_tune_nmeasure2_values(
            min_nmeasure2=min_nmeasure2,
            max_nmeasure2=max_nmeasure2,
            step_size_multiplication_factor=step_size_multiplication_factor,
        )

        tune_simulation.file_logger.info(
            f"Starting tuning with Nmeasure2 planned values: {nmeasure2_values}.\n\n"
        )

        skip_next_counter = 0
        # tune Nmeasure, Nmeasure2, thermalization, sweeps
        for idx, Nmeasure2 in tqdm(
            enumerate(nmeasure2_values), total=len(nmeasure2_values)
        ):
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

            tune_simulation.record["steps"].append(
                {
                    "sweeps": tune_simulation.input_parameters.sweeps,
                    "thermalization": tune_simulation.input_parameters.thermalization,
                    "Nmeasure2": Nmeasure2,
                    "tau_max": tune_simulation.max_tau_int,
                    "max_density_error": tune_simulation.max_density_error,
                }
            )

            tune_simulation.plot_observables()

            if break_condition():
                tune_simulation.file_logger.info(
                    (
                        "Tuning finished at: \n Nmeasure2: {}, tau_max: {},"
                        " max_density_error: {}.\n\n"
                    ).format(
                        tune_simulation.input_parameters.Nmeasure2,
                        tune_simulation.max_tau_int,
                        tune_simulation.max_density_error,
                    )
                )
                break

            if tune_simulation.max_tau_int is not None:
                # get biggest smaller 2**x value - 2,
                # at most min(3,len(nmeasure2_values)/3, remaining steps-1)
                skip_next_counter = min(
                    3,
                    len(nmeasure2_values) // 4,
                    len(nmeasure2_values) - idx - 2,
                    int(
                        np.emath.logn(
                            step_size_multiplication_factor, tune_simulation.max_tau_int
                        )
                    )
                    - 2,
                )

                tune_simulation.file_logger.info(
                    f"Skipping next {skip_next_counter} steps.\n\n"
                )

        if not break_condition():
            tune_simulation.file_logger.info(
                "Tuning finished without reaching tau threshold."
                "Current tau_max: {}.\n\n".format(tune_simulation.max_tau_int)
            )
