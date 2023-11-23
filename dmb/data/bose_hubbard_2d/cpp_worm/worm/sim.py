import subprocess
from pathlib import Path
import numpy as np
import h5py
from typing import Optional
import os
from tqdm import tqdm
from copy import deepcopy
from dmb.utils import create_logger
from collections import defaultdict
from dmb.data.bose_hubbard_2d.cpp_worm.worm.helpers import (
    check_if_slurm_is_installed_and_running,
    write_sbatch_script,
    call_sbatch_and_wait,
)
from dmb.utils import REPO_DATA_ROOT
from dmb.utils.syjson import SyJson

import matplotlib.pyplot as plt
import datetime
from dmb.data.bose_hubbard_2d.cpp_worm.worm.parameters import WormInputParameters
from dmb.data.bose_hubbard_2d.cpp_worm.worm.outputs import WormOutput
import matplotlib.pyplot as plt
from dmb.data.bose_hubbard_2d.cpp_worm.worm.ac import GammaPathologicalError
import time

log = create_logger(__name__)


class SimulationExecution:
    @property
    def executable(self):
        if not hasattr(self, "_executable") or self._executable is None:
            raise RuntimeError("No executable set!")

        return self._executable

    @executable.setter
    def executable(self, executable: Optional[str] = None) -> None:
        if executable is None:
            log.info("No executable set. Won't be able to execute worm calculation.")
            return

        if not Path(executable).is_file():
            raise RuntimeError(f"Executable {executable} does not exist!")

        self._executable = executable

    def execute_worm(
        self,
        input_file: Optional[Path] = None,
        num_restarts: int = 1,
    ):
        for run_idx in range(num_restarts):
            try:
                self.execute_worm_single_try(input_file=input_file)
            except subprocess.CalledProcessError as e:
                log.error(f"Restarting worm calculation... run {run_idx} failed!")

                if run_idx == num_restarts - 1:
                    raise RuntimeError(
                        f"Worm calculation failed {num_restarts} times. Aborting."
                    )
                continue
            else:
                break

    def execute_worm_continue(
        self,
        num_restarts: int = 1,
    ):
        self.execute_worm(
            input_file=self.input_parameters.checkpoint,
            num_restarts=num_restarts,
        )

    def execute_worm_single_try(self, input_file: Optional[Path] = None):
        if input_file is None:
            input_file = self.input_parameters.ini_path

        pipeout_dir = self.save_dir / "pipe_out"
        pipeout_dir.mkdir(parents=True, exist_ok=True)

        # determine scheduler
        if check_if_slurm_is_installed_and_running():
            write_sbatch_script(
                script_path=self.save_dir / "run.sh",
                worm_executable_path=self.executable,
                parameters_path=input_file,
                pipeout_dir=pipeout_dir,
            )

            # submit job
            call_sbatch_and_wait(script_path=self.save_dir / "run.sh")

        else:
            env = os.environ.copy()
            env["TMPDIR"] = "/tmp"

            p = subprocess.run(
                [
                    "mpirun",
                    "--use-hwthread-cpus",
                    str(self.executable),
                    str(input_file),
                ],
                env=env,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=False,
            )

            # write output to file
            now = datetime.datetime.now()
            with open(pipeout_dir / f"stdout_{now}.txt", "w") as f:
                f.write(p.stdout.decode("utf-8"))
            with open(pipeout_dir / f"stderr_{now}.txt", "w") as f:
                f.write(p.stderr.decode("utf-8"))

            # if job failed, raise error
            if p.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=p.returncode,
                    cmd=p.args,
                    output=p.stdout,
                    stderr=p.stderr,
                )


class SimulationResult:
    @property
    def results(self):
        if self.input_parameters.outputfile_relative is None:
            out_file_path = (
                REPO_DATA_ROOT / self.input_parameters.outputfile.split("data/")[-1]
            )
            log.debug(f"Using default output file path: {out_file_path}")
        else:
            out_file_path = self.save_dir / self.input_parameters.outputfile_relative

        output = WormOutput(
            out_file_path=out_file_path, input_parameters=self.input_parameters
        )

        return output

    @property
    def convergence_stats(self):
        RELATIVE_THRESHOLD: float = 0.01
        ABSOLUTE_THRESHOLD: float = 0.01
        ERROR_STD_THRESHOLD: float = 0.01
        TAU_THRESHOLD: float = 5

        observables = self.results.vector_observables

        stats_dict = defaultdict(dict)
        converged_dict = defaultdict(dict)
        for obs in observables:
            converged_dict[obs]["absolute_error"] = (
                observables[obs]["mean"]["error"] < ABSOLUTE_THRESHOLD
            ).all()
            stats_dict[obs]["absolute_error"] = float(
                (observables[obs]["mean"]["error"]).max()
            )

            stats_dict[obs]["error_std"] = float(
                (observables[obs]["mean"]["error"]).std()
            )
            stats_dict[obs]["num_samples"] = int(observables[obs]["count"])

            # # get max tau without nans
            if np.isnan(observables[obs]["tau"]).all():
                tau_max = np.nan
                log.debug("All tau values are nan")
            else:
                tau_max = np.nanmax(observables[obs]["tau"])

            converged_dict[obs]["tau_max"] = tau_max < TAU_THRESHOLD
            stats_dict[obs]["tau_max"] = float(tau_max)

        # get tau with ulli wolff method
        try:
            density_errors = self.results.density_errors
            stats_dict["uw_tau_max_density"] = float(np.max(density_errors.tau_int))
            stats_dict["uw_dmax_density"] = float(np.max(density_errors.dvalue))
        except GammaPathologicalError as e:
            log.info(e)
            stats_dict["uw_tau_max_density"] = -1
            stats_dict["uw_dmax_density"] = -1

        return stats_dict

    @property
    def max_density_error(self):
        return float(np.max(self.results.density_errors.dvalue))

    @property
    def uncorrected_max_density_error(self):
        return float(np.max(self.results.densities.std(axis=0)))

    @property
    def max_tau_int(self):
        return float(np.max(self.results.density_errors.tau_int))


class WormSimulation(SimulationExecution, SimulationResult):
    """Class to manage worm simulations."""

    def __init__(
        self,
        input_parameters: WormInputParameters,
        save_dir: Path,
        worm_executable: Optional[Path] = None,
    ):
        self.input_parameters = input_parameters
        self.executable = worm_executable
        self.save_dir = save_dir

        self.record = SyJson(path=save_dir / "record.json")

    @classmethod
    def from_dir(cls, dir_path: Path, worm_executable: Optional[Path] = None):
        # Read in parameters
        input_parameters = WormInputParameters.from_dir(save_dir_path=dir_path)

        # Create simulation
        return cls(
            input_parameters=input_parameters,
            save_dir=dir_path,
            worm_executable=worm_executable,
        )

    def plot_observables(self):
        """
        Plot the results of the worm calculation.
        """

        inputs = self.input_parameters.mu
        outputs = self.results.vector_observables

        for obs_idx, (obs, obs_dict) in enumerate(outputs.items()):
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            plt.subplots_adjust(wspace=0.5)

            value_plot = ax[0].imshow(
                obs_dict["mean"]["value"].reshape(
                    int(np.sqrt(len(obs_dict["mean"]["value"]))),
                    -1,
                )
            )
            ax[0].set_title(obs)
            fig.colorbar(value_plot, ax=ax[0])

            error_plot = ax[1].imshow(
                obs_dict["mean"]["error"].reshape(
                    int(np.sqrt(len(obs_dict["mean"]["error"]))),
                    -1,
                )
            )
            ax[1].set_title("Error")
            fig.colorbar(error_plot, ax=ax[1])

            chem_pot_plot = ax[2].imshow(
                inputs.reshape(self.input_parameters.Lx, self.input_parameters.Ly)
            )
            ax[2].set_title("Chemical Potential")
            fig.colorbar(chem_pot_plot, ax=ax[2])

            for a in ax:
                a.set_xticks([])
                a.set_yticks([])

            # save figure. append current time formatted to avoid overwriting
            # plots dir
            plots_dir = (self.save_dir / "plots") / obs
            plots_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d_%H-%M-%S")
            fig.savefig(plots_dir / f"{obs}_{now}.png", dpi=150)
            plt.close()

    def save_parameters(self):
        self.input_parameters.save(save_dir_path=self.save_dir)
        self.input_parameters.save_h5()

    def set_extension_sweeps_in_checkpoints(self, extension_sweeps: int):
        for checkpoint_file in self.save_dir.glob("checkpoint.h5*"):
            with h5py.File(checkpoint_file, "r+") as f:
                try:
                    f["parameters/extension_sweeps"][...] = extension_sweeps
                except KeyError:
                    f["parameters/extension_sweeps"] = extension_sweeps

    @property
    def tune_simulation(self):
        tune_dir = self.save_dir / "tune"
        tune_dir.mkdir(parents=True, exist_ok=True)

        try:
            tune_simulation = WormSimulation.from_dir(
                dir_path=tune_dir, worm_executable=self.executable
            )
        except FileNotFoundError:
            tune_simulation = WormSimulation(
                input_parameters=deepcopy(self.input_parameters),
                save_dir=tune_dir,
                worm_executable=self.executable,
            )
        return tune_simulation


class WormSimulationRunner:
    def __init__(self, worm_simulation: WormSimulation):
        self.worm_simulation = worm_simulation

    def run(self, num_restarts: int = 1):
        self.worm_simulation.save_parameters()
        self.worm_simulation.execute_worm(num_restarts=num_restarts)

    def run_continue(self, num_restarts: int = 1):
        self.worm_simulation.execute_worm_continue(num_restarts=num_restarts)

    def run_iterative_until_converged(
        self,
        max_num_measurements_per_nmeasure2: int = 100000,
        min_num_measurements_per_nmeasure2: int = 1000,
        num_sweep_increments: int = 25,
        sweeps_to_thermalization_ratio: int = 10,
        max_abs_error_threshold: int = 0.01,
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
        elapsed_time = 0
        for step_idx, num_sweeps in pbar:
            self.worm_simulation.set_extension_sweeps_in_checkpoints(
                extension_sweeps=num_sweeps
            )

            # execute worm
            start_time = time.perf_counter()

            if step_idx > 0:
                self.worm_simulation.execute_worm_continue(
                    num_restarts=num_restarts,
                )
            else:
                self.worm_simulation.execute_worm(num_restarts=num_restarts)

            elapsed_time += time.perf_counter() - start_time

            # get current error
            error = self.worm_simulation.max_density_error
            pbar.set_description(
                f"Current error: {error}. Sweeps: {num_sweeps}. Num Nmeasure2: {num_sweeps/self.worm_simulation.input_parameters.Nmeasure2}. At step {step_idx} of {len(num_sweeps_values)}. Expected required number of measurements: {expected_required_num_measurements}"
            )

            self.worm_simulation.plot_observables()
            self.worm_simulation.record["steps"].append(
                {
                    "sweeps": int(num_sweeps),
                    "error": float(error),
                    "elapsed_time": float(elapsed_time),
                }
            )

            # if error is below threshold, break
            if error < max_abs_error_threshold:
                break

    def tune_nmeasure2(
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
                tune_runner.run()
            except RuntimeError as e:
                continue

            stats_dict = tune_simulation.convergence_stats

            tune_simulation.record["steps"].append(
                {
                    "sweeps": sweeps,
                    "thermalization": thermalization,
                    "Nmeasure2": Nmeasure2,
                    **stats_dict,
                }
            )

            tune_simulation.plot_observables()

            # print all tau values
            for obs in tune_simulation.results.vector_observables:
                plt.plot(
                    [step[obs]["tau_max"] for step in tune_simulation.record["steps"]]
                )
                plt.title(obs)
                plt.yscale("log")
                plt.savefig(tune_simulation.save_dir / f"tau_{obs}.png")
                plt.close()

            plt.plot(
                [step["uw_tau_max_density"] for step in tune_simulation.record["steps"]]
            )
            plt.yscale("log")
            plt.savefig(tune_simulation.save_dir / f"tau_max_uw_density.png")
            plt.close()

            tau_max_values = np.array(
                [step["uw_tau_max_density"] for step in tune_simulation.record["steps"]]
            )

            if (tau_max_values[-1:] < tau_threshold).all() and (
                tau_max_values[-1:] > 0
            ).all():
                break

            # get biggest smaller 2**x value
            if tau_max_values[-1] > 0:
                skip_next_counter = int(np.log2(tau_max_values[-2:].min())) - 2

        self.worm_simulation.input_parameters.Nmeasure2 = Nmeasure2
        self.worm_simulation.input_parameters.Nmeasure = Nmeasure
        self.worm_simulation.save_parameters()

        # save uncorrected density error
        tune_simulation.record[
            "uncorrected_max_density_error"
        ] = tune_simulation.uncorrected_max_density_error
