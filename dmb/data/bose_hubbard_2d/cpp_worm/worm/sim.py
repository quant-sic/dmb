import asyncio
import datetime
import os
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from auto_correlation import GammaPathologicalError, PrimaryAnalysis

from dmb.data.bose_hubbard_2d.cpp_worm.worm.helpers import (
    call_sbatch_and_wait,
    check_if_slurm_is_installed_and_running,
    write_sbatch_script,
)
from dmb.data.bose_hubbard_2d.cpp_worm.worm.observables import SimulationObservables
from dmb.data.bose_hubbard_2d.cpp_worm.worm.outputs import WormOutput
from dmb.data.bose_hubbard_2d.cpp_worm.worm.parameters import WormInputParameters
from dmb.utils import REPO_DATA_ROOT, create_logger
from dmb.utils.syjson import SyJson

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
            log.debug("No executable set. Won't be able to execute worm calculation.")
            return

        if not Path(executable).is_file():
            raise RuntimeError(f"Executable {executable} does not exist!")

        self._executable = executable

    async def execute_worm(
        self,
        input_file: Optional[Path] = None,
        num_restarts: int = 1,
    ):
        errors = []
        for run_idx in range(num_restarts):
            try:
                await self.execute_worm_single_try(input_file=input_file)
            except subprocess.CalledProcessError as e:
                log.error(f"Restarting worm calculation... run {run_idx} failed!")
                errors.append(e)

                if run_idx == num_restarts - 1:
                    raise RuntimeError(
                        f"""Worm calculation failed {num_restarts} times.
                        Errors: {errors}. Aborting."""
                    )
                continue
            else:
                break

    async def execute_worm_continue(
        self,
        num_restarts: int = 1,
    ):
        await self.execute_worm(
            input_file=self.input_parameters.checkpoint,
            num_restarts=num_restarts,
        )

    async def execute_worm_single_try(self, input_file: Optional[Path] = None):
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
            await call_sbatch_and_wait(script_path=self.save_dir / "run.sh")

        else:
            env = os.environ.copy()
            env["TMPDIR"] = "/tmp"

            cmd = [
                "mpirun",
                "--use-hwthread-cpus",
                str(self.executable),
                str(input_file),
            ]
            p = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stderr=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await p.communicate()

            # write output to file
            now = datetime.datetime.now()
            with open(pipeout_dir / f"stdout_{now}.txt", "w") as f:
                f.write(stdout.decode("utf-8"))
            with open(pipeout_dir / f"stderr_{now}.txt", "w") as f:
                f.write(stderr.decode("utf-8"))

            # if job failed, raise error
            if p.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=p.returncode,
                    cmd=cmd,
                    output=stdout,
                    stderr=stderr,
                )


class SimulationResult:
    @property
    def output(self):
        if self.input_parameters.outputfile_relative is None:
            out_file_path = (
                REPO_DATA_ROOT / self.input_parameters.outputfile.split("data/")[-1]
            )
            log.debug(f"Using default output file path: {out_file_path}")
        else:
            out_file_path = self.save_dir / self.input_parameters.outputfile_relative

        _output = WormOutput(
            out_file_path=out_file_path, input_parameters=self.input_parameters
        )

        return _output

    @property
    def density_error_analysis(self):
        analysis = PrimaryAnalysis(
            self.output.densities.reshape(1, *self.output.densities.shape),
            rep_sizes=[len(self.output.densities)],
            name=[
                f"{int(idx/self.input_parameters.Lx)}{idx%self.input_parameters.Lx}"
                for idx in range(self.input_parameters.Lx**2)
            ],
        )
        analysis.mean()
        try:
            results = analysis.errors()
        except GammaPathologicalError as e:
            log.warning(f"GammaPathologicalError: {e}")
            results = None

        return results

    @property
    def density_error(self) -> Optional[np.ndarray]:
        return (
            self.density_error_analysis.dvalue
            if self.density_error_analysis is not None
            else None
        )

    @property
    def density_variance(self) -> Optional[np.ndarray]:
        return (
            np.var(self.output.densities, axis=0)
            if self.densities is not None
            else None
        )

    @property
    def max_density_error(self) -> Optional[float]:
        return (
            float(np.max(self.density_error_analysis.dvalue))
            if self.density_error_analysis is not None
            else None
        )

    @property
    def uncorrected_max_density_error(self) -> float:
        return float(np.max(self.output.densities.std(axis=0)))

    @property
    def max_tau_int(self) -> float:
        return (
            float(np.max(self.density_error_analysis.tau_int))
            if self.density_error_analysis is not None
            else -1
        )

    @property
    def observables(self):
        return SimulationObservables(output=self.output)

    @property
    def convergence_stats(self):
        ABSOLUTE_THRESHOLD: float = 0.01
        TAU_THRESHOLD: float = 5

        observables = self.output.accumulator_vector_observables

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
            stats_dict["uw_tau_max_density"] = self.max_tau_int
            stats_dict["uw_dmax_density"] = self.max_density_error
        except GammaPathologicalError as e:
            log.info(e)
            stats_dict["uw_tau_max_density"] = -1
            stats_dict["uw_dmax_density"] = -1

        return stats_dict


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
        self.save_parameters()

    @classmethod
    def from_dir(
        cls,
        dir_path: Path,
        worm_executable: Optional[Path] = None,
        check_tune_dir: bool = True,
    ):
        try:
            # Read in parameters
            input_parameters = WormInputParameters.from_dir(save_dir_path=dir_path)
        except FileNotFoundError:
            if check_tune_dir:
                tune_dir = cls.get_tune_dir(save_dir=dir_path)
                if tune_dir.is_dir():
                    log.info(f"Parameters are loaded from tune_dir {tune_dir}.")
                    loaded_simulation = cls.from_dir(
                        dir_path=tune_dir,
                        worm_executable=worm_executable,
                        check_tune_dir=False,
                    )
                    loaded_simulation.input_parameters.save_parameters(
                        save_dir_path=dir_path
                    )
                    loaded_simulation.save_dir = dir_path
                    return loaded_simulation

            raise FileNotFoundError(f"Could not find input parameters in {dir_path}.")

        # Create simulation
        return cls(
            input_parameters=input_parameters,
            save_dir=dir_path,
            worm_executable=worm_executable,
        )

    @staticmethod
    def get_tune_dir(save_dir: Path):
        return save_dir / "tune"

    @property
    def tune_simulation(self):
        tune_dir = self.get_tune_dir(save_dir=self.save_dir)
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

    def save_parameters(self):
        self.input_parameters.save_parameters(save_dir_path=self.save_dir)

    def set_extension_sweeps_in_checkpoints(self, extension_sweeps: int):
        for checkpoint_file in self.save_dir.glob("checkpoint.h5*"):
            with h5py.File(checkpoint_file, "r+") as f:
                try:
                    f["parameters/extension_sweeps"][...] = extension_sweeps
                except KeyError:
                    f["parameters/extension_sweeps"] = extension_sweeps

    @staticmethod
    def get_plot_dir(save_dir: Path):
        plot_dir = save_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

    def plot_observables(self):
        """
        Plot the results of the worm calculation.
        """

        inputs = self.input_parameters.mu
        outputs = self.output.accumulator_vector_observables

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
            plots_dir = self.get_plot_dir(self.save_dir) / obs
            plots_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d_%H-%M-%S")
            fig.savefig(plots_dir / f"{obs}_{now}.png", dpi=150)
            plt.close()

    def plot_inputs(self):
        self.input_parameters.plot_inputs(plots_dir=self.get_plot_dir(self.save_dir))

    def plot_phase_diagram_inputs(self):
        self.input_parameters.plot_phase_diagram_inputs(
            plots_dir=self.get_plot_dir(self.save_dir)
        )
