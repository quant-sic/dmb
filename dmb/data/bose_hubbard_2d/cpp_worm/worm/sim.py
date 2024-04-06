import datetime
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np


from dmb.data.bose_hubbard_2d.cpp_worm.worm.observables import SimulationObservables
from dmb.data.bose_hubbard_2d.cpp_worm.worm.outputs import WormOutput
from dmb.data.bose_hubbard_2d.cpp_worm.worm.parameters import WormInputParameters
from dmb.utils import REPO_DATA_ROOT, create_logger
from dmb.utils.syjson import SyJson
import logging
from dmb.data.bose_hubbard_2d.cpp_worm.worm.dispatching import AutoDispatcher

__all__ = [
    "WormSimulation",
    "plot_sim_observables",
    "plot_sim_inputs",
    "plot_sim_phase_diagram_inputs",
]

log = create_logger(__name__)


class _SimulationExecution:
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

    @property
    def dispatcher(self):
        return AutoDispatcher()

    async def execute_worm(
        self,
        input_file_path: Optional[Path] = None,
    ):
        self.file_logger.info(
            f"""Running simulation with:
            sweeps: {self.input_parameters.sweeps},
            Nmeasure2: {self.input_parameters.Nmeasure2},
            Nmeasure: {self.input_parameters.Nmeasure},
            thermalization: {self.input_parameters.thermalization}
            Executable: {self.executable}
            Input file Path: {input_file_path}
            Extension sweeps: {self.get_extension_sweeps_from_checkpoints()}
            """
        )

        try:
            await self.dispatcher.dispatch(
                task=[
                    "mpirun",
                    "--use-hwthread-cpus",
                    str(self.executable),
                    str(input_file_path or self.input_parameters.ini_path),
                ],
                job_name="worm",
                work_directory=self.save_dir,
                pipeout_dir=self.save_dir / "pipe_out",
                timeout=60 * 60 * 24,
            )
        except subprocess.CalledProcessError as e:
            self.file_logger.error(
                f"Worm calculation failed with error code {e.returncode}."
            )

    async def execute_worm_continue(
        self,
    ):
        await self.execute_worm(
            input_file_path=self.input_parameters.checkpoint,
        )


class _SimulationResult:
    @property
    def output(self):
        if self.input_parameters.outputfile_relative is None:
            out_file_path = (
                REPO_DATA_ROOT / self.input_parameters.outputfile.split("data/")[-1]
            )
            self.file_logger.debug(f"Using default output file path: {out_file_path}")
        else:
            out_file_path = self.save_dir / self.input_parameters.outputfile_relative

        _output = WormOutput(
            out_file_path=out_file_path,
            input_parameters=self.input_parameters,
            logging_instance=self.file_logger,
        )

        return _output

    @property
    def observables(self):
        return SimulationObservables(output=self.output)

    @property
    def max_tau_int(self) -> float | None:
        tau_int = self.observables.get_error_analysis("primary", "density")["tau_int"]
        return float(np.max(tau_int)) if tau_int is not None else None

    @property
    def uncorrected_max_density_error(self) -> float | None:
        naive_error = self.observables.get_error_analysis("primary", "density")[
            "naive_error"
        ]
        return float(np.max(naive_error)) if naive_error is not None else None

    @property
    def max_density_error(self) -> float | None:
        error = self.observables.get_error_analysis("primary", "density")["error"]
        return float(np.max(error)) if error is not None else None

    def plot_observables(self, observable_names: list[str] = ["density"]):
        """
        Plot the results of the worm calculation.
        """

        inputs = self.input_parameters.mu

        for obs in observable_names:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            plt.subplots_adjust(wspace=0.5)

            error_analysis = self.observables.get_error_analysis("primary", obs)

            value_plot = ax[0].imshow(error_analysis["expectation_value"])
            ax[0].set_title(obs)
            fig.colorbar(value_plot, ax=ax[0])

            error_plot = ax[1].imshow(error_analysis["error"])
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


class WormSimulation(_SimulationExecution, _SimulationResult):
    """Class to manage worm simulations."""

    def __init__(
        self,
        input_parameters: WormInputParameters,
        save_dir: Path,
        worm_executable: Optional[Path] = None,
        reloaded_from_dir: bool = False,
    ):
        self.input_parameters = input_parameters
        self.executable = worm_executable
        self.save_dir = save_dir

        self.record = SyJson(path=save_dir / "record.json")

        if not reloaded_from_dir:
            self.save_parameters()

        self.file_logger = create_logger(
            app_name=(
                f"worm_simulation_{self.save_dir.name}"
                if self.save_dir.name != "tune"
                else f"worm_simulation_{self.save_dir.parent.name}_{self.save_dir.name}"
            ),
            level=logging.INFO,
            file=self.save_dir / "log.txt",
        )

        if not reloaded_from_dir:
            self.file_logger.info(f"Initialized worm simulation in {self.save_dir}")

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
            reloaded_from_dir=True,
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

    def get_extension_sweeps_from_checkpoints(self):
        extension_sweeps = None
        for checkpoint_file in self.save_dir.glob("checkpoint.h5*"):
            with h5py.File(checkpoint_file, "r") as f:
                try:
                    extension_sweeps = f["parameters/extension_sweeps"][()]
                except KeyError:
                    pass
        return extension_sweeps

    @staticmethod
    def get_plot_dir(save_dir: Path):
        plot_dir = save_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir
