"""Module to manage worm simulations."""

from __future__ import annotations

import abc
import datetime
import logging
import os
from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field
from syjson import SyJson

from dmb.data.dispatching import Dispatcher, ReturnCode
from dmb.logging import create_logger

from .observables import SimulationObservables
from .output import Output, WormOutput
from .parameters import WormInputParameters

__all__ = ["WormSimulation"]

log = create_logger(__name__)


class WormSimulationInterface(metaclass=abc.ABCMeta):
    """Interface for the worm simulation."""

    @property
    @abc.abstractmethod
    def file_logger(self) -> logging.Logger:
        """Return the logger of the worm simulation."""

    @file_logger.setter
    @abc.abstractmethod
    def file_logger(self, file_logger: logging.Logger) -> None:
        """Set the logger of the worm simulation."""

    @property
    @abc.abstractmethod
    def record(self) -> SyJson:
        """Return the record of the worm simulation."""

    @record.setter
    @abc.abstractmethod
    def record(self, record: SyJson) -> None:
        """Set the record of the worm simulation."""

    @property
    @abc.abstractmethod
    def input_parameters(self) -> WormInputParameters:
        """Return the input parameters of the worm simulation."""

    @input_parameters.setter
    @abc.abstractmethod
    def input_parameters(self, input_parameters: WormInputParameters) -> None:
        """Set the input parameters of the worm simulation."""

    @property
    @abc.abstractmethod
    def tune_simulation(self) -> WormSimulationInterface:
        """Return the tune simulation."""

    @abc.abstractmethod
    async def execute_worm(self, input_file_path: Path | None = None) -> ReturnCode:
        """Execute the worm simulation.

        Args:
            input_file_path: Path to the input file (.h5,.ini).
                If None, the input parameters file will be used.
                Defaults to None.
        """

    @abc.abstractmethod
    async def execute_worm_continue(self) -> ReturnCode:
        """Execute the worm simulation from the last checkpoint."""

    @property
    @abc.abstractmethod
    def output(self) -> Output:
        """Return the worm simulation output."""

    @property
    @abc.abstractmethod
    def observables(self) -> SimulationObservables:
        """Return the observables of the worm simulation."""

    @property
    @abc.abstractmethod
    def max_tau_int(self) -> float | None:
        """Return the maximum integrated autocorrelation time."""

    @property
    @abc.abstractmethod
    def uncorrected_max_density_error(self) -> float | None:
        """Return the maximum uncorrected density error."""

    @property
    @abc.abstractmethod
    def max_density_error(self) -> float | None:
        """Return the maximum density error."""

    @property
    @abc.abstractmethod
    def valid(self) -> bool:
        """Return True if the simulation is valid."""

    @abc.abstractmethod
    def plot_observables(
        self, observable_names: dict[str, list[str]] | None = None
    ) -> None:
        """Plot the observables of the worm simulation."""

    @abc.abstractmethod
    def plot_inputs(self) -> None:
        """Plot the input parameters of the worm simulation."""

    @abc.abstractmethod
    def plot_phase_diagram_inputs(self) -> None:
        """Plot the input parameters of the worm simulation."""

    @abc.abstractmethod
    def save_parameters(self) -> None:
        """Save the input parameters of the worm simulation."""

    @abc.abstractmethod
    def set_extension_sweeps_in_checkpoints(self, extension_sweeps: int) -> None:
        """Set the extension sweeps in the checkpoint files."""


@define
class WormSimulation(WormSimulationInterface):
    """Class to manage worm simulations.

    The class provides methods to execute worm simulations and
    analyze the results. The simulation data is saved in the ``save_dir``.
    A record of the simulation steps is saved in the ``save_dir/record.json``
    file. Logs are saved in the ``save_dir/log.txt`` file.

    Attributes:
        input_parameters: Input parameters for the worm simulation.
        save_dir: Directory to save the simulation data.
        executable: Path to the worm executable.
        dispatcher: Dispatcher instance to run the worm simulation.
        reloaded_from_dir: Flag to indicate if the simulation
            was reloaded from a directory.
    """

    input_parameters: WormInputParameters
    save_dir: Path
    executable: Path | None = None
    dispatcher: Dispatcher | None = None
    reloaded_from_dir: bool = False

    file_logger: logging.Logger = field(init=False)
    record: SyJson = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.record = SyJson(path=self.save_dir / "record.json")

        if "steps" not in self.record:
            self.record["steps"] = []

        self.file_logger = create_logger(
            app_name=(
                f"worm_simulation_{self.save_dir.name}"
                if self.save_dir.name != "tune"
                else f"worm_simulation_{self.save_dir.parent.name}_{self.save_dir.name}"
            ),
            level=logging.INFO,
            file=self.save_dir / "log.txt",
        )

        if not self.reloaded_from_dir:
            self.save_parameters()

        self.file_logger.debug(f"Initialized worm simulation in {self.save_dir}\n\n")

    @classmethod
    def from_dir(
        cls,
        dir_path: Path,
        dispatcher: Dispatcher | None = None,
        executable: Path | None = None,
    ) -> WormSimulation:
        """Create a worm simulation from a directory.

        Args:
            dir_path: Path to the directory containing the simulation data.
            dispatcher: Dispatcher instance to run the worm simulation.
                Defaults to None.
            executable: Path to the worm executable.
                Defaults to None.

        Returns:
            WormSimulation: Worm simulation instance.
        """
        input_parameters = WormInputParameters.from_dir(save_dir_path=dir_path)
        return cls(
            input_parameters=input_parameters,
            save_dir=dir_path,
            executable=executable,
            dispatcher=dispatcher,
        )

    @staticmethod
    def get_tune_dir_path(save_dir: Path) -> Path:
        """Return the path to the tune directory."""

        return save_dir / "tune"

    @staticmethod
    def get_plot_dir_path(save_dir: Path) -> Path:
        """Return the path to the plot directory."""

        plot_dir = save_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

    @property
    def tune_simulation(self) -> WormSimulation:
        """Return the tune simulation."""

        tune_dir = self.get_tune_dir_path(save_dir=self.save_dir)
        tune_dir.mkdir(parents=True, exist_ok=True)

        try:
            tune_simulation = WormSimulation.from_dir(
                dir_path=tune_dir,
                executable=self.executable,
                dispatcher=self.dispatcher,
            )
        except FileNotFoundError:
            tune_simulation = WormSimulation(
                input_parameters=deepcopy(self.input_parameters),
                save_dir=tune_dir,
                executable=self.executable,
                dispatcher=self.dispatcher,
            )
        return tune_simulation

    def save_parameters(self) -> None:
        """Save the input parameters of the worm simulation."""

        self.input_parameters.save(save_dir=self.save_dir)

    def set_extension_sweeps_in_checkpoints(self, extension_sweeps: int) -> None:
        """Set the extension sweeps in the checkpoint files."""

        checkpoint_path = self.input_parameters.get_checkpoint_path(self.save_dir)
        for checkpoint_file in checkpoint_path.parent.glob(f"{checkpoint_path.name}*"):
            with h5py.File(checkpoint_file, "r+") as f:
                try:
                    f["parameters/extension_sweeps"][...] = extension_sweeps
                except KeyError:
                    f["parameters/extension_sweeps"] = extension_sweeps

    def get_extension_sweeps_from_checkpoints(self) -> int | None:
        """Get the extension sweeps from the checkpoint files."""

        extension_sweeps = None
        checkpoint_path = self.input_parameters.get_checkpoint_path(self.save_dir)
        for checkpoint_file in checkpoint_path.parent.glob(f"{checkpoint_path.name}*"):
            with h5py.File(checkpoint_file, "r") as f:
                try:
                    extension_sweeps = f["parameters/extension_sweeps"][()]
                except KeyError:
                    pass
        return extension_sweeps

    async def execute_worm(
        self,
        input_file_path: Path | None = None,
    ) -> ReturnCode:
        """Execute the worm simulation.

        Args:
            input_file_path: Path to the input file (.h5,.ini).
                If None, the input parameters file will be used.
                Defaults to None.
        """
        if self.executable is None or self.dispatcher is None:
            raise ValueError(
                "To execute the worm simulation, the executable "
                "and dispatcher must be set."
            )

        self.file_logger.info(f"""Running simulation with:
            sweeps: {self.input_parameters.sweeps},
            Nmeasure2: {self.input_parameters.Nmeasure2},
            Nmeasure: {self.input_parameters.Nmeasure},
            thermalization: {self.input_parameters.thermalization}
            Executable: {self.executable}
            Input file Path: {input_file_path}
            Extension sweeps: {self.get_extension_sweeps_from_checkpoints()}\n
            """)

        return_code = await self.dispatcher.dispatch(
            task=[
                "mpirun",
                "--use-hwthread-cpus",
                str(self.executable),
                str(
                    input_file_path or self.input_parameters.get_ini_path(self.save_dir)
                ),
            ],
            job_name=os.environ.get("WORM_JOB_NAME", "worm"),
            work_directory=self.save_dir,
            pipeout_dir=self.save_dir / "pipe_out",
            timeout=60 * 60 * 24 * 7,
        )
        self.file_logger.info(f"Simulation finished with return code: {return_code}")
        return return_code

    async def execute_worm_continue(self) -> ReturnCode:
        return await self.execute_worm(
            input_file_path=self.input_parameters.get_checkpoint_path(self.save_dir)
        )

    @property
    def output(self) -> WormOutput:
        return WormOutput(
            out_file_path=self.input_parameters.get_outputfile_path(self.save_dir),
            input_parameters=self.input_parameters,
            logging_instance=self.file_logger,
        )

    @property
    def observables(self) -> SimulationObservables:
        return SimulationObservables(output=self.output)

    @property
    def max_tau_int(self) -> float | None:
        tau_int = self.observables.get_error_analysis("primary", "density")["tau_int"]
        if tau_int is not None and not np.isnan(tau_int).all():
            return float(np.nanmax(tau_int))

        return None

    @property
    def uncorrected_max_density_error(self) -> float | None:
        naive_error = self.observables.get_error_analysis("primary", "density")[
            "naive_error"
        ]
        if naive_error is not None and not np.isnan(naive_error).all():
            return float(np.nanmax(naive_error))

        return None

    @property
    def max_density_error(self) -> float | None:
        error = self.observables.get_error_analysis("primary", "density")["error"]
        if error is not None and not np.isnan(error).all():
            return float(np.nanmax(error))

        return None

    @property
    def valid(self) -> bool:
        return self.max_density_error is not None

    def plot_observables(
        self, observable_names: dict[str, list[str]] | None = None
    ) -> None:
        """
        Plot the results of the worm calculation.
        """
        if observable_names is None:
            observable_names = {"primary": ["density"]}

        inputs = self.input_parameters.mu

        for _, obs_names in observable_names.items():
            for obs in obs_names:
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                plt.subplots_adjust(wspace=0.5)

                error_analysis = self.observables.get_error_analysis("primary", obs)

                if (
                    expectation_value := error_analysis["expectation_value"]
                ) is not None:
                    if expectation_value.ndim == 2:
                        value_plot = ax[0].imshow(expectation_value)

                        ax[0].set_title(obs)
                        fig.colorbar(value_plot, ax=ax[0])

                if (error := error_analysis["error"]) is not None:
                    if error.ndim == 2:
                        error_plot = ax[1].imshow(error)

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
                plots_dir = self.get_plot_dir_path(self.save_dir) / obs
                plots_dir.mkdir(parents=True, exist_ok=True)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fig.savefig(plots_dir / f"{obs}_{now}.png", dpi=150)
                plt.close()

    def plot_inputs(self) -> None:
        self.input_parameters.plot_input_parameters(
            plots_dir=self.get_plot_dir_path(self.save_dir)
        )

    def plot_phase_diagram_inputs(self) -> None:
        self.input_parameters.plot_phase_diagram_input_parameters(
            plots_dir=self.get_plot_dir_path(self.save_dir)
        )
