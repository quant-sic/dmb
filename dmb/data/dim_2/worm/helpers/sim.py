import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
from typing import Optional, Union
import os
from tqdm import tqdm
from copy import deepcopy
from .io import create_logger
from collections import defaultdict
import json
from dmb.data.dim_2.helpers import check_if_slurm_is_installed_and_running,write_sbatch_script,call_sbatch_and_wait

log = create_logger(__name__)


@dataclass
class WormInputParameters:
    mu: Union[np.ndarray, float]
    t_hop: Union[np.ndarray, float] = 1.0
    U_on: Union[np.ndarray, float] = 4.0
    V_nn: Union[np.ndarray, float] = 0.0
    model: str = "BoseHubbard"
    runtimelimit: int = 10000
    sweeps: int = 25000
    thermalization: int = 100
    Lx: int = 4
    Ly: int = 4
    Lz: int = 1
    pbcx: int = 1
    pbcy: int = 1
    pbcz: int = 1
    beta: float = 20.0
    nmax: int = 3
    E_off: float = 1.0
    canonical: int = -1
    seed: int = 30
    Ntest: int = 10000000
    Nsave: int = 100000000
    Nmeasure: int = 1
    Nmeasure2: int = 10
    C_worm: float = 2.0
    p_insertworm: float = 1.0
    p_moveworm: float = 0.3
    p_insertkink: float = 0.2
    p_deletekink: float = 0.2
    p_glueworm: float = 0.3

    h5_path: Optional[Path] = None
    checkpoint: Optional[Path] = None
    outputfile: Optional[Path] = None

    @classmethod
    def from_dir(cls, save_dir_path: Path):
        # Read ini file
        with open(save_dir_path / "parameters.ini", "r") as f:
            lines = f.readlines()

        # Fill dictionary for ini parameters
        params = {}
        for line in lines:
            if not line.startswith("#"):
                key, value = map(lambda s: s.strip(), line.split("="))

                if key in cls.__dataclass_fields__.keys():
                    params[key] = value

        # read in h5 site dependent arrays
        with h5py.File(save_dir_path / "parameters.h5", "r") as file:
            for name in ("mu", "t_hop", "U_on", "V_nn"):
                params[name] = file[f"/{name}"][()]

        # Create input parameters
        return cls(**params)

    def save_h5(self):
        if self.h5_path is None:
            raise RuntimeError("h5_path must be set")

        # create parent directory if it does not exist
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)

        # Create h5 file
        with h5py.File(self.h5_path, "w") as file:
            for name, attribute in (
                ("mu", self.mu),
                ("t_hop", self.t_hop),
                ("U_on", self.U_on),
                ("V_nn", self.V_nn),
            ):
                file[f"/{name}"] = (
                    attribute if isinstance(attribute, float) else attribute.flatten()
                )

    @property
    def ini_path(self):
        if self._ini_path is None:
            raise RuntimeError(
                "ini_path must be set. By saving the parameters to a directory, the ini_path is set automatically."
            )
        else:
            return self._ini_path

    @ini_path.setter
    def ini_path(self, ini_path: Path):
        self._ini_path = ini_path

    def to_ini(self, checkpoint, outputfile, save_path: Path):
        # create parent directory if it does not exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create ini file
        with open(save_path, "w") as f:
            for key in self.__dataclass_fields__.keys():
                if not (
                    key in ("mu", "t_hop", "U_on", "V_nn")
                    and isinstance(self.__getattribute__(key), np.ndarray)
                ):
                    f.write(f"{key} = {self.__getattribute__(key)}\n")

            if self.h5_path is None:
                raise RuntimeError("h5_path must be set")
            else:
                f.write(f"site_arrays = {self.h5_path}\n")

    def save(
        self,
        save_dir_path: Path,
        checkpoint: Optional[Path] = None,
        outputfile: Optional[Path] = None,
    ):
        # create parent directory if it does not exist
        save_dir_path.parent.mkdir(parents=True, exist_ok=True)

        self.outputfile = (
            save_dir_path / "output.h5" if outputfile is None else outputfile
        )
        self.checkpoint = (
            save_dir_path / "checkpoint.h5" if checkpoint is None else checkpoint
        )

        self.h5_path = save_dir_path / "parameters.h5"
        self.ini_path = save_dir_path / "parameters.ini"

        # Create ini file
        self.to_ini(
            save_path=self.ini_path,
            checkpoint=self.checkpoint,
            outputfile=self.outputfile,
        )
        self.save_h5()


@dataclass
class WormOutput:
    out_file_path: Path

    @property
    def observables(self):
        h5_file = h5py.File(self.out_file_path, "r")

        observables_dict = {}

        observables_dict = defaultdict(dict)

        for obs, obs_dataset in h5_file["simulation/results"].items():
            for measure, value in obs_dataset.items():
                if isinstance(value, h5py.Dataset):
                    observables_dict[obs][measure] = value[()]

                elif isinstance(value, h5py.Group):
                    observables_dict[obs][measure] = {}
                    for sub_measure, sub_value in value.items():
                        observables_dict[obs][measure][sub_measure] = sub_value[()]

        return observables_dict

class SimulationRecord(object):

    def __init__(self, record_dir: Path):
        
        self.record_file_path = record_dir / "record.json"

        if self.record_file_path.exists():
            with open(self.record_file_path, "r") as f:
                self.record = json.load(f)
        else:
            self.record = {}

    def save(self):

        with open(self.record_file_path, "w") as f:
            json.dump(self.record, f)
            
    def update(self, record: dict):
        self.record.update(record)

        self.save()

    def __getitem__(self, key: str):
        return self.record.get(key, None)
    
    def __setitem__(self, key: str, value):
        self.record[key] = value

        self.save()

    
class WormSimulation(object):
    def __init__(
        self,
        input_parameters: WormInputParameters,
        save_dir: Path,
    ):
        self.input_parameters = input_parameters
        self.save_dir = save_dir

        self.record = SimulationRecord(record_dir=save_dir)


    @classmethod
    def from_dir(cls, dir_path: Path):
        # Read in parameters
        input_parameters = WormInputParameters.from_dir(save_dir_path=dir_path)

        # Create simulation
        return cls(
            input_parameters=input_parameters,
            save_dir=dir_path,
        )

    def _save_parameters(self, save_dir: Path):
        self.input_parameters.save(save_dir_path=save_dir)
        self.input_parameters.save_h5()

    def save_parameters(self):
        self._save_parameters(save_dir=self.save_dir)

    def _execute_worm(self,executable,input_file:Optional[Path]=None):

        if input_file is None:
            input_file = self.input_parameters.ini_path

        # determine scheduler
        if check_if_slurm_is_installed_and_running():
            write_sbatch_script(script_path=self.save_dir / "run.sh",worm_executable_path=executable,parameters_path=input_file,pipeout_dir=self.save_dir)

            # submit job
            call_sbatch_and_wait(script_path=self.save_dir / "run.sh")

        else:
            try:
                env = os.environ.copy()
                env["TMPDIR"] = "/tmp"

                p = subprocess.run(
                    ["mpirun","--use-hwthread-cpus",str(executable), str(input_file)],
                    env=env,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    check=True,
                )

            except subprocess.CalledProcessError as e:
                log.error(e.stderr.decode("utf-8"))
                raise e


    @property
    def results(self):
        try:
            output = WormOutput(out_file_path=self.input_parameters.outputfile)
        except OSError:
            output = None

        return output

    def check_convergence(self, results, error_threshold: float = 0.01):

        rel_dens_error = (
            self.results.observables["Density_Distribution"]["mean"]["error"]
            / self.results.observables["Density_Distribution"]["mean"]["value"]
        )

        converged = (rel_dens_error < error_threshold).all()
        max_rel_error = rel_dens_error.max()

        n_measurements = results.observables["Density_Distribution"]["count"]

        # get max tau without nans
        tau_max = np.nanmax(results.observables["Density_Distribution"]["tau"])

        return converged, max_rel_error, n_measurements, tau_max

    def _set_extension_sweeps_in_checkpoints(self, extension_sweeps: int):
        for checkpoint_file in self.save_dir.glob("checkpoint.h5*"):
            with h5py.File(checkpoint_file, "r+") as f:
                try:
                    f["parameters/extension_sweeps"][...] = extension_sweeps
                except KeyError:
                    f["parameters/extension_sweeps"] = extension_sweeps
        

    def run_until_convergence(self,executable, max_sweeps: int = 10**8, tune: bool = True):
        # tune measurement interval
        if tune:
            measure2 = self.tune(executable=executable)
            self.input_parameters.Nmeasure2 = measure2

        self.save_parameters()

        self._execute_worm(input_file=self.input_parameters.ini_path,executable=executable)

        pbar = tqdm(range(10**5, max_sweeps, 5 * 10**5))
        for sweeps in pbar:
            self._set_extension_sweeps_in_checkpoints(extension_sweeps=sweeps)
            self._execute_worm(input_file=self.input_parameters.checkpoint,executable=executable)

            converged, max_rel_error, n_measurements, tau_max = self.check_convergence(
                self.results
            )

            # update tqdm description
            pbar.set_description(
                f"Running {sweeps} sweeps. Max rel error: {max_rel_error:.2e}. Measurements: {n_measurements}. Tau_max: {tau_max:.2e}"
            )

            if converged:
                break

    def tune(self,executable: Path = None):

        self.record["tune"] = {"status": "running"}

        tune_dir = self.save_dir / "tune"
        tune_dir.mkdir(parents=True, exist_ok=True)

        tune_parameters = deepcopy(self.input_parameters)

        # initial thermalization and measurement sweeps
        tune_parameters.thermalization = 10**4
        tune_parameters.sweeps = 5 * 10**4
        tune_parameters.Nmeasure2 = 500
        tune_parameters.save(save_dir_path=tune_dir)

        self._save_parameters(tune_dir)

        log.info("Tuning measurement interval. Running 50000 sweeps.")
        
        self._execute_worm(input_file=tune_parameters.ini_path,executable=executable)

        self.record["tune"]["status"] = "finished"

        converged, max_rel_error, n_measurements, tau_max = self.check_convergence(
            results=WormOutput(out_file_path=tune_parameters.outputfile)
        )

        if np.isnan(tau_max):
            log.warning("Tau_max is nan. Setting new Nmeasure2 to 10.")
            tau_max = 0
        
        new_measure2 = max(int(tune_parameters.Nmeasure2 * (tau_max / 2)), 10)
        log.info(f"New Nmeasure2: {new_measure2}")

        return new_measure2


    
