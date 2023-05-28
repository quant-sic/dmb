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
from dmb.utils import REPO_DATA_ROOT
from dmb.utils.syjson import SyJson
import shutil
import time

log = create_logger(__name__)


@dataclass
class WormInputParameters:
    mu: Union[np.ndarray, float]
    t_hop: Union[np.ndarray, float] = 1.0
    U_on: Union[np.ndarray, float] = 4.0
    V_nn: Union[np.ndarray, float] = 0.0
    model: str = "BoseHubbard"
    runtimelimit: int = 24*60*60
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

    h5_path_relative: Optional[Path] = None
    checkpoint_relative: Optional[Path] = None
    outputfile_relative: Optional[Path] = None

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
        self.outputfile_relative = Path("output.h5")

        self.checkpoint = (
            save_dir_path / "checkpoint.h5" if checkpoint is None else checkpoint
        )
        self.checkpoint_relative = Path("checkpoint.h5")

        self.h5_path = save_dir_path / "parameters.h5"
        self.ini_path = save_dir_path / "parameters.ini"

        self.h5_path_relative = Path("parameters.h5")
        

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

        if not self.out_file_path.exists():
            return None
        
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

        self.record = SyJson(path=save_dir/"record.json")#SimulationRecord(record_dir=save_dir)


    @classmethod
    def from_dir(cls, dir_path: Path):
        # Read in parameters
        input_parameters = WormInputParameters.from_dir(save_dir_path=dir_path)

        # Create simulation
        return cls(
            input_parameters=input_parameters,
            save_dir=dir_path,
        )

    @staticmethod
    def _save_parameters(input_parameters:WormInputParameters,save_dir: Path):
        input_parameters.save(save_dir_path=save_dir)
        input_parameters.save_h5()

    def save_parameters(self):
        self._save_parameters(input_parameters=self.input_parameters,save_dir=self.save_dir)

    def _execute_worm(self,executable,input_file:Optional[Path]=None):

        if input_file is None:
            input_file = self.input_parameters.ini_path

        # determine scheduler
        if check_if_slurm_is_installed_and_running():
            write_sbatch_script(script_path=self.save_dir / "run.sh",worm_executable_path=executable,parameters_path=input_file,pipeout_dir=self.save_dir/"pipe_out")

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
                log.error(e.stdout.decode("utf-8"))
                raise e


    @property
    def results(self):

        if self.input_parameters.outputfile_relative is None:
            out_file_path = REPO_DATA_ROOT / self.input_parameters.outputfile.split("data/")[-1]
            log.debug(f"Using default output file path: {out_file_path}")
        else:
            out_file_path = self.save_dir / self.input_parameters.outputfile_relative

        output = WormOutput(out_file_path=out_file_path)

        return output

    def check_convergence(self, results, relative_error_threshold: float = 0.01,absolute_error_threshold: float = 0.01):

        if results.observables is None:
            return False, np.nan, np.nan, np.nan

        # if value is zero, error is nan
        # if not (results.observables["Density_Distribution"]["mean"]["value"] == 0).any():
            
        #     rel_dens_error = (
        #         results.observables["Density_Distribution"]["mean"]["error"]
        #         / results.observables["Density_Distribution"]["mean"]["value"]
        #     )

        #     converged = (rel_dens_error < relative_error_threshold).all()
        #     max_error = rel_dens_error.max()
        # else:

        log.debug("Density is zero, resorting to absolute error")
        abs_dens_error = results.observables["Density_Distribution"]["mean"]["error"]
        converged = (abs_dens_error < absolute_error_threshold).all()
        max_error = abs_dens_error.max()

        n_measurements = results.observables["Density_Distribution"]["count"]

        # get max tau without nans
        if np.isnan(results.observables["Density_Distribution"]["tau"]).all():
            tau_max = np.nan
            log.debug("All tau values are nan")
        else:
            tau_max = np.nanmax(results.observables["Density_Distribution"]["tau"])

        return converged, max_error, n_measurements, tau_max

    def _set_extension_sweeps_in_checkpoints(self, extension_sweeps: int):
        for checkpoint_file in self.save_dir.glob("checkpoint.h5*"):
            with h5py.File(checkpoint_file, "r+") as f:
                try:
                    f["parameters/extension_sweeps"][...] = extension_sweeps
                except KeyError:
                    f["parameters/extension_sweeps"] = extension_sweeps
        

    def run(self,executable):
        self.save_parameters()
        self._execute_worm(input_file=self.input_parameters.ini_path,executable=executable)
        

    def run_until_convergence(self,executable, tune: bool = True,intermediate_steps=False):
        # tune measurement interval
        if tune:
            measure2,thermalization,sweeps = self.tune(executable=executable)
            self.input_parameters.Nmeasure2 = measure2
            self.input_parameters.thermalization = thermalization
            self.input_parameters.sweeps = sweeps
            self.input_parameters.thermalization = max(int(measure2 * 20),50000)

        self.save_parameters()

        self._execute_worm(input_file=self.input_parameters.ini_path,executable=executable)

        max_multiplier = 250e3
        if intermediate_steps:
            steps = range(self.input_parameters.Nmeasure2 * 100, int(min(max(self.input_parameters.Nmeasure2 * max_multiplier,1e6 + 1 + self.input_parameters.Nmeasure2 * 100),1e8)), int(max(self.input_parameters.Nmeasure2 * 500, 1e6)))
        else:
            steps = [int(min(max(self.input_parameters.Nmeasure2 * max_multiplier,1e6 + 1 + self.input_parameters.Nmeasure2 * 100),1e8))]

        pbar = tqdm(steps, disable=True)
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
            
        self.record["convergence"] = {
            "status": "not converged" if not converged else "converged",
            "max_rel_error": float(max_rel_error),
            "n_measurements": float(n_measurements),
            "tau_max": float(tau_max),
            "sweeps": sweeps,
            "Nmeasure2": self.input_parameters.Nmeasure2,
            "num_steps": len(steps),
        }


    def tune(self,executable: Path = None,ntries: int = 100):

        sweeps = [1000000000,12500000,1000000000,500000000,5000000]
        Nmeasure2 = [50,500,50000]
        Nmeasure = [10,100,1000]

        #length 100 array with exponentially increasing sweep numbers from 0.01*min(sweeps) to max(sweeps)*0.01
        sweep_steps = np.logspace(np.log10(min(sweeps)*0.1),np.log10(max(sweeps)*0.1),num=ntries)
        nmeasure2_steps = np.logspace(np.log10(min(Nmeasure2)),np.log10(max(Nmeasure2)),num=ntries)
        nmeasure_steps = np.logspace(np.log10(min(Nmeasure)),np.log10(max(Nmeasure)),num=ntries)

        pbar = tqdm(range(ntries), disable=False)
        _try = 0
        while _try < ntries:
            _try += 1

            tune_dir = self.save_dir / "tune"

            shutil.rmtree(self.save_dir / "tune", ignore_errors=True)
            tune_dir.mkdir(parents=True, exist_ok=True)

            tune_parameters = deepcopy(self.input_parameters)

            # initial thermalization and measurement sweeps
            tune_parameters.sweeps = int(sweep_steps[_try])
            tune_parameters.thermalization = int(0.2 * tune_parameters.sweeps)
            tune_parameters.Nmeasure2 = int(nmeasure2_steps[_try])
            tune_parameters.Nmeasure = int(nmeasure_steps[_try])
            tune_parameters.save(save_dir_path=tune_dir)

            self._save_parameters(tune_parameters,tune_dir)

            log.debug("Tuning measurement interval. Running 50000 sweeps.")
            
            # time execution
            start_time = time.time()

            try:
                self._execute_worm(input_file=tune_parameters.ini_path,executable=executable)
            except subprocess.CalledProcessError as e:
                log.error(e.stderr.decode("utf-8"))
                log.error(e.stdout.decode("utf-8"))
                continue


            time_interval = time.time() - start_time

            if time_interval < 10:
                _try += 10
        

            converged, max_rel_error, n_measurements, tau_max = self.check_convergence(
                results=WormOutput(out_file_path=tune_parameters.outputfile)
            )

            pbar.set_description(
                f"Running {tune_parameters.sweeps} sweeps. Nmeasure2: {tune_parameters.Nmeasure2}. Nmeasure: {tune_parameters.Nmeasure}. Time: {time_interval:.2f}. Max rel error: {max_rel_error:.2e}. Measurements: {n_measurements}. Tau_max: {tau_max:.2e}. Converged: {converged}"
            )

            if np.isnan(tau_max) or tau_max<1e-3:
                _try += 5

            if not np.isnan(tau_max) and tau_max>1 and tau_max<5:
                break                
                



        if np.isnan(tau_max):
            log.debug("Tau_max is nan. Setting new Nmeasure2 to 10.")
            tau_max = 0
        
        new_measure2 = max(int(tune_parameters.Nmeasure2 * (tau_max / 2)), 10)
        log.debug(f"New Nmeasure2: {new_measure2}")

        self.record["tune"] = {
            "status": "finished",
            "max_rel_error": float(max_rel_error),
            "n_measurements": float(n_measurements),
            "tau_max": float(tau_max),
            "new_Nmeasure2": float(new_measure2),
            "thermalization": float(tune_parameters.thermalization),
            "sweeps": float(tune_parameters.sweeps),
            "tries": _try + 1,
        }

        return new_measure2, tune_parameters.thermalization, tune_parameters.sweeps


    
