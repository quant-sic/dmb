import itertools
import shutil
from functools import cached_property, partial
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import torch
from attrs import define
from joblib import delayed
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.network_input import net_input
from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2DTransforms
from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulation
from dmb.data.dataset import IdDataset
from dmb.io import ProgressParallel
from dmb.logging import create_logger

log = create_logger(__name__)


class _PhaseDiagramSamplesMixin:

    def phase_diagram_position(self, idx):
        pars = WormSimulation.from_dir(self.sim_dirs[idx]).input_parameters
        U_on = pars.U_on
        mu = float(pars.mu_offset)
        J = pars.t_hop
        V_nn = pars.V_nn

        return (
            4 * V_nn[0, 0, 0] / U_on[0, 0],
            mu / U_on[0, 0],
            4 * J[0, 0, 0] / U_on[0, 0],
        )

    def has_phase_diagram_sample(
        self,
        ztU: float,
        muU: float,
        zVU: float,
        L: int,
        ztU_tol: float = 0.01,
        muU_tol: float = 0.01,
        zVU_tol: float = 0.01,
    ):
        for idx, _ in enumerate(self):
            zVU_i, muU_i, ztU_i = self.phase_diagram_position(idx)
            L_i = WormSimulation.from_dir(
                self.sim_dirs[idx]).input_parameters.Lx

            if (abs(ztU_i - ztU) <= ztU_tol and abs(muU_i - muU) <= muU_tol
                    and abs(zVU_i - zVU) <= zVU_tol and L_i == L):
                return True

        return False

    def get_phase_diagram_sample(
        self,
        ztU: float,
        muU: float,
        zVU: float,
        L: int,
        ztU_tol: float = 0.01,
        muU_tol: float = 0.01,
        zVU_tol: float = 0.01,
    ):
        for idx, _ in enumerate(self):
            zVU_i, muU_i, ztU_i = self.phase_diagram_position(idx)
            L_i = WormSimulation.from_dir(
                self.sim_dirs[idx]).input_parameters.Lx

            if (abs(ztU_i - ztU) <= ztU_tol and abs(muU_i - muU) <= muU_tol
                    and abs(zVU_i - zVU) <= zVU_tol and L_i == L):
                return self[idx]

        return None


@define
class BoseHubbardDataset(IdDataset, _PhaseDiagramSamplesMixin):
    """Dataset for the Bose-Hubbard model."""

    data_dir: Path | str
    transforms: BoseHubbard2DTransforms

    observables: list[str] = [
        "density",
        "density_density_corr_0",
        "density_density_corr_1",
        "density_density_corr_2",
        "density_density_corr_3",
        "density_squared",
        "density_max",
        "density_min",
        "density_variance",
    ]
    clean: bool = True
    reload: bool = False
    verbose: bool = False
    max_density_error: float = 0.015
    include_tune_dirs: bool = False
    recalculate_errors: bool = False
    delete_unreadable: bool = False

    def __attrs_post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        log.info(
            f"Loading {self.__class__.__name__} dataset from {self.data_dir}")

    @cached_property
    def sim_dirs(self):
        sim_dirs = sorted(self.data_dir.glob("*"))
        if self.include_tune_dirs:
            sim_dirs += [sim_dir / "tune" for sim_dir in sim_dirs]

        log.info(f"Found {len(sim_dirs)} simulation directories.")

        if self.clean:
            sim_dirs = self._clean_sim_dirs(
                sim_dirs,
                redo=self.reload,
                verbose=self.verbose,
                max_density_error=self.max_density_error,
                recalculate_errors=self.recalculate_errors,
                delete_unreadable=self.delete_unreadable,
            )

        return sim_dirs

    def log(self, msg, level="info"):
        if self.verbose:
            getattr(log, level)(msg)

    def get_simulation_valid(self,
                             simulation_dir: Path,
                             redo: bool = False) -> bool:
        try:
            sim = WormSimulation.from_dir(simulation_dir)
        except (OSError, KeyError):
            self.log(f"Could not load simulation from {simulation_dir}",
                     "error")
            return False

        if "clean" in sim.record and not sim.record["clean"] and not redo:
            return False
        elif "clean" in sim.record and sim.record["clean"] and not redo:
            return True
        else:
            valid = sim.valid

        # general purpose validation
        sim.record["clean"] = valid

        return valid

    def filter_by_error(
        self,
        simulation: WormSimulation,
        max_density_error: float,
        recalculate_errors: bool = False,
    ) -> bool:
        print(type(simulation))
        try:
            if recalculate_errors:
                self.log(f"Recalculating errors for {simulation.save_dir}",
                         "info")
                if len(simulation.record["steps"]) == 0:
                    simulation.record["steps"] = [{
                        "error": None,
                        "tau_max": None
                    }]

                simulation.record["steps"][-1][
                    "error"] = simulation.max_density_error
                simulation.record["steps"][-1][
                    "tau_max"] = simulation.max_tau_int

            return (simulation.record["steps"][-1]["error"]
                    <= max_density_error) and (
                        simulation.record["steps"][-1]["tau_max"] > 0)
        except (IndexError, TypeError, KeyError) as e:
            self.log(f"Error {e} During error filtering for {simulation}",
                     "error")
            return False

    def _clean_sim_dirs(
        self,
        sim_dirs,
        redo=False,
        verbose=False,
        max_density_error: float | None = None,
        recalculate_errors: bool = False,
        delete_unreadable: bool = False,
    ):

        valid_sim_dirs = [
            self.get_simulation_valid(sim_dir, redo=redo) for sim_dir in tqdm(
                sim_dirs,
                desc="Filtering valid simulations",
                total=len(sim_dirs),
            )
        ]

        if delete_unreadable:
            for sim_dir, valid in zip(sim_dirs, valid_sim_dirs):
                if not valid:
                    self.log(f"Deleting {sim_dir}")
                    shutil.rmtree(sim_dir)

        sim_dirs = list(itertools.compress(
            sim_dirs,
            valid_sim_dirs,
        ))

        if max_density_error is not None:
            sim_dirs = list(
                itertools.compress(
                    sim_dirs,
                    map(
                        partial(
                            self.filter_by_error,
                            max_density_error=max_density_error,
                            recalculate_errors=recalculate_errors,
                        ),
                        tqdm(
                            map(WormSimulation.from_dir, sim_dirs),
                            desc="Filtering valid simulations by error",
                            total=len(sim_dirs),
                        ),
                    )))

        return sim_dirs

    def __len__(self):
        return len(self.sim_dirs)

    @property
    def loaded_samples(self):
        """Dict that stores loaded samples."""
        if not hasattr(self, "_loaded_samples"):
            self._loaded_samples = {}
        return self._loaded_samples

    def load_sample(self, idx, reload=False):
        if idx not in self.loaded_samples or reload:
            sim_dir = self.sim_dirs[idx]

            inputs_path = sim_dir / "inputs.pt"
            outputs_path = sim_dir / "outputs.pt"

            try:
                sim = WormSimulation.from_dir(sim_dir)
                saved_observables = sim.record["saved_observables"]
            except Exception:
                reload = True

            if not inputs_path.exists() or not outputs_path.exists() or reload:
                sim = WormSimulation.from_dir(sim_dir)

                saved_observables = (
                    sim.observables.observable_names["primary"] +
                    sim.observables.observable_names["derived"])

                expectation_values = [
                    sim.observables.get_expectation_value(obs_type, obs_name)
                    for obs_type in ["primary", "derived"]
                    for obs_name in sim.observables.observable_names[obs_type]
                ]
                expanded_expectation_values = [(np.full(
                    shape=(sim.input_parameters.Lx, sim.input_parameters.Ly),
                    fill_value=obs,
                ) if obs.ndim == 0 else obs) for obs in expectation_values]
                # stack observables
                outputs = torch.stack(
                    [
                        torch.from_numpy(obs)
                        for obs in expanded_expectation_values
                    ],
                    dim=0,
                )

                inputs = net_input(
                    sim.input_parameters.mu,
                    sim.input_parameters.U_on,
                    sim.input_parameters.V_nn,
                    cb_projection=True,
                    target_density=sim.observables.get_expectation_value(
                        "primary", "density"),
                )

                # save to .npy files
                torch.save(inputs, inputs_path)
                torch.save(outputs, outputs_path)

                # save saved_observables
                sim.record["saved_observables"] = saved_observables

            else:
                inputs = torch.load(inputs_path)
                outputs = torch.load(outputs_path)

                # load saved_observables
                sim = WormSimulation.from_dir(sim_dir)
                saved_observables = sim.record["saved_observables"]

            # filter observables
            outputs = outputs[[
                saved_observables.index(obs) for obs in self.observables
            ]]

            self.loaded_samples[idx] = (inputs, outputs)

        return self.loaded_samples[idx]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample by index."""
        inputs, outputs = self.load_sample(idx)
        inputs_transformed, outputs_transformed = self.transforms(
            (inputs, outputs))

        return inputs_transformed, outputs_transformed

    def get_ids_from_indices(self, indices: tuple[int, ...]):
        return tuple(self.sim_dirs[idx].name for idx in indices)

    def get_indices_from_ids(self, ids):
        return tuple(self.sim_dirs.index(self.data_dir / id) for id in ids)
