from torch.utils.data import Dataset
from dmb.data.bose_hubbard_2d.worm.sim import WormSimulation
from pathlib import Path
from functools import cached_property
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset
from dmb.utils import create_logger
from tqdm.auto import tqdm
import os
import torch
from joblib import delayed
from dmb.utils.io import ProgressParallel
import itertools
import math
import numpy as np
from dmb.data.bose_hubbard_2d.network_input import net_input

log = create_logger(__name__)


class BoseHubbardDataset(Dataset):

    """Dataset for the Bose-Hubbard model."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        observables: List[str] = [
            "Density_Distribution",
            "Density_Matrix",
            "DensDens_CorrFun",
            "DensDens_CorrFun_local_0",
            "DensDens_CorrFun_local_1",
            "DensDens_CorrFun_local_2",
            "DensDens_CorrFun_local_3",
            "DensDens_Diff_0",
            "DensDens_Diff_1",
            "DensDens_Diff_2",
            "DensDens_Diff_3",
            "DensDens_Diff_Diag_0",
            "DensDens_Diff_Diag_1",
            "DensDens_Diff_Diag_2",
            "DensDens_Diff_Diag_3",
            "DensDens_CorrFun_local_2_step_0",
            "DensDens_CorrFun_local_2_step_1",
            "DensDens_CorrFun_local_2_step_2",
            "DensDens_CorrFun_local_2_step_3",
            "DensDens_CorrFun_local_diag_0",
            "DensDens_CorrFun_local_diag_1",
            "DensDens_CorrFun_local_diag_2",
            "DensDens_CorrFun_local_diag_3",
            "DensDens_CorrFun_sq_0",
            "DensDens_CorrFun_sq_1",
            "DensDens_CorrFun_sq_2",
            "DensDens_CorrFun_sq_3",
            "DensDens_CorrFun_sq_0_",
            "DensDens_CorrFun_sq_1_",
            "DensDens_CorrFun_sq_2_",
            "DensDens_CorrFun_sq_3_",
            "DensDens_CorrFun_sq_diag_0",
            "DensDens_CorrFun_sq_diag_1",
            "DensDens_CorrFun_sq_diag_2",
            "DensDens_CorrFun_sq_diag_3",
            "DensDens_CorrFun_sq_diag_0_",
            "DensDens_CorrFun_sq_diag_1_",
            "DensDens_CorrFun_sq_diag_2_",
            "DensDens_CorrFun_sq_diag_3_",
            "DensDens_CorrFun_sq_0",
            "Density_Distribution_squared",
        ],
        base_transforms=None,
        train_transforms=None,
        clean=True,
        reload=False,
        verbose=False,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.observables = observables

        log.info(f"Loading {self.__class__.__name__} dataset from {self.data_dir}")

        self.base_transforms = base_transforms
        self.train_transforms = train_transforms

        self.clean = clean
        self.reload = reload
        self.verbose = verbose

    @cached_property
    def sim_dirs(self):
        sim_dirs = sorted(self.data_dir.glob("*"))

        if self.clean:
            sim_dirs = self._clean_sim_dirs(
                self.observables, sim_dirs, redo=self.reload, verbose=self.verbose
            )

        return sim_dirs

    @staticmethod
    def _clean_sim_dirs(observables, sim_dirs, redo=False, verbose=False):
        def filter_fn(sim_dir):
            try:
                sim = WormSimulation.from_dir(sim_dir)
            except:
                return False

            if "clean" in sim.record and sim.record["clean"] == False and not redo:
                return False
            elif "clean" in sim.record and sim.record["clean"] == True and not redo:
                valid = True
            else:
                try:
                    sim.results.observables["Density_Distribution"]["mean"]["value"]
                    valid = True
                except:
                    valid = False

                finally:
                    sim.record["clean"] = valid

                    # after saving the record, return if not valid
                    if not valid:
                        return False

            # general purpose validation
            sim.record["clean"] = valid

            # check if all observables are present
            if valid:
                if "observables" not in sim.record:
                    sim.record["observables"] = list(sim.results.observables.keys())

                obs_present = set(observables).issubset(set(sim.record["observables"]))
                valid = valid and obs_present

            return valid

        sim_dirs = list(
            itertools.compress(
                sim_dirs,
                ProgressParallel(
                    n_jobs=10,
                    total=len(sim_dirs),
                    desc="Filtering Dataset",
                    use_tqdm=verbose,
                )(delayed(filter_fn)(sim_dir) for sim_dir in sim_dirs),
            )
        )

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
            except:
                reload = True

            if not inputs_path.exists() or not outputs_path.exists() or reload:
                sim = WormSimulation.from_dir(sim_dir)

                inputs = net_input(
                    sim.input_parameters.mu,
                    sim.input_parameters.U_on,
                    sim.input_parameters.V_nn,
                    cb_projection=True,
                )

                # filter function for observables. Only keep the ones that have shape (L**2). also check if isinstance(...,np.ndarray)
                obs_filter_fn = lambda obs: isinstance(
                    sim.results.observables[obs]["mean"]["value"], np.ndarray
                ) and len(sim.results.observables[obs]["mean"]["value"]) == int(
                    sim.input_parameters.Lx
                ) * int(
                    sim.input_parameters.Ly
                )

                saved_observables = list(
                    filter(obs_filter_fn, sorted(sim.results.observables.keys()))
                )
                # stack observables
                outputs = torch.stack(
                    [
                        torch.from_numpy(
                            sim.results.observables[obs]["mean"]["value"]
                        ).float()
                        for obs in saved_observables
                    ],
                    dim=0,
                )
                # reshape
                outputs = outputs.view(
                    outputs.shape[0],
                    int(math.sqrt(outputs.shape[1])),
                    int(math.sqrt(outputs.shape[1])),
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
            outputs = outputs[
                [saved_observables.index(obs) for obs in self.observables]
            ]

            self.loaded_samples[idx] = (inputs, outputs)

        # apply transforms
        if self.base_transforms is not None:
            inputs, outputs = self.base_transforms((inputs, outputs))

        if self.train_transforms is not None and self.apply_train_transforms:
            inputs, outputs = self.train_transforms((inputs, outputs))

        return self.loaded_samples[idx]

    @property
    def apply_train_transforms(self) -> bool:
        if hasattr(self, "_apply_train_transforms"):
            return self._apply_train_transforms
        else:
            raise AttributeError("apply_transforms is not set. Please set it first.")

    @apply_train_transforms.setter
    def apply_train_transforms(self, value: bool) -> None:
        self._apply_train_transforms = value

    def __getitem__(self, idx, reload=False):
        inputs, outputs = self.load_sample(idx, reload=reload)

        return inputs, outputs

    def get_sim(self, idx):
        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        return sim

    def get_parameters(self, idx):
        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        return sim.input_parameters

    def phase_diagram_position(self, idx):
        pars = self.get_parameters(idx)
        U_on = pars.U_on
        mu = float(pars.mu_offset)
        J = pars.t_hop
        V_nn = pars.V_nn

        # return mu,U_on,J
        return 4 * V_nn[0] / U_on[0], mu / U_on[0], 4 * J[0] / U_on[0]

    def get_dataset_ids_from_indices(
        self, indices: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        ids = [
            self.sim_dirs[idx].name
            for idx in (indices if isinstance(indices, list) else [indices])
        ]

        if isinstance(indices, int):
            ids = ids[0]

        return ids
