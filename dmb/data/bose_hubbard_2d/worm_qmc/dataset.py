import itertools
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import Dataset

from dmb.data.bose_hubbard_2d.worm_qmc.worm.sim import WormSimulation
from dmb.data.bose_hubbard_2d.network_input import net_input
from dmb.utils import create_logger
from dmb.utils import ProgressParallel
from joblib import delayed
from functools import partial

from tqdm import tqdm


log = create_logger(__name__)


class BoseHubbardDataset(Dataset):
    """Dataset for the Bose-Hubbard model."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        observables: List[str] = [
            "density",
            "density_variance",
            "density_density_corr_0",
            "density_density_corr_1",
            "density_density_corr_2",
            "density_density_corr_3",
            "density_squared",
        ],
        base_transforms=None,
        train_transforms=None,
        clean=True,
        reload=False,
        verbose=False,
        max_density_error: float = 0.015,
        include_tune_dirs: bool = False,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.observables = observables

        log.info(f"Loading {self.__class__.__name__} dataset from {self.data_dir}")

        self.base_transforms = base_transforms
        self.train_transforms = train_transforms
        self.max_density_error = max_density_error

        self.clean = clean
        self.reload = reload
        self.verbose = verbose
        self.include_tune_dirs = include_tune_dirs

    @cached_property
    def sim_dirs(self):
        sim_dirs = sorted(self.data_dir.glob("*"))
        if self.include_tune_dirs:
            sim_dirs += [sim_dir / "tune" for sim_dir in sim_dirs]

        log.info(f"Found {len(sim_dirs)} simulation directories.")

        if self.clean:
            sim_dirs = self._clean_sim_dirs(
                self.observables,
                sim_dirs,
                redo=self.reload,
                verbose=self.verbose,
                max_density_error=self.max_density_error,
            )

        return sim_dirs

    @staticmethod
    def _clean_sim_dirs(
        observables,
        sim_dirs,
        redo=False,
        verbose=False,
        max_density_error: Optional[float] = None,
    ):
        def filter_fn(sim_dir):
            try:
                sim = WormSimulation.from_dir(sim_dir)
            except (OSError, KeyError):
                log.error("Could not load simulation from %s.", sim_dir)
                return False

            if "clean" in sim.record and not sim.record["clean"] and not redo:
                return False
            elif "clean" in sim.record and sim.record["clean"] and not redo:
                valid = True
            else:
                try:
                    sim.output.accumulator_observables["Density_Distribution"]["mean"][
                        "value"
                    ]
                    if sim.output.reshape_densities is None:
                        raise KeyError

                except (OSError, KeyError, TypeError, AttributeError):
                    log.error("Could not load density distribution from %s.", sim_dir)
                    valid = False
                else:
                    valid = True
                finally:
                    sim.record["clean"] = valid

                    # after saving the record, return if not valid
                    if not valid:
                        return False

            # general purpose validation
            sim.record["clean"] = valid

            return valid

        def filter_by_error(sim_dir):
            try:
                sim = WormSimulation.from_dir(sim_dir)
            except (OSError, KeyError):
                log.error("Could not load simulation from %s.", sim_dir)
                return False

            try:
                # if not sim.record["steps"][-1]["error"] == sim.max_density_error:
                #     log.warning(
                #         "max_density_error in record does not match sim.max_density_error. "
                #     )
                #     # sim.record["steps"][-1]["error"] = sim.max_density_error
                # return (
                #     sim.max_density_error <= max_density_error and sim.max_tau_int > 0
                # )
                return (sim.record["steps"][-1]["error"] <= max_density_error) and (
                    sim.record["steps"][-1]["tau_max"] > 0
                )
            except (IndexError, TypeError, KeyError):
                return False

        sim_dirs = list(
            itertools.compress(
                sim_dirs,
                # [filter_fn(sim_dir) for sim_dir in sim_dirs]
                ProgressParallel(
                    n_jobs=10,
                    total=len(sim_dirs),
                    desc="Filtering Dataset",
                    use_tqdm=verbose,
                )(delayed(filter_fn)(sim_dir) for sim_dir in sim_dirs),
            )
        )

        if max_density_error is not None:
            sim_dirs = list(
                itertools.compress(
                    sim_dirs,
                    # [filter_by_error(sim_dir) for sim_dir in sim_dirs]
                    ProgressParallel(
                        n_jobs=10,
                        total=len(sim_dirs),
                        desc="Filtering Dataset by Error",
                        use_tqdm=verbose,
                    )(delayed(filter_by_error)(sim_dir) for sim_dir in sim_dirs),
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
            except Exception:
                reload = True

            if not inputs_path.exists() or not outputs_path.exists() or reload:
                sim = WormSimulation.from_dir(sim_dir)

                saved_observables = sim.observables.observables_names

                # stack observables
                outputs = torch.stack(
                    [
                        torch.from_numpy(sim.observables[obs]).float()
                        for obs in saved_observables
                    ],
                    dim=0,
                )

                inputs = net_input(
                    sim.input_parameters.mu,
                    sim.input_parameters.U_on,
                    sim.input_parameters.V_nn,
                    cb_projection=True,
                    target_density=sim.observables["density"],
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

        inputs, outputs = self.loaded_samples[idx]

        # apply transforms
        if self.base_transforms is not None:
            inputs, outputs = self.base_transforms((inputs, outputs))

        if self.train_transforms is not None and self.apply_train_transforms:
            inputs, outputs = self.train_transforms((inputs, outputs))

        return inputs, outputs

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

    def reload_samples(self):
        with ProgressParallel(n_jobs=10, total=len(self)) as parallel:
            parallel(
                delayed(partial(self.load_sample, reload=True))(idx)
                for idx in tqdm(range(len(self)))
            )

    def get_sim(self, idx):
        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        return sim

    def get_parameters(self, idx):
        sim_dir = self.sim_dirs[idx]
        try:
            sim = WormSimulation.from_dir(sim_dir)
        except (OSError, KeyError):
            log.error("Could not load simulation from %s.", sim_dir)
            return None

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
            L_i = self.get_parameters(idx).Lx

            if (
                abs(ztU_i - ztU) <= ztU_tol
                and abs(muU_i - muU) <= muU_tol
                and abs(zVU_i - zVU) <= zVU_tol
                and L_i == L
            ):
                return True

        return False
