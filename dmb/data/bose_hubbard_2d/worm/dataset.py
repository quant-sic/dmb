import itertools
import shutil
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from attrs import define
from joblib import delayed

from dmb.data.bose_hubbard_2d.nn_input import get_nn_input
from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2DTransforms
from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulation
from dmb.data.dataset import DMBDataset
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
            L_i = WormSimulation.from_dir(self.sim_dirs[idx]).input_parameters.Lx

            if (
                abs(ztU_i - ztU) <= ztU_tol
                and abs(muU_i - muU) <= muU_tol
                and abs(zVU_i - zVU) <= zVU_tol
                and L_i == L
            ):
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
            L_i = WormSimulation.from_dir(self.sim_dirs[idx]).input_parameters.Lx

            if (
                abs(ztU_i - ztU) <= ztU_tol
                and abs(muU_i - muU) <= muU_tol
                and abs(zVU_i - zVU) <= zVU_tol
                and L_i == L
            ):
                return self[idx]

        return None


@define
class BoseHubbardDataset(DMBDataset, _PhaseDiagramSamplesMixin):
    """Dataset for the Bose-Hubbard model."""

    data_dir: Path | str
    transforms: BoseHubbard2DTransforms

    def __attrs_post_init__(self):
        super().__init__(self.data_dir, self.transforms)
