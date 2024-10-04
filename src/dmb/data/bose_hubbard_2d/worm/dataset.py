"""Dataset for the Bose-Hubbard model."""

import json
from pathlib import Path

from attrs import define

from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2dTransforms
from dmb.data.dataset import DMBDataset
from dmb.logging import create_logger

log = create_logger(__name__)


@define
class BoseHubbard2dDataset(DMBDataset):
    """Dataset for the Bose-Hubbard model."""

    dataset_dir_path: Path
    transforms: BoseHubbard2dTransforms

    def get_metadata(self, idx):
        with open(self.sample_id_paths[idx] / "metadata.json", "r") as f:
            return json.load(f)

    def get_phase_diagram_position(self, idx):

        metadata = self.get_metadata(idx)

        return (
            4 * metadata["V_nn"] / metadata["U_on"],
            metadata["mu"] / metadata["U_on"],
            4 * metadata["J"] / metadata["U_on"],
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
            zVU_i, muU_i, ztU_i = self.get_phase_diagram_position(idx)

            metadata = self.get_metadata(idx)
            L_i = metadata["L"]

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
            zVU_i, muU_i, ztU_i = self.get_phase_diagram_position(idx)
            metadata = self.get_metadata(idx)
            L_i = metadata["L"]

            if (abs(ztU_i - ztU) <= ztU_tol and abs(muU_i - muU) <= muU_tol
                    and abs(zVU_i - zVU) <= zVU_tol and L_i == L):
                return self[idx]

        return None
