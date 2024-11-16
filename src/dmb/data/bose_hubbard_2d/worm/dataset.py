"""Dataset for the Bose-Hubbard model."""

import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from attrs import define, field, frozen

from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2dTransforms
from dmb.data.dataset import DMBData, DMBDataset, DMBSample, IdDataset, \
    SampleFilterStrategy
from dmb.data.split import IdDatasetSplitStrategy
from dmb.logging import create_logger

log = create_logger(__name__)


class WormSimulationsSplitStrategy(IdDatasetSplitStrategy):
    """A strategy for splitting a Bose-Hubbard 2D worm dataset into multiple subsets."""

    def split(
        self,
        dataset: IdDataset,
        split_fractions: dict[str, float],
        seed: int = 42,
    ) -> dict[str, list[str]]:
        """Split a dataset into multiple subsets."""
        # make sure that samples, where ids only differ by _tune, are in the same split

        dataset_ids = dataset.ids
        simulation_ids = defaultdict(list)
        for sample_id in dataset_ids:
            simulation_id = re.sub(r"_tune", "", sample_id)
            simulation_ids[simulation_id].append(sample_id)

        unique_simulation_ids, weights = map(
            np.array, zip(*[(k, len(v)) for k, v in simulation_ids.items()]))

        order = np.arange(len(unique_simulation_ids))
        random.seed(seed)
        random.shuffle(order)  # type: ignore

        agnostic_split_lengths = [
            int(split_fraction * len(dataset))  # type: ignore
            for split_fraction in split_fractions.values()
        ]
        split_indices = [0]
        for split_idx, split_length in enumerate(np.cumsum(agnostic_split_lengths)):
            next_splits = np.argwhere(
                np.cumsum(np.array(weights)[order]) > split_length)
            if len(next_splits) == 0 or split_idx == len(
                    agnostic_split_lengths) - 1:  # enforce last split to reach the end
                split_indices.append(len(weights))
            else:
                split_indices.append(int(np.min(next_splits)))

        split_ids = {}
        for split_name, start_index, end_index in zip(split_fractions,
                                                      split_indices[:-1],
                                                      split_indices[1:]):
            split_ids[split_name] = [
                sample_id
                for simulation_id in unique_simulation_ids[order[start_index:end_index]]
                for sample_id in simulation_ids[simulation_id]
            ]

        return split_ids


@frozen
class BoseHubbard2dSampleFilterStrategy(SampleFilterStrategy):
    """A strategy for filtering samples in a Bose-Hubbard 2D dataset."""

    ztU_range: tuple[float, float]
    muU_range: tuple[float, float]
    zVU_range: tuple[float, float]
    L_range: tuple[int, int]
    max_density_error: float
    allow_negative_mu_null_error: bool = False

    def _filter_error(self, metadata: dict[str, float]) -> bool:
        """Filter samples based on the maximum density error."""
        if metadata["max_density_error"] and metadata[
                "max_density_error"] < self.max_density_error:
            return True

        if self.allow_negative_mu_null_error and metadata["mu"] < 0:
            return True

        return False

    def filter(self, sample: DMBSample) -> bool:
        """Return whether a sample should be included in the dataset."""

        metadata = sample.metadata
        zVU = 4 * metadata["V_nn"] / metadata["U_on"]
        muU = metadata["mu"] / metadata["U_on"]
        ztU = 4 * metadata["J"] / metadata["U_on"]
        L = metadata["L"]

        return bool(self.ztU_range[0] <= ztU <= self.ztU_range[1]
                    and self.muU_range[0] <= muU <= self.muU_range[1]
                    and self.zVU_range[0] <= zVU <= self.zVU_range[1]
                    and self.L_range[0] <= L <= self.L_range[1]
                    and self._filter_error(metadata))


@define
class BoseHubbard2dDataset(DMBDataset):
    """Dataset for the Bose-Hubbard model."""

    dataset_dir_path: Path | str
    transforms: BoseHubbard2dTransforms = field(factory=BoseHubbard2dTransforms)

    sample_filter_strategy: SampleFilterStrategy = field(
        factory=lambda: BoseHubbard2dSampleFilterStrategy(
            ztU_range=(-0.05, 1.0),
            muU_range=(0.05, 3.0),
            zVU_range=(0.75, 1.75),
            L_range=(2, 20),
            max_density_error=0.015,
        ))

    def get_phase_diagram_position(self, idx: int) -> tuple[float, float, float]:
        """Get the phase diagram position for a sample."""

        metadata = self.samples[idx].metadata

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
    ) -> bool:
        """Check if a phase diagram sample exists in the dataset."""

        for idx in range(len(self)):
            zVU_i, muU_i, ztU_i = self.get_phase_diagram_position(idx)

            L_i = self.samples[idx].metadata["L"]

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
    ) -> DMBData | None:
        """Get a phase diagram sample from the dataset."""

        for idx, _ in enumerate(iter(self)):
            zVU_i, muU_i, ztU_i = self.get_phase_diagram_position(idx)
            L_i = self.samples[idx].metadata["L"]

            if (abs(ztU_i - ztU) <= ztU_tol and abs(muU_i - muU) <= muU_tol
                    and abs(zVU_i - zVU) <= zVU_tol and L_i == L):
                return self[idx]

        return None
