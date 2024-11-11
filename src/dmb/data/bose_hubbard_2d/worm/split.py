import random
import re
from collections import defaultdict

import numpy as np

from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbard2dDataset
from dmb.data.dataset import IdDataset
from dmb.data.split import IdDatasetSplitStrategy


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
