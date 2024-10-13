"""Dataset splitting functionality."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from attrs import define
from torch.utils.data import Subset

from dmb.data.dataset import IdDataset


@define
class Split:
    """A Split into multiple subsets based on a dictionary of split IDs.

    Attributes:
        split_ids: A dictionary of split names and corresponding IDs.
    """

    split_ids: dict[str, tuple[str, ...]]

    def apply(self, dataset: IdDataset) -> dict[str, Subset]:
        """Apply the split to a dataset."""
        return {
            split_name:
            Subset(
                dataset=dataset,
                indices=dataset.get_indices_from_ids(
                    self.split_ids[split_name]),
            )
            for split_name in self.split_ids
        }

    @classmethod
    def from_dataset(cls,
                     dataset: IdDataset,
                     split_fractions: dict[str, float],
                     seed: int = 42):
        """Generate a split from a dataset and split fractions."""
        split_ids = cls.generate(split_fractions, dataset, seed)
        return cls(split_ids=split_ids)

    @staticmethod
    def generate(split_fractions: dict[str, float],
                 dataset: IdDataset,
                 seed: int = 42) -> dict[str, tuple[str, ...]]:
        """Generate a split based on split fractions."""
        if sum(split_fractions.values()) > 1.0 + 1e-8 or any(
                fraction < 0.0 for fraction in split_fractions.values()):
            raise ValueError(
                "Split fractions be positive and sum to at most 1.")

        dataset_indices = np.arange(len(dataset))

        random.seed(seed)
        random.shuffle(dataset_indices)

        split_lengths = [
            int(split_fraction * len(dataset))
            for split_fraction in split_fractions.values()
        ]

        split_indices = [0] + list(np.cumsum(split_lengths))

        split_ids = {}
        for (split_name,
             split_fraction), start_shuffled_idx, end_shuffled_idx in zip(
                 split_fractions.items(), split_indices[:-1],
                 split_indices[1:]):
            split_ids[split_name] = dataset.get_ids_from_indices(
                dataset_indices[start_shuffled_idx:end_shuffled_idx])

        return split_ids

    @classmethod
    def from_file(cls, file_path: Path) -> Split:
        """Load a split from a file."""
        with open(file_path, "r") as f:
            split_ids = json.load(f)

        return cls(split_ids=split_ids)

    def to_file(self, file_path: Path) -> None:
        """Save a split to a file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.split_ids, f)
