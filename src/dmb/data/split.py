"""Dataset splitting functionality."""

from __future__ import annotations

import json
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
from attrs import define
from torch.utils.data import Subset

from dmb.data.dataset import IdDataset


class IdDatasetSplitStrategy(metaclass=ABCMeta):
    """A strategy for splitting a dataset into multiple subsets."""

    @abstractmethod
    def split(
        self, dataset: IdDataset, split_fractions: dict[str, float], seed: int = 42
    ) -> dict[str, list[str]]:
        """Split a dataset into multiple subsets."""


class AllIdsEqualSplitStrategy(IdDatasetSplitStrategy):
    """A strategy for splitting a dataset into multiple subsets,
    where all IDs are equal."""

    def split(
        self, dataset: IdDataset, split_fractions: dict[str, float], seed: int = 42
    ) -> dict[str, list[str]]:
        """Split a dataset into multiple subsets."""

        generator = np.random.default_rng(seed)
        dataset_indices = generator.permutation(len(dataset))  # type: ignore

        split_lengths = [
            int(split_fraction * len(dataset))  # type: ignore
            for split_fraction in split_fractions.values()
        ]

        split_indices = [0] + list(np.cumsum(split_lengths))

        # enforce last split to reach the end
        split_indices[-1] = len(dataset)  # type: ignore

        split_ids = {}
        for (split_name, split_fraction), start_shuffled_idx, end_shuffled_idx in zip(
            split_fractions.items(), split_indices[:-1], split_indices[1:]
        ):
            split_ids[split_name] = list(
                dataset.get_ids_from_indices(
                    dataset_indices[start_shuffled_idx:end_shuffled_idx]
                )
            )

        return split_ids


@define
class Split:
    """A Split into multiple subsets based on a dictionary of split IDs.

    Attributes:
        split_ids: A dictionary of split names and corresponding IDs.
    """

    split_ids: dict[str, list[str]]

    def apply(self, dataset: IdDataset) -> dict[str, Subset]:
        """Apply the split to a dataset."""
        return {
            split_name: Subset(
                dataset=dataset,
                indices=dataset.get_indices_from_ids(self.split_ids[split_name]),
            )
            for split_name in self.split_ids
        }

    @classmethod
    def from_dataset(
        cls,
        dataset: IdDataset,
        split_fractions: dict[str, float],
        seed: int = 42,
        split_strategy: IdDatasetSplitStrategy | None = None,
    ) -> Split:
        """Generate a split from a dataset and split fractions."""
        if not split_strategy:
            split_strategy = AllIdsEqualSplitStrategy()

        split_ids = cls.generate(
            split_fractions=split_fractions,
            dataset=dataset,
            seed=seed,
            split_strategy=split_strategy,
        )
        return cls(split_ids=split_ids)

    @staticmethod
    def generate(
        split_fractions: dict[str, float],
        dataset: IdDataset,
        split_strategy: IdDatasetSplitStrategy,
        seed: int = 42,
    ) -> dict[str, list[str]]:
        """Generate a split based on split fractions."""
        if sum(split_fractions.values()) > 1.0 + 1e-8 or any(
            fraction < 0.0 for fraction in split_fractions.values()
        ):
            raise ValueError("Split fractions be positive and sum to at most 1.")

        return split_strategy.split(dataset, split_fractions, seed)

    @classmethod
    def from_file(cls, file_path: Path) -> Split:
        """Load a split from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            split_ids = json.load(f)

        return cls(split_ids=split_ids)

    def to_file(self, file_path: Path) -> None:
        """Save a split to a file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.split_ids, f)
