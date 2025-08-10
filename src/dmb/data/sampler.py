"""Sampler for DMB DataLoader."""

from typing import Iterator, Sized, cast

import numpy as np
from torch.utils.data import Dataset, Sampler


class MDuplicatesPerBatchSampler(Sampler):
    """Sampler that samples `n_duplicates` times from the dataset per batch."""

    def __init__(
        self,
        dataset: Dataset,
        n_duplicates: int = 1,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: np.random.Generator | None = None,
    ):
        """Initialize the sampler.

        Args:
            dataset: The dataset to sample from.
            n_duplicates: The number of times to sample each element from the dataset.
            batch_size: The batch size.
            shuffle: Whether to shuffle the dataset indices.
            drop_last: Whether to drop the last batch if it is smaller
                than `batch_size`.
        """

        super().__init__()

        self.dataset = cast(Sized, dataset)
        self.n_duplicates = n_duplicates
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.generator = generator or np.random.default_rng(seed=42)

        self.current_index = 0

    def __iter__(self) -> Iterator[list[int]]:
        dataset_indices = list(range(len(self.dataset)))

        if self.shuffle:
            self.generator.shuffle(dataset_indices)

        batch_indices = []
        while self.current_index < len(dataset_indices):
            dataset_idx = dataset_indices[self.current_index]
            for _ in range(self.n_duplicates):
                batch_indices.append(dataset_idx)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    batch_indices = []
            self.current_index += 1

        if len(batch_indices) > 0 and not self.drop_last:
            yield batch_indices

        self.current_index = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size

        return int(np.ceil((len(self.dataset) * self.n_duplicates) / self.batch_size))

    def state_dict(self) -> dict:
        """Return the state of the sampler."""
        return {
            "generator_state": self.generator.bit_generator.state,
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the sampler."""
        self.generator.bit_generator.state = state_dict["generator_state"]
        self.current_index = state_dict["current_index"]
