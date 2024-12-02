from typing import Iterator, Sized, cast

import numpy as np
from torch.utils.data import Dataset, Sampler


class MDuplicatesPerBatchSampler(Sampler):

    def __init__(self,
                 dataset: Dataset,
                 n_duplicates: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False):

        super().__init__()

        self.dataset = cast(Sized, dataset)
        self.n_duplicates = n_duplicates
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:

        dataset_indices = list(range(len(self.dataset)))

        if self.shuffle:
            np.random.shuffle(dataset_indices)

        batch_indices = []
        for dataset_idx in dataset_indices:
            for _ in range(self.n_duplicates):
                batch_indices.append(dataset_idx)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    batch_indices = []

        if len(batch_indices) > 0 and not self.drop_last:
            yield batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return int(
                np.ceil((len(self.dataset) * self.n_duplicates) / self.batch_size))
