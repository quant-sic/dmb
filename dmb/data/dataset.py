from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch
from attrs import define
from torch.utils.data import Dataset


class IdDataset(Dataset, ABC):

    @abstractmethod
    def get_ids_from_indices(self, indices: tuple[int, ...]) -> tuple[str, ...]: ...

    @abstractmethod
    def get_indices_from_ids(self, ids: tuple[str, ...]) -> tuple[int, ...]: ...


@define
class DMBDataset(IdDataset):
    """Dataset for DMB data.

    Expected directory structure:

        - 'samples': Directory containing the samples.
            - Each sample is stored in a separate directory.
            - The directory name is the sample ID.
            - The directory contains the following files:
                - 'inputs.pt': The input tensor.
                - 'outputs.pt': The output tensor.
                - 'metadata.json': The metadata dictionary.

        - 'metadata.json': The metadata dictionary for the dataset.
    """

    dataset_dir_path: Path
    transforms: Callable[
        [tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
    ]

    def __attrs_post_init__(self):
        self.sample_ids = [
            path.name
            for path in (self.dataset_dir_path / "samples").iterdir()
            if path.is_dir()
        ]
        self.sample_id_paths = [
            self.dataset_dir_path / "samples" / sample_id
            for sample_id in self.sample_ids
        ]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        inputs = torch.load(self.sample_id_paths[idx] / "inputs.pt")
        outputs = torch.load(self.sample_id_paths[idx] / "outputs.pt")

        inputs_transformed, outputs_transformed = self.transforms((inputs, outputs))

        return inputs_transformed, outputs_transformed

    def get_ids_from_indices(self, indices: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(self.sample_ids[idx] for idx in indices)

    def get_indices_from_ids(self, ids: tuple[str, ...]) -> tuple[int, ...]:
        contained_ids = set(self.sample_ids).intersection(ids)
        return tuple(self.sample_ids.index(id) for id in contained_ids)
