from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from attrs import define
from pathlib import Path
import json
import torch
from typing import Callable


class IdDataset(Dataset, ABC):

    @abstractmethod
    def get_ids_from_indices(self, indices: tuple[int, ...]) -> tuple[str, ...]: ...

    @abstractmethod
    def get_indices_from_ids(self, ids: tuple[str, ...]) -> tuple[int, ...]: ...


@define
class DMBDataset(IdDataset):
    """Dataset for DMB data.

    Expected directory structure:
        - Each sample is stored in a separate directory.
        - The directory name is the sample ID.
        - The directory contains the following files:
            - 'inputs.pt': The input tensor.
            - 'outputs.pt': The output tensor.
            - 'metadata.json': The metadata dictionary.
    """

    dataset_dir_path: Path
    transforms: Callable[
        [tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
    ]

    def __attrs_post_init__(self):
        self.sample_ids = [
            path.name for path in self.dataset_dir_path.iterdir() if path.is_dir()
        ]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        sample_dir_path = self.dataset_dir_path / sample_id
        inputs = torch.load(sample_dir_path / "inputs.pt")
        outputs = torch.load(sample_dir_path / "outputs.pt")
        return {
            "inputs": inputs,
            "outputs": outputs,
        }

    def get_ids_from_indices(self, indices: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(self.sample_ids[idx] for idx in indices)

    def get_indices_from_ids(self, ids: tuple[str, ...]) -> tuple[int, ...]:
        contained_ids = set([d.name for d in self.sim_dirs]).intersection(ids)
        return tuple(self.sample_ids.index(id) for id in contained_ids)
