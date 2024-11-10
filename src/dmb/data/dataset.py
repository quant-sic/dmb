"""Dataset classes for DMB data."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, TypedDict

import torch
from attrs import define
from torch.utils.data import Dataset

from dmb.data.transforms import InputOutputDMBTransform


class DMBData(TypedDict):
    """A DMB data sample."""

    inputs: torch.Tensor
    outputs: torch.Tensor


class IdDataset(Dataset, ABC):
    """Dataset with sample IDs."""

    @abstractmethod
    def get_ids_from_indices(self, indices: Iterable[int]) -> tuple[str, ...]:
        ...

    @abstractmethod
    def get_indices_from_ids(self, ids: Iterable[str]) -> tuple[int, ...]:
        ...


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

    dataset_dir_path: Path | str
    transforms: InputOutputDMBTransform

    def __attrs_post_init__(self) -> None:
        samples_dir_path = Path(self.dataset_dir_path) / "samples"
        samples_dir_path.mkdir(parents=True, exist_ok=True)

        self.sample_ids = [
            path.name for path in samples_dir_path.iterdir() if path.is_dir()
        ]
        self.sample_id_paths = [
            samples_dir_path / sample_id for sample_id in self.sample_ids
        ]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> DMBData:

        inputs = torch.load(
            self.sample_id_paths[idx] / "inputs.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        outputs = torch.load(
            self.sample_id_paths[idx] / "outputs.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )

        inputs_transformed, outputs_transformed = self.transforms(inputs, outputs)

        return DMBData(inputs=inputs_transformed, outputs=outputs_transformed)

    def get_ids_from_indices(self, indices: Iterable[int]) -> tuple[str, ...]:
        return tuple(self.sample_ids[idx] for idx in indices)

    def get_indices_from_ids(self, ids: Iterable[str]) -> tuple[int, ...]:
        contained_ids = set(self.sample_ids).intersection(ids)
        return tuple(self.sample_ids.index(id) for id in contained_ids)
