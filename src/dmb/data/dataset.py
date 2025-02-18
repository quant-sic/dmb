"""Dataset classes for DMB data."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import torch
from attrs import define, field, frozen
from torch.utils.data import Dataset

from dmb.data.transforms import (
    DMBData,
    DMBDatasetTransform,
    IdentityDMBDatasetTransform,
)


class IdDataset(Dataset, ABC):
    """Dataset with sample IDs."""

    @property
    @abstractmethod
    def ids(self) -> tuple[str, ...]:
        """Return the IDs of the samples in the dataset."""

    @abstractmethod
    def get_ids_from_indices(self, indices: Iterable[int]) -> tuple[str, ...]:
        """Return the IDs of the samples at the given indices."""

    @abstractmethod
    def get_indices_from_ids(self, ids: Iterable[str]) -> tuple[int, ...]:
        """Return the indices of the samples with the given IDs."""


@frozen
class DMBSample:
    """A sample in a DMB dataset.

    - The directory name is the sample ID.
    - The directory contains the following files:
        - 'inputs.pt': The input tensor.
        - 'outputs.pt': The output tensor.
        - 'metadata.json': The metadata dictionary.
    """

    id: str
    sample_dir_path: Path

    @property
    def inputs(self) -> torch.Tensor:
        """Return the inputs tensor."""
        inputs: torch.Tensor = torch.load(
            self.sample_dir_path / "inputs.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        return inputs

    @property
    def outputs(self) -> torch.Tensor:
        """Return the outputs tensor."""
        outputs: torch.Tensor = torch.load(
            self.sample_dir_path / "outputs.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata dictionary."""
        with open(self.sample_dir_path / "metadata.json", encoding="utf-8") as file:
            metadata: dict = json.load(file)

        return metadata


class SampleFilterStrategy(ABC):
    """Strategy for filtering samples."""

    @abstractmethod
    def filter(self, sample: DMBSample) -> bool:
        """Return whether the sample should be included."""


class UseAllSamplesFilterStrategy(SampleFilterStrategy):
    """Strategy for using all samples."""

    def filter(self, sample: DMBSample) -> bool:
        """Return True for all samples."""
        return True


@define
class DMBDataset(IdDataset):
    """Dataset for DMB data.

    Expected directory structure:

        - 'samples': Directory containing the samples.
            - Each sample is stored in a separate directory.

        - 'metadata.json': The metadata dictionary for the dataset.
    """

    dataset_dir_path: Path | str

    transforms: DMBDatasetTransform = field(factory=IdentityDMBDatasetTransform)
    sample_filter_strategy: SampleFilterStrategy = field(
        factory=UseAllSamplesFilterStrategy
    )

    sample_ids: list[str] = field(init=False)
    samples: list[DMBSample] = field(init=False)

    def __attrs_post_init__(self) -> None:
        samples_dir_path = Path(self.dataset_dir_path) / "samples"
        samples_dir_path.mkdir(parents=True, exist_ok=True)

        self.samples = list(
            filter(
                self.sample_filter_strategy.filter,
                (
                    DMBSample(
                        id=sample_id, sample_dir_path=samples_dir_path / sample_id
                    )
                    for sample_id in (
                        path.name
                        for path in samples_dir_path.iterdir()
                        if path.is_dir()
                    )
                ),
            )
        )

        self.sample_ids = [sample.id for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DMBData:
        dmb_data = DMBData(
            inputs=self.samples[idx].inputs,
            outputs=self.samples[idx].outputs,
            sample_id=self.samples[idx].id,
        )

        return self.transforms(dmb_data)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.sample_ids)

    def get_ids_from_indices(self, indices: Iterable[int]) -> tuple[str, ...]:
        return tuple(self.sample_ids[idx] for idx in indices)

    def get_indices_from_ids(self, ids: Iterable[str]) -> tuple[int, ...]:
        contained_ids = set(self.sample_ids).intersection(ids)
        return tuple(self.sample_ids.index(id) for id in contained_ids)
