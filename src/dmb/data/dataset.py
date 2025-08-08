"""Dataset classes for DMB data."""

from __future__ import annotations

import itertools
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable

import torch
from attrs import define, field, frozen
from torch.utils.data import Dataset
from tqdm import tqdm

from dmb.data.transforms import (
    DMBData,
    DMBDatasetTransform,
    IdentityDMBDatasetTransform,
)
from dmb.logging import create_logger

log = create_logger(__name__)


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

    @cached_property
    def inputs(self) -> torch.Tensor:
        """Return the inputs tensor."""
        inputs: torch.Tensor = torch.load(
            self.sample_dir_path / "inputs.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        return inputs

    @cached_property
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
        """Initialize the dataset."""

        samples_dir_path = Path(self.dataset_dir_path) / "samples"
        samples_dir_path.mkdir(parents=True, exist_ok=True)

        log.info("Loading samples from %s", samples_dir_path.resolve())
        samples_iterator = list(
            DMBSample(id=sample_id, sample_dir_path=samples_dir_path / sample_id)
            for sample_id in (
                path.name for path in samples_dir_path.iterdir() if path.is_dir()
            )
        )

        log.info("Filtering samples from %s", samples_dir_path.resolve())
        self.samples = [
            sample
            for sample in tqdm(
                samples_iterator,
                desc="Filtering samples",
                total=len(samples_iterator),
                disable=True,
            )
            if self.sample_filter_strategy.filter(sample)
        ]

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
        contained_ids = set(self.sample_ids)
        return tuple(self.sample_ids.index(_id) for _id in ids if _id in contained_ids)

    def state_dict(self) -> dict[str, Any]:
        """Return the state dictionary of the dataset."""
        return {
            "transforms": self.transforms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load the state dictionary of the dataset."""
        if "transforms" in state:
            self.transforms.load_state_dict(state["transforms"])
            log.info("Loaded state dict for transforms: %s", self.transforms.__repr__())


class EvaluationData(Dataset):
    """Dataset for evaluation data.

    This dataset is used to evaluate the model on a specific set of inputs and outputs.
    It is expected to be used with a DataLoader.
    """

    def __init__(self, inputs: list[torch.Tensor], outputs: list[torch.Tensor]) -> None:
        """Initialize the evaluation data."""
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def from_data_generation_functions(
        cls,
        data_generation_functions: list[
            Callable[[], tuple[torch.Tensor, torch.Tensor, ...]]
        ],
    ) -> EvaluationData:
        """Create an EvaluationData instance from data generation functions."""
        inputs, outputs = map(
            list,
            map(
                itertools.chain.from_iterable,
                zip(*[func()[:2] for func in data_generation_functions]),
            ),
        )

        return cls(list(inputs), list(outputs))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> DMBData:
        """Return the inputs and outputs for the given index."""
        return DMBData(
            inputs=self.inputs[idx],
            outputs=self.outputs[idx],
            sample_id=f"eval_{idx}",
        )
