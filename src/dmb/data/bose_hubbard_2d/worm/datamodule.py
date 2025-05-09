"""LightningDataModule for BoseHubbard2dDataset."""

from functools import partial
from typing import Any, Callable, Literal

import lightning.pytorch as pl
from lightning.fabric.utilities.seed import pl_worker_init_function
from torch.utils.data import DataLoader, Subset

from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbard2dDataset
from dmb.data.collate import collate_sizes
from dmb.data.sampler import MDuplicatesPerBatchSampler
from dmb.data.split import Split
from dmb.data.utils import chain_fns
from dmb.logging import create_logger

log = create_logger(__name__)


class BoseHubbard2dDataModule(pl.LightningDataModule):
    """LightningDataModule for BoseHubbard2dDataset."""

    def __init__(
        self,
        dataset: BoseHubbard2dDataset,
        split: Split,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_sampler: dict[
            Literal["train", "val", "test"], partial[MDuplicatesPerBatchSampler]
        ]
        | None = None,
    ) -> None:
        """Initialize the data module.

        Args:
            dataset: The dataset to use.
            split: The split to use.
            batch_size: The batch size to use.
            num_workers: The number of workers to use.
            pin_memory: Whether to pin memory.
        """
        super().__init__()

        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.batch_sampler = batch_sampler or {
            "train": partial(MDuplicatesPerBatchSampler, n_duplicates=1),
            "val": partial(MDuplicatesPerBatchSampler, n_duplicates=1),
            "test": partial(MDuplicatesPerBatchSampler, n_duplicates=1),
        }
        self.stage_subsets: dict[str, Subset[BoseHubbard2dDataset]] = {}
        self.initialized_batch_samplers: dict[str, MDuplicatesPerBatchSampler] = {}

    def get_collate_fn(self) -> Callable:
        """Get the collate function for the dataset."""

        collate_fns: list[Callable] = [collate_sizes]

        return chain_fns(collate_fns)

    def setup(self, stage: str = "fit") -> None:
        """Setup the dataset for the given stage."""

        log.info("Splitting dataset for stage: %s", stage)
        self.stage_subsets = self.split.apply(self.dataset)

        if stage == "fit":
            self.dataset.transforms.mode = "train"
        else:
            self.dataset.transforms.mode = "base"

        log.info("Setup for stage: %s", stage)
        log.info(
            "Dataset sizes: %s", {k: len(v) for k, v in self.stage_subsets.items()}
        )
        log.info("Dataset transforms: %s", self.dataset.transforms)

        self.initialized_batch_samplers = {
            sampler_stage: self.batch_sampler[sampler_stage](  # type: ignore
                dataset=self.stage_subsets[sampler_stage],
                batch_size=self.batch_size,
                shuffle=sampler_stage == "train",
            )
            for sampler_stage in ("train", "val", "test")
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["train"],
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
            batch_sampler=self.initialized_batch_samplers["train"],
            worker_init_fn=pl_worker_init_function,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["val"],
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
            batch_sampler=self.initialized_batch_samplers["val"],
            worker_init_fn=pl_worker_init_function,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
            batch_sampler=self.initialized_batch_samplers["test"],
            worker_init_fn=pl_worker_init_function,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the state dictionary."""
        state = {}
        for stage, sampler in self.initialized_batch_samplers.items():
            state[f"{stage}_sampler"] = sampler.state_dict()

        state["dataset"] = self.dataset.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load the state dictionary."""
        for stage, sampler in self.initialized_batch_samplers.items():
            sampler.load_state_dict(state[f"{stage}_sampler"])
            log.info("Loaded sampler state for stage: %s", stage)

        if "dataset" in state:
            self.dataset.load_state_dict(state["dataset"])
            log.info("Loaded dataset state")
