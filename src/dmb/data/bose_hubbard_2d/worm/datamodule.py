"""LightningDataModule for BoseHubbard2dDataset."""

from functools import partial
from typing import Callable, Literal

import lightning.pytorch as pl
from attrs import define, field
from torch.utils.data import DataLoader, Sampler, Subset

from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbard2dDataset
from dmb.data.collate import collate_sizes
from dmb.data.sampler import MDuplicatesPerBatchSampler
from dmb.data.split import Split
from dmb.data.utils import chain_fns
from dmb.logging import create_logger

log = create_logger(__name__)


@define(hash=False, eq=False)
class BoseHubbard2dDataModule(pl.LightningDataModule):
    """LightningDataModule for BoseHubbard2dDataset."""

    dataset: BoseHubbard2dDataset
    split: Split
    batch_size: int
    num_workers: int
    pin_memory: bool
    batch_sampler: dict[Literal["train", "val", "test"], partial[Sampler]] = field(
        factory=lambda: {
            "train": partial(MDuplicatesPerBatchSampler, n_duplicates=1),
            "val": partial(MDuplicatesPerBatchSampler, n_duplicates=1),
            "test": partial(MDuplicatesPerBatchSampler, n_duplicates=1),
        }
    )

    stage_subsets: dict[str, Subset[BoseHubbard2dDataset]] = field(init=False)

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def get_collate_fn(self) -> Callable:
        """Get the collate function for the dataset."""

        collate_fns: list[Callable] = [collate_sizes]

        return chain_fns(collate_fns)

    def setup(self, stage: str = "fit") -> None:
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["train"],
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
            batch_sampler=self.batch_sampler["train"](
                dataset=self.stage_subsets["train"],
                batch_size=self.batch_size,
                shuffle=True,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["val"],
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
            batch_sampler=self.batch_sampler["val"](
                dataset=self.stage_subsets["val"],
                batch_size=self.batch_size,
                shuffle=False,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
            batch_sampler=self.batch_sampler["test"](
                dataset=self.stage_subsets["test"],
                batch_size=self.batch_size,
                shuffle=False,
            ),
        )
