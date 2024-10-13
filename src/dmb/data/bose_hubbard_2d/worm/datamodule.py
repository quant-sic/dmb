from typing import Literal

import lightning.pytorch as pl
from attrs import define
from torch.utils.data import DataLoader

from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbard2dDataset
from dmb.data.split import Split
from dmb.data.utils import chain_fns, collate_sizes
from dmb.logging import create_logger

log = create_logger(__name__)


@define(hash=False, eq=False)
class BoseHubbard2dDataModule(pl.LightningDataModule):

    dataset: BoseHubbard2dDataset
    split: Split
    batch_size: int
    num_workers: int
    pin_memory: bool

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        self.stage_subsets: dict[str, BoseHubbard2dDataset]

    def get_collate_fn(self) -> Callable:
        collate_fns: list[Callable] = [collate_sizes]

        return chain_fns(collate_fns)

    def setup(self, stage: Literal["fit", "test", "predict"]) -> None:
        self.stage_subsets = self.split.apply(self.dataset)

        if stage == "fit":
            self.dataset.transforms.mode = "train"
        else:
            self.dataset.transforms.mode = "base"

        log.info("Setup for stage: %s", stage)
        log.info("Dataset sizes: %s", {
            k: len(v)
            for k, v in self.stage_subsets.items()
        })
        log.info("Dataset transforms: %s", self.dataset.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.stage_subsets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn(),
        )
