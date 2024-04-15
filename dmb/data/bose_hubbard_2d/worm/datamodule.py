from typing import Literal

import lightning.pytorch as pl
from attrs import define
from torch.utils.data import DataLoader

from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbardDataset
from dmb.data.split import Split
from dmb.data.utils import chain_fns, collate_sizes
from dmb.logging import create_logger

log = create_logger(__name__)


@define(hash=False, eq=False)
class BoseHubbardDataModule(pl.LightningDataModule):

    dataset: BoseHubbardDataset
    split: Split
    batch_size: int
    num_workers: int
    pin_memory: bool

    def __attrs_pre_init__(self):
        super().__init__()

    def __attrs_post_init__(self):
        self.stage_subsets: dict[str, BoseHubbardDataset]

    def get_collate_fn(self) -> callable:
        collate_fns: list[callable] = [collate_sizes]

        return chain_fns(collate_fns)

    def setup(self, stage: Literal["fit", "test", "predict"]) -> None:

        self.stage_subsets = self.split.apply(self.dataset)

        if stage == "fit":
            self.dataset.transforms.mode = "train"
        else:
            self.dataset.transforms.mode = "base"

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
