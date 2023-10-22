from typing import Optional
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
from dmb.data.utils import collate_sizes
import numpy as np
from dmb.utils.io import create_logger
from omegaconf import DictConfig
import hydra

log = create_logger(__name__)


class DataModule2d(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,
        train_val_split=0.9,
        split_version=0,
        train_set_fractions=None,
        set_accumulation=False,
        accumulation_idx=None,
        n_accumulated=10,
        shuffle=False,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size

        if isinstance(dataset, (dict, DictConfig)):
            dataset = hydra.utils.instantiate(dataset)

        self.dataset = dataset
        self.train_val_split = train_val_split
        self.shuffle = shuffle

        # for fraction training
        self.set_accumulation = set_accumulation
        self.accumulation_idx = accumulation_idx
        self.n_accumulated = n_accumulated
        self.train_set_fractions = train_set_fractions
        self.split_version = split_version

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.dataset.stage = "train"

            if not self.set_accumulation:
                self.dataset_train, self.dataset_val = map(
                    lambda indices: Subset(dataset=self.dataset, indices=indices),
                    self.dataset.split_wrt_sizes(
                        [self.train_val_split, 1 - self.train_val_split]
                    ),
                )
            else:
                if (
                    self.accumulation_idx is None
                    or self.accumulation_idx >= self.n_accumulated
                ):
                    raise ValueError("Accumulation index must be an integer")

                if self.train_set_fractions is None:
                    splits = list(
                        np.full(
                            self.n_accumulated,
                            fill_value=self.train_val_split / self.n_accumulated,
                        )
                    ) + [1 - self.train_val_split]
                else:
                    assert (
                        len(self.train_set_fractions) == self.n_accumulated
                    ), "If train set fractions are given, n_accumulated fractions need to be given"
                    if all(isinstance(f, int) for f in self.train_set_fractions):
                        log.info("Assuming all ints")
                        assert sum(self.train_set_fractions) < int(
                            self.train_val_split * len(self.dataset)
                        ), "Integer sum too large"

                        splits = (
                            [f for f in self.train_set_fractions]
                            + [
                                int(self.train_val_split * len(self.dataset))
                                - sum(self.train_set_fractions)
                            ]
                            + [int((1 - self.train_val_split) * len(self.dataset))]
                        )
                    else:
                        splits = [
                            f * self.train_val_split for f in self.train_set_fractions
                        ] + [1 - self.train_val_split]

                split_indices = self.dataset.split_wrt_sizes(
                    splits, split_version=self.split_version
                )

                subsets = tuple(
                    map(
                        lambda indices: Subset(dataset=self.dataset, indices=indices),
                        split_indices,
                    )
                )

                self.dataset_train, self.dataset_val = (
                    ConcatDataset(subsets[: (self.accumulation_idx + 1)]),
                    subsets[-1],
                )

            log.info(
                f"Create train dataset with length {len(self.dataset_train)} and val dataset with length {len(self.dataset_val)}"
            )

        if stage == "test":
            self.dataset.stage = "test"
            self.dataset_test = self.dataset

            log.info(f"Create test dataset with length {len(self.dataset_test)}")

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_sizes,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=500, num_workers=4, collate_fn=collate_sizes
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=500, num_workers=4, collate_fn=collate_sizes
        )
