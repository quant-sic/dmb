from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sized, cast

import hydra
import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.transforms import Compose

from dmb.data.utils import random_seeded_split
from dmb.utils import create_logger

log = create_logger(__name__)


class DataModuleMixin(pl.LightningDataModule, ABC):
    """Mixin class for data modules.

    It is used to define the interface for data modules.
    """

    data_train: Optional[Subset] = None
    data_val: Optional[Subset] = None
    data_test: Optional[Subset] = None

    split_datasets: List[Subset] = []
    dataset: Optional[Dataset] = None

    @abstractmethod
    def get_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_collate_fn(self) -> Optional[Callable]:
        pass

    @property
    def observables(self) -> List[str]:
        if self.dataset is None:
            raise ValueError("Data not loaded. Call `setup()` method  first.")
        return self.dataset.observables

    # data transformations
    @property
    def base_transforms(self) -> Compose:
        return (
            hydra.utils.instantiate(self.hparams["base_transforms"])
            if not isinstance(self.hparams["base_transforms"], Compose)
            else self.hparams["base_transforms"]
        )

    @property
    def train_transforms(self) -> Compose:
        return (
            hydra.utils.instantiate(self.hparams["train_transforms"])
            if not isinstance(self.hparams["train_transforms"], Compose)
            else self.hparams["train_transforms"]
        )

    @property
    def num_classes(self) -> int:
        return len(self.observables)

    @property
    def classes(self) -> List[str]:
        return self.observables

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like
        random split twice!
        """

        # load and split datasets only if not loaded already
        if not len(self.split_datasets) == len(self.hparams["train_val_test_split"]):
            self.dataset: Dataset = self.get_dataset()

            split_datasets = random_seeded_split(
                self.hparams["train_val_test_split"],
                self.dataset,
                seed=42,
                num_split_versions=self.hparams["num_split_versions"]
                if "num_split_versions" in self.hparams
                else 1,
                split_version_id=self.hparams["split_version_id"]
                if "split_version_id" in self.hparams
                else 0,
                resplit=self.hparams["resplit"] if "resplit" in self.hparams else None,
            )

            self.split_datasets = split_datasets

            # get expected number of split datasets. Right not only one layer of resplitting is supported
            expected_num_split_datasets = len(self.hparams["train_val_test_split"])
            if self.hparams["resplit"] is not None:
                for _resplit in self.hparams["resplit"]:
                    if _resplit is not None:
                        expected_num_split_datasets += (
                            len(_resplit["train_val_test_split"]) - 1
                        )

            if not len(split_datasets) == expected_num_split_datasets:
                raise RuntimeError(
                    f"Unexpected number of split datasets {len(split_datasets)}"
                )

        # check dataset is not None
        if self.dataset is None:
            raise ValueError("Dataset is None.")

        if any(
            idx < 0 or idx > len(self.split_datasets)
            for idx in self.hparams["split_usage"].values()
        ):
            raise ValueError(f"Invalid split usage: {self.hparams['split_usage']}")

        self.data_train, self.data_val, self.data_test = (
            self.split_datasets[self.hparams["split_usage"]["train"]],
            self.split_datasets[self.hparams["split_usage"]["val"]],
            self.split_datasets[self.hparams["split_usage"]["test"]],
        )

        self.check_loaded_consistency()

        # turn on data transforms if needed
        if stage not in ["fit", "test", "validate"]:
            raise ValueError(f"Invalid stage: {stage}")

        if stage in ["fit", "validate"]:
            self.dataset.apply_train_transforms = True
        if stage in ["test"]:
            self.dataset.apply_train_transforms = False

        # log stage
        log.info("Moving to stage: " + str(stage))
        log.info(
            f"Train dataset size: {len(cast(Sized,self.data_train))}; Val dataset size: {len(cast(Sized,self.data_val))}; Test dataset size: {len(cast(Sized,self.data_test))}."
        )
        # log data transforms
        log.info(
            f"\n Data transforms: {self.base_transforms if self.base_transforms is not None else 'None'}. \n Train transforms: {self.train_transforms if self.train_transforms is not None else 'None'} are applied to data: {self.dataset.apply_train_transforms}"
        )

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None:
            raise ValueError(
                "Data not loaded. Call `setup()` method or pass `train` to `setup()` method."
            )

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
            collate_fn=self.get_collate_fn(),
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.data_val is None:
            raise ValueError(
                "Data not loaded. Call `setup()` method or pass `val` to `setup()` method."
            )

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            collate_fn=self.get_collate_fn(),
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.data_test is None:
            raise ValueError(
                "Data not loaded. Call `setup()` method or pass `test` to `setup()` method."
            )

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
            collate_fn=self.get_collate_fn(),
            persistent_workers=True,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        state_dict: Dict[str, Any] = {"classes": self.classes}

        if self.hparams["save_split_indices"]:
            split_indices = {}
            split_ids = {}
            for stage, dataset in zip(
                ("train", "val", "test"),
                (self.data_train, self.data_val, self.data_test),
            ):
                if dataset is not None:  # needed because of mypy
                    split_indices[stage] = dataset.indices
                    split_ids[stage] = [
                        dataset.dataset.get_dataset_ids_from_indices(idx)
                        for idx in dataset.indices
                    ]

            state_dict["split_indices"] = split_indices
            state_dict["split_ids"] = split_ids

        # save dataset usage
        state_dict["split_usage"] = self.hparams["split_usage"]

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""

        if "split_indices" in state_dict:
            self._previous_split_indices = state_dict["split_indices"]

        if "split_ids" in state_dict:
            self._previous_split_ids = state_dict["split_ids"]

    def check_loaded_consistency(self):
        if (
            hasattr(self, "_previous_split_indices")
            and self._previous_split_indices is not None
        ):
            for stage, dataset in zip(
                ("train", "val", "test"),
                (self.data_train, self.data_val, self.data_test),
            ):
                if dataset is not None:
                    if (
                        hasattr(self, "_previous_split_indices")
                        and self._previous_split_indices is not None
                    ):
                        if not np.array_equal(
                            dataset.indices, self._previous_split_indices[stage]
                        ):
                            raise RuntimeError(
                                f"Loaded indices for {stage} dataset are not consistent with previous ones."
                            )

                    if (
                        hasattr(self, "_previous_split_ids")
                        and self._previous_split_ids is not None
                    ):
                        if not np.array_equal(
                            [
                                dataset.dataset.get_dataset_id_from_index(idx)
                                for idx in dataset.indices
                            ],
                            self._previous_split_ids[stage],
                        ):
                            raise RuntimeError(
                                f"Loaded ids for {stage} dataset are not consistent with previous ones."
                            )
        else:
            log.warning("No previous split indices found. Skipping consistency check.")
