from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Type,
    Union,
    cast,
    list,
)

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from dmb.data.dim_2.worm.helpers.sim import WormSimulation
from dmb.utils import create_logger

log = create_logger(__name__)


def random_seeded_split(
    train_val_test_split: Sequence[float],
    dataset: Dataset,
    seed: int = 42,
    split_version_id: int = 0,
    num_split_versions: int = 1,
    resplit: Optional[list[Dict]] = None,
) -> list[Subset]:
    """Splits a dataset into train, val and test subsets with a fixed seed.

    Args:
        train_val_test_split (Sequence[float]): Sequence of floats that sum to 1.0. The first element is the fraction of the dataset that will be used for training, the second for validation and the third for testing.
        dataset (Dataset): Dataset to split.
        seed (int): Seed for the random number generator.
    """

    if not hasattr(dataset, "__len__"):
        raise ValueError("Dataset must be Sized")
    else:
        dataset_length = len(dataset)

    if isinstance(dataset, Subset):
        og_dataset = dataset.dataset
        dataset_indices = dataset.indices
    else:
        og_dataset = dataset
        dataset_indices = list(range(dataset_length))

    log.info("Splitting dataset with default indices")

    sample_numbers = [int(f * float(dataset_length)) for f in train_val_test_split]
    sample_numbers[-1] = dataset_length - sum(sample_numbers[:-1])

    split_datasets = random_split(
        dataset=dataset,
        lengths=sample_numbers,
        generator=torch.Generator().manual_seed(seed),
    )

    split_datasets_out: list[Subset] = []
    if resplit is not None:
        if not len(resplit) == len(split_datasets):
            raise ValueError("resplit must have the same length as split_datasets")

        for idx, (split_config, _dataset) in enumerate(zip(resplit, split_datasets)):
            if split_config is None:
                split_datasets_out.append(_dataset)
            else:
                split_datasets_out.extend(
                    random_seeded_split(
                        dataset=_dataset,
                        **split_config,
                    ))
    else:
        split_datasets_out = split_datasets

    if (not len(
            set.intersection(
                *[set(_dataset.indices) for _dataset in split_datasets_out])) == 0):
        raise ValueError("Split datasets are not disjoint. Intersections: {}".format(
            set.intersection(
                *[set(_dataset.indices) for _dataset in split_datasets_out])))

    # check that ids are disjoint
    if (not len(
            set.intersection(*[
                set([
                    og_dataset.get_dataset_id_from_index(idx)
                    for idx in _dataset.indices
                ]) for _dataset in split_datasets_out
            ])) == 0):
        raise ValueError(
            "Split datasets have overlapping ids. Intersections: {}".format(
                set.intersection(*[
                    set([
                        og_dataset.get_dataset_id_from_index for idx in _dataset.indices
                    ]) for _dataset in split_datasets_out
                ])))

    if not sum([len(_dataset) for _dataset in split_datasets_out]) == dataset_length:
        raise ValueError(
            "Split datasets do not add up to original dataset. Lengths: {}. Sum: {}. Dataset Length: {}."
            .format(
                [len(_dataset) for _dataset in split_datasets_out],
                sum([len(_dataset) for _dataset in split_datasets_out]),
                dataset_length,
            ))

    return split_datasets_out


class BoseHubbardDataset(Dataset):

    def __init__(self, data_dir: Path, data_transform=None, clean=True):
        self.data_dir = data_dir
        self.data_transform = data_transform

        self.clean = clean

    @cached_property
    def sim_dirs(self):
        sim_dirs = sorted(self.data_dir.glob("*"))

        if self.clean:
            sim_dirs = self._clean_sim_dirs(sim_dirs)

        return sim_dirs

    @staticmethod
    def _clean_sim_dirs(sim_dirs):

        def filter_fn(sim_dir):
            sim = WormSimulation.from_dir(sim_dir)

            try:
                sim.results.observables["Density_Distribution"]["mean"]["value"]
                valid = True
            except:
                valid = False

            return valid

        sim_dirs = list(filter(filter_fn, sim_dirs))

        return sim_dirs

    def __len__(self):
        return len(self.sim_dirs)

    def __getitem__(self, idx):
        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        inputs = sim.input_parameters.mu
        outputs = sim.results.observables

        density = outputs["Density_Distribution"]["mean"]["value"]
        corr_1 = outputs["DensDens_CorrFun"]["mean"]["value"]
        return inputs, outputs

    def get_sim(self, idx):
        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        return sim

    def get_parameters(self, idx):
        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        return sim.input_parameters


class BoseHubbardDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 0,
        clean=True,
        base_transforms=None,
        train_transforms=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clean = clean

    def setup(self, stage=None):
        # load and split datasets only if not loaded already
        if not len(self.split_datasets) == len(self.hparams["train_val_test_split"]):
            self.dataset: Dataset = self.get_dataset()

            split_datasets = random_seeded_split(
                self.hparams["train_val_test_split"],
                self.dataset,
                seed=42,
                mode=self.hparams["split_mode"]
                if "split_mode" in self.hparams else "random",
                num_split_versions=self.hparams["num_split_versions"]
                if "num_split_versions" in self.hparams else 1,
                split_version_id=self.hparams["split_version_id"]
                if "split_version_id" in self.hparams else 0,
                num_split_clusters=self.hparams["num_split_clusters"]
                if "num_split_clusters" in self.hparams else 1,
                resplit=self.hparams["resplit"] if "resplit" in self.hparams else None,
            )

            self.split_datasets = split_datasets

            # get expected number of split datasets. Right not only one layer of resplitting is supported
            expected_num_split_datasets = len(self.hparams["train_val_test_split"])
            if self.hparams["resplit"] is not None:
                for _resplit in self.hparams["resplit"]:
                    if _resplit is not None:
                        expected_num_split_datasets += (
                            len(_resplit["train_val_test_split"]) - 1)

            if not len(split_datasets) == expected_num_split_datasets:
                raise RuntimeError(
                    f"Unexpected number of split datasets {len(split_datasets)}")

        # check dataset is not None
        if self.dataset is None:
            raise ValueError("Dataset is None.")

        if any(idx < 0 or idx > len(self.split_datasets)
               for idx in self.hparams["split_usage"].values()):
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
            f"Train dataset size: {len(cast(Sized,self.data_train))}; Val dataset size: {len(cast(Sized,self.data_val))}; Test dataset size: {len(cast(Sized,self.data_test))}. Split with mode {self.hparams['split_mode']}."
        )
        # log data transforms
        log.info(
            f"\n Data transforms: {self.transforms if self.transforms is not None else 'None'}. \n Train transforms: {self.train_transforms if self.train_transforms is not None else 'None'} are applied to data: {self.dataset.apply_train_transforms}"
        )

    def get_dataset(self):
        return BoseHubbardDataset(self.data_dir, clean=self.clean)

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
                        dataset.dataset.get_dataset_id_from_index(idx)
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
        if (hasattr(self, "_previous_split_indices")
                and self._previous_split_indices is not None):
            for stage, dataset in zip(
                ("train", "val", "test"),
                (self.data_train, self.data_val, self.data_test),
            ):
                if dataset is not None:
                    if (hasattr(self, "_previous_split_indices")
                            and self._previous_split_indices is not None):
                        if not np.array_equal(dataset.indices,
                                              self._previous_split_indices[stage]):
                            raise RuntimeError(
                                f"Loaded indices for {stage} dataset are not consistent with previous ones."
                            )

                    if (hasattr(self, "_previous_split_ids")
                            and self._previous_split_ids is not None):
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
