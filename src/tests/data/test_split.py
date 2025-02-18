"""Test the split module."""

from pathlib import Path

import pytest
from pytest_cases import case, filters, parametrize_with_cases

from dmb.data.bose_hubbard_2d.worm.dataset import WormSimulationsSplitStrategy
from dmb.data.split import AllIdsEqualSplitStrategy, IdDatasetSplitStrategy, \
    Split
from tests.data.fake_id_dataset import FakeIdDataset


class SplitStrategyCases:
    """Test cases for the IdDatasetSplitStrategy."""

    @staticmethod
    @case(tags=("split_strategy", "general"))
    def case_all_ids_equal_split_strategy() -> IdDatasetSplitStrategy:
        """Return an instance of the AllIdsEqualSplitStrategy."""
        return AllIdsEqualSplitStrategy()

    @staticmethod
    @case(tags=("split_strategy", "worm"))
    def case_worm_simulations_split_strategy() -> IdDatasetSplitStrategy:
        """Return an instance of the WormSimulationsSplitStrategy."""
        return WormSimulationsSplitStrategy()


class SplitDataCases:
    """Test cases for the split data."""

    @staticmethod
    @case(tags=("id_dataset", "general"))
    def case_id_dataset() -> FakeIdDataset:
        """Return a fake id dataset."""
        return FakeIdDataset(
            ids=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"),
            indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

    @staticmethod
    @case(tags=("split_data", "general"))
    @parametrize_with_cases("id_dataset", cases=case_id_dataset, has_tag="general")
    def case_split_data_1(
        id_dataset: FakeIdDataset,
    ) -> tuple[
        FakeIdDataset,
        dict[str, list[str]],
        dict[str, list[int]],
        dict[str, float],
        dict[str, int],
    ]:
        """Return a split data dictionary."""
        return (
            id_dataset,
            {"train": ["a", "b", "i", "c"], "test": ["d", "e"]},
            {
                "train": [0, 1, 2, 8],
                "test": [3, 4],
            },
            {"train": 0.4, "test": 0.6},
            {"train": 4, "test": 6},
        )

    @staticmethod
    @case(tags=("split_data", "general"))
    @parametrize_with_cases("id_dataset", cases=case_id_dataset, has_tag="general")
    def case_split_data_2(
        id_dataset: FakeIdDataset,
    ) -> tuple[
        FakeIdDataset,
        dict[str, list[str]],
        dict[str, list[int]],
        dict[str, float],
        dict[str, int],
    ]:
        """Return a split data dictionary."""
        return (
            id_dataset,
            {
                "train": ["a", "b", "c", "d", "e", "f", "g"],
                "val": ["h", "z"],
                "test": ["i", "j"],
            },
            {
                "train": [0, 1, 2, 3, 4, 5, 6],
                "val": [7],
                "test": [8, 9],
            },
            {"train": 0.6, "val": 0.1, "test": 0.3},
            {"train": 6, "val": 1, "test": 3},
        )

    @staticmethod
    @case(tags=("split_data", "general"))
    @parametrize_with_cases("id_dataset", cases=case_id_dataset, has_tag="general")
    def case_split_data_3(
        id_dataset: FakeIdDataset,
    ) -> tuple[
        FakeIdDataset,
        dict[str, list[str]],
        dict[str, list[int]],
        dict[str, float],
        dict[str, int],
    ]:
        """Return a split data dictionary."""

        return (
            id_dataset,
            {
                "train": ["i", "j"],
                "val": ["h"],
                "test": ["a", "b", "c", "d", "e", "f", "g"],
            },
            {
                "train": [8, 9],
                "val": [7],
                "test": [0, 1, 2, 3, 4, 5, 6],
            },
            {"train": 0.2, "val": 0.1, "test": 0.7},
            {"train": 2, "val": 1, "test": 7},
        )

    @staticmethod
    @case(tags=("split_data", "general"))
    @parametrize_with_cases("id_dataset", cases=case_id_dataset, has_tag="general")
    def case_split_data_4(
        id_dataset: FakeIdDataset,
    ) -> tuple[
        FakeIdDataset,
        dict[str, list[str]],
        dict[str, list[int]],
        dict[str, float],
        dict[str, int],
    ]:
        """Return a split data dictionary."""
        return (
            id_dataset,
            {"test": ["g"]},
            {
                "test": [6],
            },
            {"test": 1.0},
            {"test": 10},
        )

    @staticmethod
    @case(tags=("split_data", "general"))
    @parametrize_with_cases("id_dataset", cases=case_id_dataset, has_tag="general")
    def case_split_data_5(
        id_dataset: FakeIdDataset,
    ) -> tuple[
        FakeIdDataset,
        dict[str, list[str]],
        dict[str, list[int]],
        dict[str, float],
        dict[str, int],
    ]:
        """Return a split data dictionary."""
        return (
            id_dataset,
            {"test": ["x"]},
            {
                "test": [],
            },
            {
                "test": 0.0,
            },
            {"test": 10},
        )


class TestSplit:
    """Test the Split class."""

    @staticmethod
    @parametrize_with_cases(
        "id_dataset, split_ids, subset_indices",
        cases=SplitDataCases,
        filter=filters.has_tag("split_data"),
    )
    def test_apply(
        id_dataset: FakeIdDataset,
        split_ids: dict[str, list[str]],
        subset_indices: dict[str, list[int]],
    ) -> None:
        """Test the apply method."""
        # for split_ids, subset_indices in zip(test_split_ids, test_subset_indices):
        split = Split(split_ids=split_ids)
        subsets = split.apply(id_dataset)

        for split_name, subset in subsets.items():
            assert len(subset) == len(subset_indices[split_name])
            assert all(
                subset[idx] == id_dataset[subset_idx]
                for idx, subset_idx in enumerate(subset_indices[split_name])
            )

    @staticmethod
    @parametrize_with_cases(
        "split_strategy",
        cases=SplitStrategyCases,
    )
    @parametrize_with_cases(
        "id_dataset, split_ids, subset_indices, split_fractions, subset_lengths",
        cases=SplitDataCases,
        filter=filters.has_tag("split_data"),
    )
    def test_generate(
        id_dataset: FakeIdDataset,
        split_ids: dict[str, list[str]],  # pylint: disable=unused-argument
        subset_indices: dict[str, list[int]],  # pylint: disable=unused-argument
        split_fractions: dict[str, float],
        subset_lengths: dict[str, int],
        split_strategy: IdDatasetSplitStrategy,
    ) -> None:
        """Test the generate method."""
        split_ids = Split.generate(
            split_fractions=split_fractions,
            dataset=id_dataset,
            split_strategy=split_strategy,
        )

        assert set(split_ids.keys()) == set(split_fractions.keys())
        assert all(
            len(split_ids[split_name]) == subset_lengths[split_name]
            for split_name in split_fractions
        )

    @staticmethod
    @parametrize_with_cases(
        "id_dataset, split_ids",
        cases=SplitDataCases,
        filter=filters.has_tag("split_data"),
    )
    def test_to_from_file(
        id_dataset: FakeIdDataset,  # pylint: disable=unused-argument
        split_ids: dict[str, list[str]],
        tmp_path: Path,
    ) -> None:
        """Test the to_file and from_file methods."""
        split = Split(split_ids=split_ids)
        split.to_file(tmp_path / "split.json")

        loaded_split = Split.from_file(tmp_path / "split.json")

        assert split.split_ids == loaded_split.split_ids

    @staticmethod
    @parametrize_with_cases("split_strategy", cases=SplitStrategyCases)
    @parametrize_with_cases(
        "id_dataset", cases=SplitDataCases, filter=filters.has_tag("id_dataset")
    )
    def test_split_fractions_check(
        id_dataset: FakeIdDataset, split_strategy: IdDatasetSplitStrategy
    ) -> None:
        """Test the split fractions check."""
        with pytest.raises(ValueError):
            Split.generate(
                split_fractions={"train": 0.5, "test": 0.6},
                dataset=id_dataset,
                split_strategy=split_strategy,
            )

        with pytest.raises(ValueError):
            Split.generate(
                split_fractions={"train": -0.5, "test": 0.5},
                dataset=id_dataset,
                split_strategy=split_strategy,
            )
