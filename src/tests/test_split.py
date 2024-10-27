from pathlib import Path
from typing import Any, Iterable

import pytest
from attrs import define, field

from dmb.data.dataset import IdDataset
from dmb.data.split import Split


def validate_same_length(instance: Any, attribute: Any, value: Any) -> None:
    """Validate that the ids and indices have the same length."""
    if len(instance.ids) != len(instance.indices):
        raise ValueError("ids and indices must have the same length")


@define
class FakeIdDataset(IdDataset):
    """Fake dataset with ids and indices."""

    ids: tuple[str, ...] = field(validator=validate_same_length)
    indices: tuple[int, ...] = field(validator=validate_same_length)

    def get_ids_from_indices(self, indices: Iterable[int]) -> tuple[str, ...]:
        """Get ids from indices.

        Indices are expected to be unique and in the range of `len(self.ids)`.
        """
        return tuple(self.ids[idx] for idx in indices)

    def get_indices_from_ids(self, ids: Iterable[str]) -> tuple[int, ...]:
        """Get indices from ids.

        Ids are not expected to be unique or a proper subset of `self.ids`.

        Args:
            ids: A tuple of ids.

        Returns:
            A tuple of indices corresponding to the ids. The indices are sorted.
        """
        contained_ids = set(self.ids).intersection(ids)
        return tuple(self.ids.index(id_) for id_ in sorted(contained_ids))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[int, str]:
        return self.indices[idx], self.ids[idx]


class TestSplit:
    """Test the Split class."""

    @staticmethod
    @pytest.fixture(scope="class", name="id_dataset")
    def get_id_dataset() -> FakeIdDataset:
        """Return a fake id dataset."""
        return FakeIdDataset(
            ids=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"),
            indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

    @staticmethod
    @pytest.fixture(scope="class", name="test_split_ids")
    def get_test_split_ids() -> tuple[dict[str, list[str]], ...]:
        """Return test split ids."""
        return (
            {
                "train": ["a", "b", "i", "c"],
                "test": ["d", "e"]
            },
            {
                "train": ["a", "b", "c", "m", "e"],
                "test": ["f", "g", "h", "i", "j"]
            },
            {
                "train": ["a", "b", "d", "e", "f", "g"],
                "val": ["h", "z"],
                "test": ["i", "j"],
            },
            {
                "train": ["i", "j"],
                "val": ["h"],
                "test": ["a", "b", "c", "d", "e", "f", "g"],
            },
            {
                "test": ["g"]
            },
            {
                "test": [
                    "x",
                ]
            },
        )

    @staticmethod
    @pytest.fixture(scope="class", name="test_subset_indices")
    def get_test_subset_indices() -> tuple[dict[str, tuple[int, ...]], ...]:
        """Return test subset indices."""
        return (
            {
                "train": (0, 1, 2, 8),
                "test": (3, 4),
            },
            {
                "train": (0, 1, 2, 4),
                "test": (5, 6, 7, 8, 9),
            },
            {
                "train": (0, 1, 3, 4, 5, 6),
                "val": (7, ),
                "test": (8, 9),
            },
            {
                "train": (8, 9),
                "val": (7, ),
                "test": (0, 1, 2, 3, 4, 5, 6),
            },
            {
                "test": (6, ),
            },
            {
                "test": (),
            },
        )

    @staticmethod
    @pytest.fixture(scope="class", name="test_split_fractions")
    def get_test_split_fractions() -> tuple[dict[str, float], ...]:
        """Return test split fractions."""
        return (
            {
                "train": 0.4,
                "test": 0.6
            },
            {
                "train": 0.5,
                "test": 0.5
            },
            {
                "train": 0.6,
                "val": 0.1,
                "test": 0.3
            },
            {
                "train": 0.2,
                "val": 0.1,
                "test": 0.7
            },
            {
                "test": 1.0
            },
            {
                "test": 0.0,
                "train": 0.8
            },
        )

    @staticmethod
    @pytest.fixture(scope="class", name="test_subset_lengths")
    def get_test_subset_lengths() -> tuple[dict[str, int], ...]:
        """Return test subset lengths."""
        return (
            {
                "train": 4,
                "test": 6
            },
            {
                "train": 5,
                "test": 5
            },
            {
                "train": 6,
                "val": 1,
                "test": 3
            },
            {
                "train": 2,
                "val": 1,
                "test": 7
            },
            {
                "test": 10
            },
            {
                "test": 0,
                "train": 8
            },
        )

    @staticmethod
    def test_apply(
        id_dataset: FakeIdDataset,
        test_split_ids: tuple[dict[str, tuple[str, ...]]],
        test_subset_indices: tuple[dict[str, tuple[int, ...]], ...],
    ) -> None:
        """Test the apply method."""
        for split_ids, subset_indices in zip(test_split_ids, test_subset_indices):
            split = Split(split_ids=split_ids)
            subsets = split.apply(id_dataset)

            for split_name, subset in subsets.items():
                assert len(subset) == len(subset_indices[split_name])
                assert all(subset[idx] == id_dataset[subset_idx]
                           for idx, subset_idx in enumerate(subset_indices[split_name]))

    @staticmethod
    def test_generate(
        id_dataset: FakeIdDataset,
        test_split_fractions: tuple[dict[str, float]],
        test_subset_lengths: tuple[dict[str, int]],
    ) -> None:
        """Test the generate method."""
        for split_fractions, subset_lengths in zip(test_split_fractions,
                                                   test_subset_lengths):
            split_ids = Split.generate(split_fractions, id_dataset)

            assert set(split_ids.keys()) == set(split_fractions.keys())
            assert all(
                len(split_ids[split_name]) == subset_lengths[split_name]
                for split_name in split_fractions)

    @staticmethod
    def test_to_from_file(
        test_split_ids: tuple[dict[str, tuple[str, ...]]],
        tmp_path: Path,
    ) -> None:
        """Test the to_file and from_file methods."""
        for split_ids in test_split_ids:
            split = Split(split_ids=split_ids)
            split.to_file(tmp_path / "split.json")

            loaded_split = Split.from_file(tmp_path / "split.json")

            assert split.split_ids == loaded_split.split_ids

    @staticmethod
    def test_split_fractions_check(id_dataset: FakeIdDataset) -> None:
        """Test the split fractions check."""
        with pytest.raises(ValueError):
            Split.generate({"train": 0.5, "test": 0.6}, id_dataset)

        with pytest.raises(ValueError):
            Split.generate({"train": -0.5, "test": 0.5}, id_dataset)
