"""Fake dataset with ids and indices for testing purposes."""

from typing import Any, Iterable

from attrs import define, field

from dmb.data.dataset import IdDataset


def validate_same_length(instance: Any, attribute: Any, value: Any) -> None:  # pylint: disable=unused-argument
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
