"""Tests for the MDuplicatesPerBatchSampler class."""

from collections import Counter
from typing import Any, Iterator

from attrs import frozen
from pytest_cases import case
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from dmb.data.sampler import Dataset, MDuplicatesPerBatchSampler


@frozen
class FakeDataset(Dataset):
    """Fake dataset for testing."""

    data: list[Any]

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)


class DatasetCases:
    """Cases for the FakeDataset."""

    def case_alphabet(self) -> FakeDataset:
        """Return a dataset with the alphabet."""
        return FakeDataset(data=list('abcdefghijklmnopqrstuvwxyz'))


class BatchsizeCases:
    """Cases for the batch size."""

    @staticmethod
    def case_batchsize_5() -> int:
        """Return a batch size of 5."""
        return 5

    @staticmethod
    def case_batchsize_10() -> int:
        """Return a batch size of 10."""
        return 10

    @staticmethod
    def case_batchsize_3() -> int:
        """Return a batch size of 3."""
        return 3

    @staticmethod
    def case_batchsize_13() -> int:
        """Return a batch size of 13."""
        return 13


class NDuplicatesCases:
    """Cases for the number of duplicates."""

    @staticmethod
    def case_n_duplicates_1() -> int:
        """Return 1 duplicate."""
        return 1

    @staticmethod
    def case_n_duplicates_2() -> int:
        """Return 2 duplicates."""
        return 2

    @staticmethod
    def case_n_duplicates_3() -> int:
        """Return 3 duplicates."""
        return 3


class ShuffleCases:
    """Cases for shuffling."""

    @staticmethod
    def case_shuffle() -> bool:
        """Return True for shuffling."""
        return True

    @staticmethod
    def case_no_shuffle() -> bool:
        """Return False for no shuffling."""
        return False


class DropLastCases:
    """Cases for dropping the last batch."""

    @staticmethod
    def case_drop_last() -> bool:
        """Return True."""
        return True

    @staticmethod
    def case_no_drop_last() -> bool:
        """Return False."""
        return False


class SamplerCases:
    """Cases for the MDuplicatesPerBatchSampler."""

    @staticmethod
    @case(tags=['batchsize'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("batch_size", cases=BatchsizeCases)
    @parametrize_with_cases("n_duplicates", cases=NDuplicatesCases)
    def case_batchsizes(
            dataset: FakeDataset, batch_size: int,
            n_duplicates: int) -> tuple[MDuplicatesPerBatchSampler, list[int]]:
        """Return a sampler with the allowed batch sizes."""

        allowed_batch_sizes = [batch_size] + ([
            (len(dataset) * n_duplicates) % batch_size
        ] if (len(dataset) * n_duplicates) % batch_size != 0 else [])
        sampler = MDuplicatesPerBatchSampler(dataset,
                                             n_duplicates=n_duplicates,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False)

        return sampler, allowed_batch_sizes

    @staticmethod
    @case(tags=['batchsize'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("batch_size", cases=BatchsizeCases)
    @parametrize_with_cases("n_duplicates", cases=NDuplicatesCases)
    def case_batchsizes_drop_last(
            dataset: Dataset, batch_size: int,
            n_duplicates: int) -> tuple[MDuplicatesPerBatchSampler, list[int]]:
        """Return a sampler with the allowed batch sizes. Drop the last batch."""

        allowed_batch_sizes = [batch_size]
        sampler = MDuplicatesPerBatchSampler(dataset,
                                             n_duplicates=n_duplicates,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=True)

        return sampler, allowed_batch_sizes

    @staticmethod
    @case(tags=['complete'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("batch_size", cases=BatchsizeCases)
    @parametrize_with_cases("n_duplicates", cases=NDuplicatesCases)
    @parametrize_with_cases("shuffle", cases=ShuffleCases)
    def case_complete(dataset: FakeDataset, batch_size: int, n_duplicates: int,
                      shuffle: bool) -> tuple[MDuplicatesPerBatchSampler, FakeDataset]:
        """Return a sampler that covers the complete dataset."""
        return MDuplicatesPerBatchSampler(dataset,
                                          n_duplicates=n_duplicates,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          drop_last=False), dataset

    @staticmethod
    @case(tags=['shuffled'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("shuffle", cases=ShuffleCases)
    def case_shuffled(
            dataset: FakeDataset,
            shuffle: bool) -> tuple[MDuplicatesPerBatchSampler, FakeDataset, bool]:
        """Return a sampler that returns shuffled indices."""
        return MDuplicatesPerBatchSampler(dataset,
                                          n_duplicates=1,
                                          batch_size=1,
                                          shuffle=shuffle,
                                          drop_last=False), dataset, shuffle

    @staticmethod
    @case(tags=['duplicated'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("n_duplicates", cases=NDuplicatesCases)
    @parametrize_with_cases("shuffle", cases=ShuffleCases)
    @parametrize_with_cases("drop_last", cases=DropLastCases)
    def case_duplicated(
            dataset: FakeDataset, n_duplicates: int, shuffle: bool,
            drop_last: bool) -> tuple[MDuplicatesPerBatchSampler, int, bool]:
        """Return a sampler that returns duplicated indices."""
        return MDuplicatesPerBatchSampler(dataset,
                                          n_duplicates=n_duplicates,
                                          batch_size=1,
                                          shuffle=shuffle,
                                          drop_last=drop_last), n_duplicates, drop_last

    @staticmethod
    @case(tags=['length'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("n_duplicates", cases=NDuplicatesCases)
    @parametrize_with_cases("batch_size", cases=BatchsizeCases)
    @parametrize_with_cases("shuffle", cases=ShuffleCases)
    @parametrize_with_cases("drop_last", cases=DropLastCases)
    def case_length(dataset: FakeDataset, n_duplicates: int, batch_size: int,
                    shuffle: bool,
                    drop_last: bool) -> tuple[MDuplicatesPerBatchSampler, int]:
        """Return a sampler with the expected length."""
        if drop_last:
            expected_length = (len(dataset) * n_duplicates // batch_size) * batch_size
        else:
            expected_length = len(dataset) * n_duplicates

        return MDuplicatesPerBatchSampler(dataset,
                                          n_duplicates=n_duplicates,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          drop_last=drop_last), int(expected_length)


class TestNDuplicatesPerBatchSampler:
    """Tests for the MDuplicatesPerBatchSampler class."""

    @staticmethod
    @parametrize_with_cases("sampler, allowed_batch_sizes",
                            cases=SamplerCases,
                            filter=ft.has_tag('batchsize'))
    def test_batchsize(sampler: MDuplicatesPerBatchSampler,
                       allowed_batch_sizes: list[int]) -> None:
        """Test that the batch sizes are as expected."""
        batch_sizes = set(len(batch) for batch in sampler)
        assert batch_sizes == set(allowed_batch_sizes)

    @staticmethod
    @parametrize_with_cases("sampler, dataset",
                            cases=SamplerCases,
                            filter=ft.has_tag('complete'))
    def test_complete(sampler: MDuplicatesPerBatchSampler,
                      dataset: FakeDataset) -> None:
        """Test that the complete dataset is covered."""
        data = []
        for batch in sampler:
            data.extend(batch)
        assert set(data) == set(range(len(dataset)))

    @staticmethod
    @parametrize_with_cases("sampler, dataset, shuffle",
                            cases=SamplerCases,
                            filter=ft.has_tag('shuffled'))
    def test_shuffled(sampler: MDuplicatesPerBatchSampler, dataset: FakeDataset,
                      shuffle: bool) -> None:
        """Test that the indices are shuffled."""
        data = []
        for batch in sampler:
            data.extend(batch)
        assert (data != list(range(
            len(dataset)))) == shuffle  # not impossible but very unlikely

    @staticmethod
    @parametrize_with_cases("sampler, n_duplicates, drop_last",
                            cases=SamplerCases,
                            filter=ft.has_tag('duplicated'))
    def test_duplicated(sampler: MDuplicatesPerBatchSampler, n_duplicates: int,
                        drop_last: bool) -> None:
        """Test that the indices are duplicated."""
        data = []
        for batch in sampler:
            data.extend(batch)

        if not drop_last:
            assert all(count == n_duplicates for count in Counter(data).values())
        else:
            # maximum one index can have a smaller count
            counts = Counter(data).values()
            assert all(count == n_duplicates for count in counts
                       if count != min(counts))

    @staticmethod
    @parametrize_with_cases("sampler, expected_length",
                            cases=SamplerCases,
                            filter=ft.has_tag('length'))
    def test_length(sampler: MDuplicatesPerBatchSampler, expected_length: int) -> None:
        """Test that the length is as expected."""
        data = []
        for batch in sampler:
            data.extend(batch)

        assert len(data) == expected_length
