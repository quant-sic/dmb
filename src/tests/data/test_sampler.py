from collections import Counter
from typing import Any, Iterator

from attrs import frozen
from pytest_cases import case
from pytest_cases import filters as ft
from pytest_cases import parametrize_with_cases

from dmb.data.sampler import Dataset, MDuplicatesPerBatchSampler


@frozen
class FakeDataset(Dataset):

    data: list[Any]

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)


class DatasetCases:

    def case_alphabet(self) -> FakeDataset:
        return FakeDataset(data=list('abcdefghijklmnopqrstuvwxyz'))


class BatchsizeCases:

    @staticmethod
    def case_batchsize_5() -> int:
        return 5

    @staticmethod
    def case_batchsize_10() -> int:
        return 10

    @staticmethod
    def case_batchsize_3() -> int:
        return 3

    @staticmethod
    def case_batchsize_13() -> int:
        return 13


class NDuplicatesCases:

    @staticmethod
    def case_n_duplicates_1() -> int:
        return 1

    @staticmethod
    def case_n_duplicates_2() -> int:
        return 2

    @staticmethod
    def case_n_duplicates_3() -> int:
        return 3


class ShuffleCases:

    @staticmethod
    def case_shuffle() -> bool:
        return True

    @staticmethod
    def case_no_shuffle() -> bool:
        return False


class DropLastCases:

    @staticmethod
    def case_drop_last() -> bool:
        return True

    @staticmethod
    def case_no_drop_last() -> bool:
        return False


class SamplerCases:

    @staticmethod
    @case(tags=['batchsize'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("batch_size", cases=BatchsizeCases)
    @parametrize_with_cases("n_duplicates", cases=NDuplicatesCases)
    def case_batchsizes(
            dataset: FakeDataset, batch_size: int,
            n_duplicates: int) -> tuple[MDuplicatesPerBatchSampler, list[int]]:

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
        return MDuplicatesPerBatchSampler(dataset,
                                          n_duplicates=1,
                                          batch_size=1,
                                          shuffle=shuffle,
                                          drop_last=False), dataset, shuffle

    @staticmethod
    @case(tags=['duplicated'])
    @parametrize_with_cases("dataset", cases=DatasetCases)
    @parametrize_with_cases("n_duplicates", cases=NDuplicatesCases)
    @parametrize_with_cases("batch_size", cases=BatchsizeCases)
    @parametrize_with_cases("shuffle", cases=ShuffleCases)
    @parametrize_with_cases("drop_last", cases=DropLastCases)
    def case_duplicated(
            dataset: FakeDataset, n_duplicates: int, batch_size: int, shuffle: bool,
            drop_last: bool
    ) -> tuple[MDuplicatesPerBatchSampler, FakeDataset, int, bool]:
        return MDuplicatesPerBatchSampler(
            dataset,
            n_duplicates=n_duplicates,
            batch_size=1,
            shuffle=shuffle,
            drop_last=drop_last), dataset, n_duplicates, drop_last

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

    @staticmethod
    @parametrize_with_cases("sampler, allowed_batch_sizes",
                            cases=SamplerCases,
                            filter=ft.has_tag('batchsize'))
    def test_batchsize(sampler: MDuplicatesPerBatchSampler,
                       allowed_batch_sizes: list[int]) -> None:
        batch_sizes = set(len(batch) for batch in sampler)
        assert batch_sizes == set(allowed_batch_sizes)

    @staticmethod
    @parametrize_with_cases("sampler, dataset",
                            cases=SamplerCases,
                            filter=ft.has_tag('complete'))
    def test_complete(sampler: MDuplicatesPerBatchSampler,
                      dataset: FakeDataset) -> None:
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
        data = []
        for batch in sampler:
            data.extend(batch)
        assert (data != list(range(
            len(dataset)))) == shuffle  # not impossible but very unlikely

    @staticmethod
    @parametrize_with_cases("sampler, dataset, n_duplicates, drop_last",
                            cases=SamplerCases,
                            filter=ft.has_tag('duplicated'))
    def test_duplicated(sampler: MDuplicatesPerBatchSampler, dataset: FakeDataset,
                        n_duplicates: int, drop_last: bool) -> None:
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
        data = []
        for batch in sampler:
            data.extend(batch)

        assert len(data) == expected_length
