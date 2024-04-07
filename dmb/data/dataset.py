from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class IdDataset(Dataset, ABC):

    @abstractmethod
    def get_ids_from_indices(self, indices: tuple[int,
                                                  ...]) -> tuple[str, ...]:
        ...

    @abstractmethod
    def get_indices_from_ids(self, ids: tuple[str, ...]) -> tuple[int, ...]:
        ...
