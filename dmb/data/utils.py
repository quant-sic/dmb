from torch.utils.data import Dataset, Subset, random_split
from typing import Dict, Optional, Sequence, List, Union, Tuple
import torch
from torch.utils.data import Dataset
from dmb.utils import create_logger
from typing import Dict, Optional, Sequence, List, Iterable, Any, Callable
from functools import reduce
import numpy as np

log = create_logger(__name__)


def random_seeded_split(
    train_val_test_split: Sequence[float],
    dataset: Dataset,
    seed: int = 42,
    split_version_id: int = 0,
    num_split_versions: int = 1,
    resplit: Optional[List[Dict]] = None,
) -> List[Subset]:
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

    split_datasets_out: List[Subset] = []
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
                    )
                )
    else:
        split_datasets_out = split_datasets

    if (
        not len(
            set.intersection(
                *[set(_dataset.indices) for _dataset in split_datasets_out]
            )
        )
        == 0
    ):
        raise ValueError(
            "Split datasets are not disjoint. Intersections: {}".format(
                set.intersection(
                    *[set(_dataset.indices) for _dataset in split_datasets_out]
                )
            )
        )

    # check that ids are disjoint
    if (
        not len(
            set.intersection(
                *[
                    set(
                        [
                            og_dataset.get_dataset_ids_from_indices(idx)
                            for idx in _dataset.indices
                        ]
                    )
                    for _dataset in split_datasets_out
                ]
            )
        )
        == 0
    ):
        raise ValueError(
            "Split datasets have overlapping ids. Intersections: {}".format(
                set.intersection(
                    *[
                        set(
                            [
                                og_dataset.get_dataset_ids_from_indices(idx)
                                for idx in _dataset.indices
                            ]
                        )
                        for _dataset in split_datasets_out
                    ]
                )
            )
        )

    if not sum([len(_dataset) for _dataset in split_datasets_out]) == dataset_length:
        raise ValueError(
            "Split datasets do not add up to original dataset. Lengths: {}. Sum: {}. Dataset Length: {}.".format(
                [len(_dataset) for _dataset in split_datasets_out],
                sum([len(_dataset) for _dataset in split_datasets_out]),
                dataset_length,
            )
        )

    return split_datasets_out


def chain_fns(fns: Iterable[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    return reduce(lambda f, g: lambda x: g(f(x)), fns)


def collate_sizes(batch):
    sizes = np.array([input_.shape[-1] for input_, _ in batch])

    size_batches_in = []
    size_batches_out = []
    for size in set(sizes):
        size_batch_in, size_batch_out = map(
            lambda array: torch.from_numpy(np.stack(array)).float(),
            zip(
                *[
                    batch[sample_idx]
                    for sample_idx in np.argwhere(sizes == size).flatten()
                ]
            ),
        )

        size_batches_in.append(size_batch_in)
        size_batches_out.append(size_batch_out)

    return size_batches_in, size_batches_out


# Transorm for symmetry of the square
class SquareSymmetryGroupAugmentations(object):
    def __call__(
        self, xy: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(xy, tuple):
            x = xy[0]
            y = xy[1]

        else:
            x = xy
            y = None

        def map_if_not_none(
            fn: Callable[[torch.Tensor], torch.Tensor], x: Optional[torch.Tensor]
        ) -> Optional[torch.Tensor]:
            if x is None:
                return None
            else:
                return fn(x)

        # with p=1/8 each choose one symmetry transform at random and apply it
        rnd = np.random.rand()

        if rnd < 1 / 8:  # unity
            pass
        elif rnd < 2 / 8:  # rotate 90 left
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.rot90(x, 1, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 3 / 8:  # rotate 180 left
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.rot90(x, 2, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 4 / 8:  # rotate 270 left
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.rot90(x, 3, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 5 / 8:  # flip x
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.flip(x, [-2]), xy), (x, y)
            )
        elif rnd < 6 / 8:  # flip y
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.flip(x, [-1]), xy), (x, y)
            )
        elif rnd < 7 / 8:  # reflection x=y
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.transpose(x, -2, -1), xy),
                (x, y),
            )
        else:  # reflection x=-y
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.flip(torch.transpose(x, -2, -1), [-2, -1]), xy
                ),
                (x, y),
            )

        if y is None:
            return x
        else:
            return x, y

    def __repr__(self):
        return self.__class__.__name__ + "()"


class TupleWrapper(object):
    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.transform = transform

    def __call__(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(x, tuple):
            return self.transform(x[0]), x[1]
        else:
            return self.transform(x)
