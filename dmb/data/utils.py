from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, \
    Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

from dmb.utils import create_logger

log = create_logger(__name__)


def chain_fns(fns: Iterable[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    return reduce(lambda f, g: lambda x: g(f(x)), fns)


def collate_sizes(batch):
    sizes = np.array([input_.shape[-1] for input_, _ in batch])

    size_batches_in = []
    size_batches_out = []
    for size in set(sizes):
        size_batch_in, size_batch_out = map(
            lambda array: torch.from_numpy(np.stack(array)).float(),
            zip(*[
                batch[sample_idx]
                for sample_idx in np.argwhere(sizes == size).flatten()
            ]),
        )

        size_batches_in.append(size_batch_in)
        size_batches_out.append(size_batch_out)

    return size_batches_in, size_batches_out
