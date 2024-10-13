"""Utility functions for data processing."""

from functools import reduce
from typing import Iterable

import numpy as np
import torch

from dmb.logging import create_logger

log = create_logger(__name__)


def chain_fns(functions: Iterable[callable]) -> callable:
    """Chain multiple functions together.

    Args:
        fns: Iterable of functions to chain together.
    
    Returns:
        callable: A function that chains the input functions together.
    """
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def collate_sizes(
    batch: Iterable[tuple(torch.Tensor, torch.Tensor)]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Collate a batch of inputs and outputs by size.

    Args:
        batch: Iterable of tuples of input and output tensors.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of lists of input and output tensors.
            Each list item contains a batch of inputs or outputs of the same size.
    """
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
