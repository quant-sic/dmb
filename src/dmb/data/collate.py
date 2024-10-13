from typing import TypedDict

import numpy as np
import torch

from dmb.data.dataset import DMBData


class MultipleSizesBatch(TypedDict):
    """A batch of inputs and outputs with multiple sizes."""
    inputs: list[torch.Tensor]
    outputs: list[torch.Tensor]


def collate_sizes(batch: list[DMBData]) -> MultipleSizesBatch:
    """Collate a batch of inputs and outputs by size.

    Args:
        batch: Iterable of tuples of input and output tensors.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of lists of input and output tensors.
            Each list item contains a batch of inputs or outputs of the same size.
    """
    sizes = np.array(
        [data_sample["inputs"].shape[-1] for data_sample in batch])

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

    return MultipleSizesBatch(inputs=size_batches_in, outputs=size_batches_out)
