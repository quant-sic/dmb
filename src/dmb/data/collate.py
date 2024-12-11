"""Collate functionality for data loading."""
from __future__ import annotations

import numpy as np
import torch
from attrs import frozen

from dmb.data.dataset import DMBData
from dmb.data.transforms import GroupElement


@frozen
class MultipleSizesBatch:
    """A batch of inputs and outputs with multiple sizes."""

    size: int
    inputs: list[torch.Tensor]
    outputs: list[torch.Tensor]
    sample_ids: list[list[str]]
    group_elements: list[list[list[GroupElement]]]

    def to(self, device: torch.device) -> MultipleSizesBatch:
        """Move the batch to a device."""
        return MultipleSizesBatch(
            size=self.size,
            inputs=[input_tensor.to(device) for input_tensor in self.inputs],
            outputs=[output_tensor.to(device) for output_tensor in self.outputs],
            sample_ids=self.sample_ids,
            group_elements=self.group_elements,
        )


def collate_sizes(samples: list[DMBData]) -> MultipleSizesBatch:
    """Collate a batch of inputs and outputs by size.

    Args:
        batch: Iterable of tuples of input and output tensors.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of lists of input and output tensors.
            Each list item contains a batch of inputs or outputs of the same size.
    """
    sizes = np.array([data_sample.inputs.shape[-1] for data_sample in samples])

    size_batches_in = []
    size_batches_out = []
    size_batches_sample_ids = []
    size_batches_group_elements = []
    for size in set(sizes):
        sample_indices = np.argwhere(sizes == size).flatten()
        size_batch_in, size_batch_out = map(
            lambda array: torch.from_numpy(np.stack(array)).float(),
            zip(*[(samples[sample_idx].inputs, samples[sample_idx].outputs)
                  for sample_idx in sample_indices]),
        )

        size_batches_sample_ids.append(
            [samples[sample_idx].sample_id for sample_idx in sample_indices])
        size_batches_group_elements.append(
            [samples[sample_idx].group_elements for sample_idx in sample_indices])

        size_batches_in.append(size_batch_in)
        size_batches_out.append(size_batch_out)

    return MultipleSizesBatch(inputs=size_batches_in,
                              outputs=size_batches_out,
                              sample_ids=size_batches_sample_ids,
                              group_elements=size_batches_group_elements,
                              size=len(samples))
