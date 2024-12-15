"""Tests for losses and metrics."""

import torch
from pytest_cases import case, parametrize_with_cases

from dmb.data.collate import MultipleSizesBatch
from dmb.model.loss import MSELoss
from dmb.model.metrics import MSE


@case(id="random_predictions_targets_pair")
def case_random_predictions_targets_pair(
) -> tuple[list[torch.Tensor], MultipleSizesBatch]:
    """Return a random prediction and target pair."""
    sizes = [4, 8, 12]

    predictions = [torch.rand(5, size, size, requires_grad=False) for size in sizes]
    targets = [torch.rand(5, size, size, requires_grad=False) for size in sizes]
    batch = MultipleSizesBatch(inputs=[],
                               outputs=targets,
                               sample_ids=[],
                               group_elements=[],
                               size=len(sizes))
    return predictions, batch


@parametrize_with_cases("predictions, batch",
                        cases=[case_random_predictions_targets_pair])
def test_mse_loss_and_metric_equal(predictions: list[torch.Tensor],
                                   batch: MultipleSizesBatch) -> None:
    """Test that the MSE loss and metric are equal."""
    loss = MSELoss()
    metric = MSE()

    loss_value = loss(predictions, batch).loss

    metric.update(predictions, batch.outputs)
    metric_value = metric.compute()

    assert torch.allclose(loss_value, metric_value, atol=1e-6)
