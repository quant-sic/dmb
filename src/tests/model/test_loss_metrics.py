"""Tests for losses and metrics."""

import torch
from pytest_cases import case, parametrize_with_cases

from dmb.model.loss import MSELoss
from dmb.model.metrics import MSE


@case(id="random_predictions_targets_pair")
def case_random_predictions_targets_pair(
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return a random prediction and target pair."""
    sizes = [4, 8, 12]

    predictions = [torch.rand(5, size, size, requires_grad=False) for size in sizes]
    targets = [torch.rand(5, size, size, requires_grad=False) for size in sizes]
    return predictions, targets


@parametrize_with_cases("predictions, targets",
                        cases=[case_random_predictions_targets_pair])
def test_mse_loss_and_metric_equal(predictions: list[torch.Tensor],
                                   targets: list[torch.Tensor]) -> None:
    """Test that the MSE loss and metric are equal."""
    loss = MSELoss()
    metric = MSE()

    loss_value = loss(predictions, targets)

    metric.update(predictions, targets)
    metric_value = metric.compute()

    assert loss_value == metric_value
