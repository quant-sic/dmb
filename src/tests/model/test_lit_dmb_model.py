from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torchmetrics
from lightning import Trainer
from pytest_cases import fixture, parametrize_with_cases
from torch.utils.data import Dataset

from dmb.data.collate import collate_sizes
from dmb.data.dataset import DMBData
from dmb.model.dmb_model import DMBModel
from dmb.model.lit_dmb_model import LitDMBModel
from dmb.model.loss import Loss, MSELoss
from dmb.model.metrics import MSE
from dmb.model.modules import EsSeResNet2d, ResNet2d, SeResNet2d
from dmb.scripts.train.callbacks import PlottingCallback


class DMBModelCases:
    """Test cases with different DMB models."""

    def case_se_resnet2d(self) -> DMBModel:
        """Return a DMB model with SeResNet2d module."""
        return DMBModel(observables=["density"],
                        module_list=[
                            SeResNet2d(in_channels=4,
                                       out_channels=7,
                                       kernel_sizes=[5, 3, 3, 3],
                                       n_channels=[16, 32, 32, 32],
                                       dropout=0.1)
                        ])

    def case_resnet2d(self) -> DMBModel:
        """Return a DMB model with ResNet2d module."""
        return DMBModel(observables=["density"],
                        module_list=[
                            ResNet2d(in_channels=4,
                                     out_channels=7,
                                     kernel_sizes=[5, 3, 3, 3],
                                     n_channels=[3, 3, 3, 3],
                                     dropout=0.1)
                        ])

    def case_es_resnet2d(self) -> DMBModel:
        """Return a DMB model with EsSeResNet2d module."""
        return DMBModel(observables=["density"],
                        module_list=[
                            EsSeResNet2d(in_channels=4,
                                         out_channels=7,
                                         kernel_sizes=[5, 3, 3, 3],
                                         n_channels=[16, 16, 16, 16],
                                         dropout=0.1,
                                         se_squeeze_factor=2)
                        ])


class OptimizerCases:
    """Test cases with different optimizers."""

    def case_adam(self) -> partial[torch.optim.Adam]:
        """Return a partial function for Adam optimizer."""
        return partial(torch.optim.Adam, lr=1e-3)

    def case_sgd(self) -> partial[torch.optim.SGD]:
        """Return a partial function for SGD optimizer."""
        return partial(torch.optim.SGD, lr=1e-3)


class LRSchedulerCases:
    """Test cases with different learning rate schedulers."""

    def case_step_lr(self) -> dict[str, partial[torch.optim.lr_scheduler.StepLR] | Any]:
        """Return a dictionary with StepLR scheduler."""
        return {
            "scheduler": partial(torch.optim.lr_scheduler.StepLR,
                                 step_size=200,
                                 gamma=0.5),
            "monitor": "train/mse",
            "interval": "epoch",
            "frequency": 1,
        }


class LossCases:
    """Test cases with different loss functions."""

    def case_mse(self) -> MSELoss:
        """Return a mean squared error loss."""
        return MSELoss()


class MetricsCases:
    """Test cases with different metrics."""

    def case_mse(self) -> torchmetrics.MetricCollection:
        """Return a metric collection with mean squared error."""
        return torchmetrics.MetricCollection(metrics={"mse": MSE()})


def get_dataloader() -> torch.utils.data.DataLoader:
    """Return a data loader with random DMB data."""

    def _get_random_dmb_data() -> DMBData:
        """Return random DMB data."""
        size = int(torch.randint(4, 20, (1, )).item())
        return DMBData(inputs=torch.randn(4, size, size),
                       outputs=torch.randn(7, size, size),
                       sample_id="random")

    return torch.utils.data.DataLoader(cast(
        Dataset, [_get_random_dmb_data() for _ in range(100)]),
                                       batch_size=2,
                                       collate_fn=collate_sizes)


class TrainerCases:
    """Test cases with different trainers."""

    def case_trainer_fast_dev(self, tmp_path: Path) -> Trainer:
        """Return a trainer with fast_dev_run."""
        return Trainer(fast_dev_run=True,
                       accelerator="auto",
                       accumulate_grad_batches=8,
                       deterministic=False,
                       check_val_every_n_epoch=1,
                       log_every_n_steps=1,
                       default_root_dir=tmp_path)

    def case_trainer_no_fast_dev(self, tmp_path: Path) -> Trainer:
        """Return a trainer without fast_dev_run."""
        return Trainer(fast_dev_run=False,
                       accelerator="auto",
                       max_epochs=3,
                       accumulate_grad_batches=2,
                       deterministic=False,
                       check_val_every_n_epoch=1,
                       log_every_n_steps=1,
                       default_root_dir=tmp_path,
                       callbacks=[PlottingCallback(plot_interval=2)])


class TestLitDMBModel:
    """Tests for LitDMBModel class."""

    @staticmethod
    @fixture(name="lit_dmb_model", scope="class")
    @parametrize_with_cases("dmb_model", cases=DMBModelCases)
    @parametrize_with_cases("optimizer", cases=OptimizerCases)
    @parametrize_with_cases("lr_scheduler", cases=LRSchedulerCases)
    @parametrize_with_cases("loss", cases=LossCases)
    @parametrize_with_cases("metrics", cases=MetricsCases)
    def fixture_lit_dmb_model(dmb_model: DMBModel,
                              optimizer: partial[torch.optim.Optimizer],
                              lr_scheduler: dict[
                                  str, partial[torch.optim.lr_scheduler._LRScheduler]],
                              loss: Loss,
                              metrics: torchmetrics.MetricCollection) -> LitDMBModel:
        """Return a LitDMBModel instance."""
        return LitDMBModel(dmb_model,
                           optimizer=optimizer,
                           lr_scheduler=lr_scheduler,
                           loss=loss,
                           metrics=metrics)

    @staticmethod
    @fixture(name="train_dataloader", scope="class")
    def fixture_train_dataloader() -> torch.utils.data.DataLoader:
        """Return a training data loader."""
        return get_dataloader()

    @staticmethod
    @fixture(name="val_dataloader", scope="class")
    def fixture_val_dataloader() -> torch.utils.data.DataLoader:
        """Return a validation data loader."""
        return get_dataloader()

    @staticmethod
    @parametrize_with_cases("trainer", cases=TrainerCases)
    def test_fit(lit_dmb_model: LitDMBModel, trainer: Trainer,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader) -> None:
        """Test the fit method of a trainer."""
        trainer.fit(lit_dmb_model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

    @staticmethod
    @parametrize_with_cases("trainer", cases=TrainerCases)
    def test_validate(lit_dmb_model: LitDMBModel, trainer: Trainer,
                      val_dataloader: torch.utils.data.DataLoader) -> None:
        """Test the validate method of a trainer."""
        trainer.validate(lit_dmb_model, dataloaders=val_dataloader)

    @staticmethod
    @parametrize_with_cases("trainer", cases=TrainerCases)
    def test_test(lit_dmb_model: LitDMBModel, trainer: Trainer,
                  val_dataloader: torch.utils.data.DataLoader) -> None:
        """Test the test method of a trainer."""
        trainer.test(lit_dmb_model, dataloaders=val_dataloader)
