"""Suggested train scripts format."""

from typing import List, Optional, Tuple

import hydra
import lightning.pytorch as pl
from lightning.pytorch import Callback, LightningDataModule, LightningModule, \
    Trainer
from omegaconf import DictConfig

from dmb.utils import REPO_ROOT, create_logger

log = create_logger(__name__)


@hydra.main(
    version_base="1.2",
    config_path=str(REPO_ROOT / "dmb/experiments/configs"),
    config_name="train.yaml",
)
def train(cfg: DictConfig):

    pl.seed_everything(cfg.seed, workers=True)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer,
                                               callbacks=callbacks,
                                               logger=logger)
    lit_model: LightningModule = hydra.utils.instantiate(cfg.model)

    trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path=cfg.get("ckpt_path"))
    trainer.test(model=model,
                 datamodule=datamodule,
                 ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    train()
