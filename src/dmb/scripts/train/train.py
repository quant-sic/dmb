"""Suggested train scripts format."""

import hydra
import lightning.pytorch as pl
from dotenv import load_dotenv
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from dmb.paths import REPO_ROOT


@hydra.main(
    version_base="1.3",
    config_path=str(REPO_ROOT / "src/dmb/scripts/train/configs"),
    config_name="train.yaml",
)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
    logger = list(hydra.utils.instantiate(cfg.logger).values())

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )
    lit_model: LightningModule = hydra.utils.instantiate(cfg.lit_model)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    trainer.fit(model=lit_model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    trainer.test(model=lit_model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    load_dotenv()  # Load .env file
    train()
