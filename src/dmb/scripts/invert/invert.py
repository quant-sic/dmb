"""Suggested train scripts format."""

import hydra
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from dmb.model.inversion import InversionFakeDataLoader

from dmb.paths import REPO_ROOT


@hydra.main(
    version_base="1.3",
    config_path=str(REPO_ROOT / "src/dmb/scripts/invert/configs"),
    config_name="invert.yaml",
)
def invert(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.seed, workers=True)

    callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
    logger = list(hydra.utils.instantiate(cfg.logger).values())

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer,
                                               logger=logger,
                                               callbacks=callbacks)



    lit_model: LightningModule = hydra.utils.instantiate(cfg.lit_model)

    trainer.fit(lit_model, train_dataloaders=InversionFakeDataLoader())

    trainer.save_checkpoint("model.ckpt")

if __name__ == "__main__":
    invert()
