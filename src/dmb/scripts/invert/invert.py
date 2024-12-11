"""Suggested train scripts format."""

from pathlib import Path

import hydra
import lightning.pytorch as pl
import numpy as np
from lightning.pytorch import LightningModule, Trainer
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

    trainer.save_checkpoint(Path(trainer.default_root_dir) / "model.ckpt")

    # save as npy
    np.save(Path(trainer.default_root_dir) / "inverted.npy", lit_model.inversion_result)


if __name__ == "__main__":
    invert()
