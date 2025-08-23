from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


class StoreResultsCallback(Callback):
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.log_dir:
            raise ValueError("log_dir is not set in Trainer")

        muU = pl_module.inversion_result.muU.cpu().detach()
        mu = pl_module.inversion_result.mu.cpu().detach()
        target = pl_module.output.cpu().detach()
        remapped = (
            pl_module.dmb_model.forward(pl_module.inversion_result().unsqueeze(0))
            .cpu()
            .detach()[0, 0]
        )

        # save as .pt
        results_dir = Path(trainer.log_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        torch.save(muU, results_dir / "muU.pt")
        torch.save(mu, results_dir / "mu.pt")
        torch.save(target, results_dir / "target.pt")
        torch.save(remapped, results_dir / "remapped.pt")

        fig, ax = plt.subplots(1, 3)

        ax[0].imshow(muU)
        ax[1].imshow(remapped)
        ax[2].imshow(target)

        ax[0].set_title("Inverted muU")
        ax[1].set_title("Model(Inverted muU)")
        ax[2].set_title("Target")

        fig_path = Path(trainer.log_dir) / "plots" / "final.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
