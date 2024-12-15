from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class PlottingCallback(Callback):

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        if not trainer.log_dir:
            raise ValueError("log_dir is not set in Trainer")

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(pl_module.inversion_result.data.cpu().detach().numpy())
        ax[1].imshow(
            pl_module.dmb_model.forward(
                pl_module.inversion_result().unsqueeze(0)).cpu().detach().numpy()[0, 0])
        ax[2].imshow(pl_module.output.cpu().detach().numpy())

        ax[0].set_title("Inverted")
        ax[1].set_title("Model(Inverted)")
        ax[2].set_title("Target")

        fig_path = Path(trainer.log_dir) / "plots" / "final.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
