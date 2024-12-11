import itertools
from pathlib import Path
from typing import Generator

import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

from dmb.data.bose_hubbard_2d.plotting.phase_diagram import plot_phase_diagram
from dmb.data.bose_hubbard_2d.plotting.sandbox import create_box_cuts_plot, \
    create_box_plot, create_wedding_cake_plot, plot_phase_diagram_mu_cut
from dmb.model.dmb_model import PredictionMapping

class PlottingCallback(Callback):

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        if not trainer.log_dir:
            raise ValueError("log_dir is not set in Trainer")

        fig, ax = plt.subplots(1,3)
        ax[0].imshow(pl_module.inversion_result.inversion_result.cpu().detach().numpy())
        ax[1].imshow(pl_module.dmb_model.forward(pl_module.inversion_result().unsqueeze(0)).cpu().detach().numpy()[0,0])
        ax[2].imshow(pl_module.output.cpu().detach().numpy())

        ax[0].set_title("Inverted")
        ax[1].set_title("Model(Inverted)")
        ax[2].set_title("Target")

        fig_path = Path(trainer.log_dir) / "plots" / "final.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)