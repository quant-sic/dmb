from lightning.pytorch.callbacks import Callback
from pathlib import Path
from typing import List, Tuple


class PhaseDiagramPlotCallback(Callback):
    def __init__(
        self,
        plot_interval: int,
        check: List[Tuple[str, ...]] = [
            ("density", "max-min"),
            ("density_variance", "mean"),
            ("mu_cut",),
        ],
    ) -> None:
        super().__init__()
        self.check = check
        self.plot_interval = plot_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.plot_interval == 0:
            pl_module.plot_model(
                check=self.check,
                save_dir=Path(trainer.logger.log_dir) / "figures",
                file_name_stem=f"epoch={trainer.current_epoch}",
            )
