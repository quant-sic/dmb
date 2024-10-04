import itertools
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

from dmb.data.bose_hubbard_2d.plotting.phase_diagram import plot_phase_diagram
from dmb.data.bose_hubbard_2d.plotting.sandbox import create_box_cuts_plot, \
    create_box_plot, create_wedding_cake_plot, plot_phase_diagram_mu_cut
from dmb.model.dmb_model import dmb_model_predict


class PlottingCallback(Callback):

    def __init__(
            self,
            plot_interval: int = 100,
            resolution: int = 300,
            check: list[tuple[str, ...]] = [
                ("density", "max-min"),
                ("density_variance", "mean"),
                ("mu_cut", ),
                ("wedding_cake", "2.67"),
                ("wedding_cake", "1.33"),
                ("wedding_cake", "2.0"),
                ("box", "1.71"),
                ("box_cuts", ),
            ],
            zVUs: list[float] = (1.0, 1.5),
            ztUs: list[float] = (0.1, 0.25),
    ):
        self.plot_interval = plot_interval
        self.resolution = resolution
        self.check = check
        self.zVUs = zVUs
        self.ztUs = ztUs

    def on_train_epoch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule) -> None:
        if not trainer.current_epoch % self.plot_interval == 0:
            return

        save_dir = Path(trainer.log_dir) / "plots"
        file_name_stem = f"epoch={trainer.current_epoch}"

        mapping = partial(dmb_model_predict, model=pl_module.model)

        for zVU, ztU in itertools.product(self.zVUs, self.ztUs):
            for figures in (
                    create_box_cuts_plot(mapping, zVU=zVU, ztU=ztU),
                    create_box_plot(mapping, zVU=zVU, ztU=ztU),
                    create_wedding_cake_plot(mapping, zVU=zVU, ztU=ztU),
                    plot_phase_diagram(mapping,
                                       n_samples=self.resolution,
                                       zVU=zVU),
                    plot_phase_diagram_mu_cut(mapping, zVU=zVU, ztU=ztU),
                    plot_phase_diagram_mu_cut(mapping, zVU=zVU, ztU=ztU),
            ):

                def recursive_iter(path, obj):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            yield from recursive_iter(path + (key, ), value)
                    elif isinstance(obj, list):
                        for idx, value in enumerate(obj):
                            yield from recursive_iter(path + (idx, ), value)
                    else:
                        yield path, obj

                # recursively visit all figures
                for path, figure in recursive_iter((), figures):

                    # * is a wildcard
                    if not any(
                            all(a == b or a == "*" or b == "*"
                                for a, b in zip(check_, path))
                            and len(check_) == len(path)
                            for check_ in self.check):
                        continue

                    if isinstance(figure, plt.Figure):
                        save_path = Path(save_dir) / (
                            file_name_stem + "_" + str(zVU).replace(".", "_") +
                            "_" + str(ztU).replace(".", "_") + "_" +
                            "_".join(path) + ".png")
                        save_path.parent.mkdir(exist_ok=True, parents=True)
                        figure.savefig(save_path)
