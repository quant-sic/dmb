from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dmb.data.bose_hubbard_2d.worm_qmc.dataset import BoseHubbardDataset
from dmb.data.mixins import DataModuleMixin
from dmb.data.utils import chain_fns, collate_sizes
from dmb.utils import create_logger

log = create_logger(__name__)


class BoseHubbardDataModule(DataModuleMixin):
    def __init__(
        self,
        data_dir: Path,
        train_val_test_split: List[float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        clean=True,
        base_transforms=None,
        train_transforms=None,
        resplit: Optional[List[Dict]] = None,
        split_usage: Dict[str, int] = {"train": 0, "val": 1, "test": 2},
        pin_memory: bool = False,
        max_density_error: float = 0.015,
        observables: List[str] = [
            "density",
            "density_variance",
            "density_density_corr_0",
            "density_density_corr_1",
            "density_density_corr_2",
            "density_density_corr_3",
            "density_squared",
        ],
        save_split_indices: bool = True,
        reload: bool = False,
        verbose: bool = False,
        include_tune_dirs: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def get_dataset(self):
        return BoseHubbardDataset(
            self.hparams["data_dir"],
            clean=self.hparams["clean"],
            observables=self.hparams["observables"],
            reload=self.hparams["reload"],
            verbose=self.hparams["verbose"],
            max_density_error=self.hparams["max_density_error"],
            include_tune_dirs=self.hparams["include_tune_dirs"],
        )

    def get_collate_fn(self) -> Optional[Callable]:
        collate_fns: List[Callable[[Any], Any]] = [collate_sizes]

        return chain_fns(collate_fns)
