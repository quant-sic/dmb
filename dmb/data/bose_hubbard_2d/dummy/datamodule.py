from typing import Any, Callable, Dict, List, Optional, Tuple

from dmb.data.bose_hubbard_2d.simulated.dataset import SimulatedBoseHubbard2dDataset
from dmb.data.mixins import DataModuleMixin
from dmb.data.utils import chain_fns, collate_sizes
from dmb.utils import create_logger

log = create_logger(__name__)


class SiumlatedBoseHubbardDataModule(DataModuleMixin):
    def __init__(
        self,
        num_samples: 5000,
        train_val_test_split: List[float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        base_transforms=None,
        train_transforms=None,
        resplit: Optional[List[Dict]] = None,
        split_usage: Dict[str, int] = {"train": 0, "val": 1, "test": 2},
        pin_memory: bool = False,
        observables: List[str] = [
            "Density_Distribution",
        ],
        save_split_indices: bool = True,
        muU_range: Tuple[float, float] = (-0.5, 3.0),
        ztU_range: Tuple[float, float] = (0.05, 1.0),
        zVU_range: Tuple[float, float] = (0.75, 1.75),
        L_range: Tuple[int, int] = (8, 20),
        z=4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def get_dataset(self):
        return SimulatedBoseHubbard2dDataset(
            num_samples=self.hparams["num_samples"],
            muU_range=self.hparams["muU_range"],
            ztU_range=self.hparams["ztU_range"],
            zVU_range=self.hparams["zVU_range"],
            L_range=self.hparams["L_range"],
            z=self.hparams["z"],
            base_transforms=self.base_transforms,
            train_transforms=self.train_transforms,
        )

    def get_collate_fn(self) -> Optional[Callable]:
        collate_fns: List[Callable[[Any], Any]] = [collate_sizes]

        return chain_fns(collate_fns)
