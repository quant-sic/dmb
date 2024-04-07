from functools import partial

from joblib import delayed
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.cpp_worm.datamodule import BoseHubbardDataModule
from dmb.utils import REPO_DATA_ROOT, create_logger
from dmb.utils.io import ProgressParallel

log = create_logger(__name__)

if __name__ == "__main__":
    dm = BoseHubbardDataModule(
        data_dir=REPO_DATA_ROOT / "bose_hubbard_2d",
        batch_size=64,
        clean=True,
        reload=True,
        verbose=True,
        observables=[
            "density",
            "density_variance",
            "density_density_corr_0",
            "density_density_corr_1",
            "density_density_corr_2",
            "density_density_corr_3",
            "density_squared",
        ],
        max_density_error=0.015,
        include_tune_dirs=True,
    )
    dm.setup("fit")

    # parallelized __getitem__ with tqdm progress bar and joblib, reload=True
    with ProgressParallel(n_jobs=10, total=len(dm.dataset)) as parallel:
        data = parallel(
            delayed(partial(dm.dataset.__getitem__, reload=True))(i)
            for i in tqdm(range(len(dm.dataset))))

    log.info("len(data): %s", len(data))
