from functools import partial

from joblib import delayed
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.datamodule import BoseHubbardDataModule
from dmb.utils import REPO_DATA_ROOT
from dmb.utils.io import ProgressParallel

if __name__ == "__main__":
    dm = BoseHubbardDataModule(
        data_dir=REPO_DATA_ROOT / "bose_hubbard_2d",
        batch_size=64,
        clean=True,
        reload=True,
        verbose=True,
        observables=["Density_Distribution"],
    )
    dm.setup("fit")

    # parallelized __getitem__ with tqdm progress bar and joblib, reload=True

    with ProgressParallel(n_jobs=10, total=len(dm.dataset)) as parallel:
        data = parallel(
            delayed(partial(dm.dataset.__getitem__, reload=True))(i)
            for i in tqdm(range(len(dm.dataset)))
        )
