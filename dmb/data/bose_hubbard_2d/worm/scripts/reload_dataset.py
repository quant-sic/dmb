from functools import partial

from joblib import delayed
from tqdm import tqdm

from dmb.data.bose_hubbard_2d.transforms import BoseHubbard2DTransforms
from dmb.data.bose_hubbard_2d.worm.dataset import BoseHubbardDataset
from dmb.io import ProgressParallel
from dmb.logging import create_logger
from dmb.paths import REPO_DATA_ROOT

log = create_logger(__name__)

if __name__ == "__main__":
    ds = BoseHubbardDataset(
        data_dir=REPO_DATA_ROOT / "bose_hubbard_2d/random",
        transforms=BoseHubbard2DTransforms(),
        clean=True,
        reload=True,
        verbose=True,
        max_density_error=0.015,
        # recalculate_errors=True,
    )

    # parallelized load_sample with tqdm progress bar and joblib, reload=True
    for i in tqdm(range(len(ds))):
        ds.load_sample(i, reload=True)

    log.info("len(data): %s", len(ds))
