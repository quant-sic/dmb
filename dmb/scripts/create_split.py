from pathlib import Path

import hydra
from omegaconf import DictConfig

from dmb.paths import REPO_ROOT


@hydra.main(
    version_base="1.2",
    config_path=str(REPO_ROOT / "dmb/scripts/configs"),
    config_name="create_split.yaml",
)
def main(cfg: DictConfig):

    split = hydra.utils.instantiate(cfg.split)
    split.to_file(Path(cfg.file_path))


if __name__ == "__main__":
    main()
