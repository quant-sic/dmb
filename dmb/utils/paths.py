import os
from pathlib import Path

import git


def get_repo_root() -> Path:
    """Returns the root path of current git repo."""
    repo = git.Repo(__file__, search_parent_directories=True)

    working_dir = repo.working_tree_dir
    if working_dir is None:
        raise ValueError("Could not find git repository working dir")
    else:
        return Path(working_dir)


def get_repo_data_path() -> Path:
    """Returns root path of data directory."""
    repo_root = get_repo_root()
    data_path = repo_root / "data"

    return data_path


# global variable for the root path of the current git repository
REPO_ROOT = get_repo_root()

# global variable for the root path of the data directory in current git repository
REPO_DATA_ROOT = get_repo_data_path()


# set environment variables
os.environ["REPO_ROOT"] = str(REPO_ROOT)
os.environ["REPO_DATA_ROOT"] = str(REPO_DATA_ROOT)