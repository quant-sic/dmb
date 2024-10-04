import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_DATA_ROOT = REPO_ROOT / "data"
REPO_LOGS_ROOT = REPO_ROOT / "logs"

# add to environment for hydra
os.environ["REPO_ROOT"] = str(REPO_ROOT)
os.environ["REPO_DATA_ROOT"] = str(REPO_DATA_ROOT)
os.environ["REPO_LOGS_ROOT"] = str(REPO_LOGS_ROOT)
