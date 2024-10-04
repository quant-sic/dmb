from pathlib import Path
from typing import Any, Optional

from joblib import Parallel
from tqdm import tqdm


class ProgressParallel(Parallel):
    """A joblib.Parallel class that provides accurate tqdm Progress bars."""

    def __init__(
        self,
        use_tqdm: bool = True,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            use_tqdm (bool, optional): Whether to use the tqdm Progress bar. Defaults to True.
            total ([type], optional): Total nmber of jobs. Defaults to None.
        """
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with tqdm(disable=not self._use_tqdm,
                  total=self._total,
                  desc=self._desc) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self) -> None:
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
