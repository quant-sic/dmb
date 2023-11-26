import logging
from typing import Any, Optional

from joblib import Parallel
from tqdm import tqdm


def create_logger(app_name: str, level: int = logging.INFO) -> logging.Logger:
    """Serves as a unified way to instantiate a new logger. Will create a new logging instance with the name app_name. The logging output is sent to the console via a logging.StreamHandler() instance. The output will be formatted using the logging time, the logger name, the level at which the logger was called and the logging message. As the root logger threshold is set to WARNING, the instantiation via logging.getLogger(__name__) results in a logger instance, which console handel also has the threshold set to WARNING. One needs to additionally set the console handler level to the desired level, which is done by this function.

    ..note:: Function might be adapted for more specialized usage in the future

    Args:
        app_name (string): Name of the logger. Will appear in the console output
        level (int): threshold level for the new logger.

    Returns:
        logging.Logger: new logging instance

    Examples::

    >>> import logging
    >>> logger=create_logger(__name__,logging.DEBUG)
    """
    logFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s] [%(funcName)s] [%(levelname)s] [%(lineno)d] %(message)s"
    )

    # create new up logger
    logger = logging.getLogger(app_name)
    logger.setLevel(level)

    # create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logFormatter)

    if not len(logger.handlers):
        logger.addHandler(ch)

    return logger


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
        with tqdm(
            disable=not self._use_tqdm, total=self._total, desc=self._desc
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self) -> None:
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
