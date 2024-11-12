"""Logging functionality."""

import logging
from pathlib import Path


def create_logger(app_name: str,
                  file: Path | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Serves as a unified way to instantiate a new logger. Will create a new logging
    instance with the name app_name. The logging output is sent to the console via
    a logging.StreamHandler() instance. The output will be formatted using the
    logging time, the logger name, the level at which the logger was called and the
    logging message. As the root logger threshold is set to WARNING, the instantiation
    via logging.getLogger(__name__) results in a logger instance, which console handel
    also has the threshold set to WARNING. One needs to additionally set the console
    handler level to the desired level, which is done by this function.

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
    log_formatter = logging.Formatter(
        "%(asctime)s [%(filename)s] [%(funcName)s] [%(levelname)s] "
        "[%(lineno)d] %(message)s")

    # create new up logger
    logger = logging.getLogger(app_name)
    logger.setLevel(level)

    # create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(log_formatter)

    # add file handler
    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(level)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    # if number of stream handlers is 0, add console handler
    if (len(
            list(filter(lambda x: isinstance(x, logging.StreamHandler),
                        logger.handlers))) == 0):
        logger.addHandler(console_handler)

    return logger
