from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings

from dmb.data.bose_hubbard_2d.worm_qmc.worm.process_runner.dispatchers import \
    Dispatcher, LocalDispatcher, SlurmDispatcher, SlurmDispatcherSettings, \
    determine_dispatcher
from dmb.utils.io import create_logger

logger = create_logger(__name__)
