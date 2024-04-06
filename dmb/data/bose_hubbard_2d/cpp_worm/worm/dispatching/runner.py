from typing import Literal, Any
from dmb.utils.io import create_logger
from pathlib import Path


from dmb.data.bose_hubbard_2d.worm_qmc.worm.process_runner.dispatchers import (
    determine_dispatcher,
    SlurmDispatcher,
    LocalDispatcher,
    SlurmDispatcherSettings,
    Dispatcher,
)
from pydantic_settings import BaseSettings

logger = create_logger(__name__)
