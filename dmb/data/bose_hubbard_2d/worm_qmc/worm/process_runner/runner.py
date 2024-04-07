from typing import Literal, Any
from dmb.utils.io import create_logger
from pathlib import Path


from dmb.data.bose_hubbard_2d.worm_qmc.worm.process_runner.dispatchers import (
    determine_dispatcher,
    SlurmDispatcher,
    LocalDispatcher,
    SlurmDispatcherSettings,
)
from pydantic_settings import BaseSettings

logger = create_logger(__name__)


class ProcessRunner:

    def __init__(
        self, dispatcher_type: Literal["auto", "slurm"] | None = "auto"
    ) -> None:
        determined_dispatcher_type = determine_dispatcher(dispatcher_type)

        if determined_dispatcher_type == "slurm":
            self.dispatcher = SlurmDispatcher()
            self.dispatcher_settings = SlurmDispatcherSettings()

        if determined_dispatcher_type is None:
            self.dispatcher = LocalDispatcher()
            self.dispatcher_settings = BaseSettings()

    async def run(
        self,
        task: list[str],
        job_name: str,
        work_directory: Path,
        pipeout_dir: Path,
        timeout: int,
        dispatcher_kwargs: dict[str, Any] = {},
    ) -> None:

        # update dispatcher settings
        self.dispatcher_settings.update(dispatcher_kwargs)

        await self.dispatcher.dispatch(
            job_name=job_name,
            task=task,
            work_directory=work_directory,
            pipeout_dir=pipeout_dir,
            timeout=timeout,
            dispatcher_settings=self.dispatcher_settings,
        )
