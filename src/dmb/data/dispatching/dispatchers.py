"""Dispatchers for running jobs in different setups."""

import abc
import asyncio
import datetime
import os
import re
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any

from attrs import define
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from dmb.data.dispatching.helpers import (
    ReturnCode,
    call_sbatch_and_wait,
    check_if_slurm_is_installed_and_running,
)
from dmb.logging import create_logger
from dmb.paths import REPO_ROOT

logger = create_logger(__name__)


class SlurmDispatcherSettings(BaseSettings):
    """Settings for the Slurm dispatcher."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="slurm_dispatcher_",
    )

    partition: str = Field(
        default="standard",
        title="Partition",
        description="The partition to use on the cluster.",
    )
    number_of_tasks_per_node: int = Field(
        default=1,
        title="Number of tasks per node",
        description="The number of tasks per node.",
        ge=1,
    )
    number_of_nodes: int = Field(
        default=1,
        title="Number of nodes",
        description="The number of nodes to use.",
        ge=1,
    )
    cpus_per_task: int = Field(
        default=1,
        title="CPUs per task",
        description="The number of CPUs per task.",
        ge=1,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings


@define
class Dispatcher(metaclass=abc.ABCMeta):
    """Abstract base class for dispatchers."""

    dispatcher_settings: BaseSettings

    @abstractmethod
    async def dispatch(
        self,
        job_name: str,
        work_directory: Path,
        pipeout_dir: Path,
        task: list[str],
        timeout: int,
    ) -> ReturnCode:
        """Dispatch a job."""


def auto_create_dispatcher() -> Dispatcher:
    """Automatically determine the dispatcher based on the environment.

    Returns:
        Dispatcher: The determined dispatcher.
    """

    if check_if_slurm_is_installed_and_running():
        return SlurmDispatcher(dispatcher_settings=SlurmDispatcherSettings())

    return LocalDispatcher(dispatcher_settings=BaseSettings())


@define
class SlurmDispatcher(Dispatcher):
    """A dispatcher that runs jobs on a Slurm cluster."""

    dispatcher_settings: SlurmDispatcherSettings
    setup_calls: tuple[str, ...] = (
        "module load gcc/11\n",
        "module load openmpi\n",
        "module load boost\n",
    )
    mpirun_calls: tuple[str, ...] = (
        "export MPIRUN_OPTIONS='--bind-to core --map-by "
        "socket:PE=${SLURM_CPUS_PER_TASK} -report-bindings'\n",
        "export TMPDIR=/tmp\n",
    )

    def create_sbatch_script(
        self,
        script_path: Path,
        task: list[str],
        job_name: str,
        pipeout_dir: Path,
        partition: str,
        timeout: int,
        number_of_nodes: int,
        number_of_tasks_per_node: int,
        cpus_per_task: int,
    ) -> dict[str, Any]:
        """Create an sbatch script for the given task."""

        script_path.parent.mkdir(exist_ok=True, parents=True)
        pipeout_dir.mkdir(exist_ok=True, parents=True, mode=0o777)

        out_id = uuid.uuid4()
        stdout_path = pipeout_dir / f"stdout_{out_id}.txt"
        stderr_path = pipeout_dir / f"stderr_{out_id}.txt"

        with open(script_path, "w", encoding="utf-8") as script_file:
            # write lines
            script_file.write("#!/bin/bash -l\n")
            script_file.write(f"#SBATCH --job-name={job_name}\n")

            script_file.write("#SBATCH --output=" + str(stdout_path) + "\n")
            script_file.write("#SBATCH --error=" + str(stderr_path) + "\n")

            script_file.write(f"#SBATCH --partition={partition}\n")

            hours, remainder = divmod(timeout, 3600)
            minutes, seconds = divmod(remainder, 60)

            script_file.write(f"#SBATCH --time={hours:02}:{minutes:02}:{seconds:02}\n")

            script_file.write(f"#SBATCH --nodes={number_of_nodes}\n")
            script_file.write(f"#SBATCH --ntasks-per-node={number_of_tasks_per_node}\n")
            script_file.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
            script_file.write("#SBATCH --mem=2G\n")

            for setup_call in self.setup_calls:
                script_file.write(setup_call)

            if task[0] == "mpirun":
                for mpirun_call in self.mpirun_calls:
                    script_file.write(mpirun_call)

            script_file.write(" ".join(task) + "\n")

        os.chmod(script_path, 0o755)

        return {"stdout_path": stdout_path, "stderr_path": stderr_path}

    async def dispatch(
        self,
        job_name: str,
        work_directory: Path,
        pipeout_dir: Path,
        task: list[str],
        timeout: int,
    ) -> ReturnCode:
        script_path = work_directory / "run.sh"

        create_script_out = self.create_sbatch_script(
            script_path=script_path,
            job_name=job_name,
            pipeout_dir=pipeout_dir,
            task=task,
            partition=self.dispatcher_settings.partition,
            timeout=timeout,
            number_of_nodes=self.dispatcher_settings.number_of_nodes,
            number_of_tasks_per_node=self.dispatcher_settings.number_of_tasks_per_node,
            cpus_per_task=self.dispatcher_settings.cpus_per_task,
        )
        code = await call_sbatch_and_wait(script_path, timeout=timeout)

        # check if the job was successful
        with open(create_script_out["stderr_path"], "r", encoding="utf-8") as f:
            stderr = f.read()
            errorcode = re.findall(r"errorcode\s(\d+)", stderr)
            if errorcode:
                if int(errorcode[0]) != 0:
                    code = ReturnCode.FAILURE

        return code


class LocalDispatcher(Dispatcher):
    """A dispatcher that runs jobs locally."""

    async def dispatch(
        self,
        job_name: str,
        work_directory: Path,
        pipeout_dir: Path,
        task: list[str],
        timeout: int,
    ) -> ReturnCode:
        env = os.environ.copy()
        env["TMPDIR"] = "/tmp"

        process = await asyncio.create_subprocess_exec(
            *task,
            env=env,
            stderr=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            await process.wait()
            return_code = (
                ReturnCode.SUCCESS if process.returncode == 0 else ReturnCode.FAILURE
            )

        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return_code = ReturnCode.FAILURE

        # write output to file
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pipeout_dir.mkdir(exist_ok=True, parents=True)
        with open(
            pipeout_dir / f"stdout_{job_name}_{now}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(stdout.decode("utf-8"))
        with open(
            pipeout_dir / f"stderr_{job_name}_{now}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(stderr.decode("utf-8"))

        return return_code
