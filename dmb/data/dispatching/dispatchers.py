import asyncio
import datetime
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from attrs import define
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, \
    SettingsConfigDict

from dmb.data.dispatching.helpers import call_sbatch_and_wait, \
    check_if_slurm_is_installed_and_running
from dmb.logging import create_logger
from dmb.paths import REPO_ROOT

logger = create_logger(__name__)


class SlurmDispatcherSettings(BaseSettings):

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env",
                                      env_file_encoding="utf-8",
                                      extra="ignore")

    partition: str
    number_of_tasks_per_node: int
    number_of_nodes: int = 1
    cpus_per_task: int = 1

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, dotenv_settings


class Dispatcher(ABC):

    @abstractmethod
    async def dispatch(
        self,
        job_name: str,
        task: list[str],
        work_directory: Path,
        pipeout_dir: Path,
    ) -> None:
        ...


def determine_dispatcher(
    scheduler: Literal["auto", "slurm"] | None,
) -> Literal["auto", "slurm"] | None:

    if scheduler == "auto":
        if not check_if_slurm_is_installed_and_running():
            return None
        return "slurm"

    if scheduler == "slurm":
        if not check_if_slurm_is_installed_and_running():
            logger.warning(
                "Slurm is not installed or running. Falling back to None.")
            return None
        return "slurm"

    return scheduler


@define
class SlurmDispatcher(Dispatcher):

    setup_calls: tuple[str, ...] = (
        "module load gcc/11\n",
        "module load openmpi\n",
        "module load boost\n",
    )
    mpirun_calls: tuple[str, ...] = (
        """export MPIRUN_OPTIONS='--bind-to core --map-by socket:PE=${SLURM_CPUS_PER_TASK} -report-bindings'\n""",
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
    ):
        script_path.parent.mkdir(exist_ok=True, parents=True)
        pipeout_dir.mkdir(exist_ok=True, parents=True)

        with open(script_path, "w") as script_file:
            # write lines
            script_file.write("#!/bin/bash -l\n")
            script_file.write("#SBATCH --job-name={}\n".format(job_name))

            script_file.write("#SBATCH --output=" + str(pipeout_dir) +
                              "/stdout_{}_%j.txt\n".format(job_name))
            script_file.write("#SBATCH --error=" + str(pipeout_dir) +
                              "/stderr_{}_%j.txt\n".format(job_name))

            script_file.write("#SBATCH --partition={}\n".format(partition))

            script_file.write("#SBATCH --time=48:00:00\n".format(timeout))
            script_file.write("#SBATCH --nodes={}\n".format(number_of_nodes))
            script_file.write("#SBATCH --ntasks-per-node={}\n".format(
                number_of_tasks_per_node))
            script_file.write(
                "#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
            script_file.write("#SBATCH --mem=2G\n")

            for setup_call in self.setup_calls:
                script_file.write(setup_call)

            if task[0] == "mpirun":
                for mpirun_call in self.mpirun_calls:
                    script_file.write(mpirun_call)

            script_file.write(" ".join(task) + "\n")

        os.chmod(script_path, 0o755)

    async def dispatch(
        self,
        job_name: str,
        work_directory: Path,
        task: list[str],
        pipeout_dir: Path,
        timeout: int,
        dispatcher_settings: SlurmDispatcherSettings,
    ):

        script_path = work_directory / "run.sh"

        self.create_sbatch_script(
            script_path=script_path,
            job_name=job_name,
            pipeout_dir=pipeout_dir,
            task=task,
            partition=dispatcher_settings.partition,
            timeout=timeout,
            number_of_nodes=dispatcher_settings.number_of_nodes,
            number_of_tasks_per_node=dispatcher_settings.
            number_of_tasks_per_node,
            cpus_per_task=dispatcher_settings.cpus_per_task,
        )
        await call_sbatch_and_wait(script_path, timeout=timeout)


class LocalDispatcher(Dispatcher):

    async def dispatch(
        self,
        job_name: str,
        work_directory: Path,
        pipeout_dir: Path,
        task: list[str],
        timeout: int,
        dispatcher_settings: BaseSettings,
    ):
        env = os.environ.copy()
        env["TMPDIR"] = "/tmp"

        process = await asyncio.create_subprocess_exec(
            *task,
            env=env,
            stderr=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(),
                                                    timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()

        # write output to file
        now = datetime.datetime.now()
        with open(pipeout_dir / f"stdout_{job_name}_{now}.txt", "w") as f:
            f.write(stdout.decode("utf-8"))
        with open(pipeout_dir / f"stderr_{job_name}_{now}.txt", "w") as f:
            f.write(stderr.decode("utf-8"))

        # if job failed, raise error
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=task,
                output=stdout,
                stderr=stderr,
            )


class AutoDispatcher(Dispatcher):
    """Automatically selects the appropriate dispatcher based on the environment."""

    dispatcher_types: dict[str, Dispatcher] = {
        "slurm": SlurmDispatcher,
        "local": LocalDispatcher,
    }
    dispatcher_settings_types: dict[str, BaseSettings] = {
        "slurm": SlurmDispatcherSettings,
        "local": BaseSettings,
    }

    def __init__(
            self,
            dispatcher_type: Literal["auto", "slurm"] | None = "auto") -> None:
        determined_dispatcher_type = determine_dispatcher(
            dispatcher_type) or "local"

        self.dispatcher = self.dispatcher_types[determined_dispatcher_type]()
        self.dispatcher_settings_type = self.dispatcher_settings_types[
            determined_dispatcher_type]

    async def dispatch(
        self,
        job_name: str,
        task: list[str],
        work_directory: Path,
        pipeout_dir: Path,
        timeout: int,
        dispatcher_kwargs: dict[str, Any] = {},
    ) -> None:

        # update dispatcher settings

        updated_dispatcher_settings = self.dispatcher_settings_type(
            **dispatcher_kwargs)

        await self.dispatcher.dispatch(
            job_name=job_name,
            task=task,
            work_directory=work_directory,
            pipeout_dir=pipeout_dir,
            timeout=timeout,
            dispatcher_settings=updated_dispatcher_settings,
        )
