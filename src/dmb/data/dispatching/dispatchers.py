"""Dispatchers for running jobs in different setups."""

import abc
import asyncio
import datetime
import os
import re
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

from attrs import define
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, \
    SettingsConfigDict

from dmb.data.dispatching.helpers import ReturnCode, call_sbatch_and_wait, \
    check_if_slurm_is_installed_and_running
from dmb.logging import create_logger
from dmb.paths import REPO_ROOT

logger = create_logger(__name__)


class SlurmDispatcherSettings(BaseSettings):
    """Settings for the Slurm dispatcher."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="slurm_dispatcher_",
    )

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
        ...


def determine_dispatcher(
    scheduler: Literal["auto", "slurm"] | None, ) -> Literal["auto", "slurm"] | None:

    if scheduler == "auto":
        if not check_if_slurm_is_installed_and_running():
            return None
        return "slurm"

    if scheduler == "slurm":
        if not check_if_slurm_is_installed_and_running():
            logger.warning("Slurm is not installed or running. Falling back to None.")
            return None
        return "slurm"

    return scheduler


@define
class SlurmDispatcher(Dispatcher):

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
        stdout_path = pipeout_dir / "stdout_{}.txt".format(out_id)
        stderr_path = pipeout_dir / "stderr_{}.txt".format(out_id)

        with open(script_path, "w") as script_file:
            # write lines
            script_file.write("#!/bin/bash -l\n")
            script_file.write(f"#SBATCH --job-name={job_name}\n")

            script_file.write("#SBATCH --output=" + str(stdout_path) + "\n")
            script_file.write("#SBATCH --error=" + str(stderr_path) + "\n")

            script_file.write("#SBATCH --partition={}\n".format(partition))

            script_file.write("#SBATCH --time=48:00:00\n".format())
            script_file.write("#SBATCH --nodes={}\n".format(number_of_nodes))
            script_file.write(
                "#SBATCH --ntasks-per-node={}\n".format(number_of_tasks_per_node))
            script_file.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
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
        with open(create_script_out["stderr_path"], "r") as f:
            stderr = f.read()
            errorcode = re.findall(r"errorcode\s(\d+)", stderr)
            if errorcode:
                if int(errorcode[0]) != 0:
                    code = ReturnCode.FAILURE

        return code


class LocalDispatcher(Dispatcher):

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
            stdout, stderr = await asyncio.wait_for(process.communicate(),
                                                    timeout=timeout)
            await process.wait()
            return_code = (ReturnCode.SUCCESS
                           if process.returncode == 0 else ReturnCode.FAILURE)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return_code = ReturnCode.FAILURE

        # write output to file
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pipeout_dir.mkdir(exist_ok=True, parents=True)
        with open(pipeout_dir / f"stdout_{job_name}_{now}.txt", "w",
                  encoding="utf-8") as f:
            f.write(stdout.decode("utf-8"))
        with open(pipeout_dir / f"stderr_{job_name}_{now}.txt", "w",
                  encoding="utf-8") as f:
            f.write(stderr.decode("utf-8"))

        return return_code


class AutoDispatcher(Dispatcher):
    """Automatically selects the appropriate dispatcher based on the environment."""

    dispatcher_types: dict[str, type[Dispatcher]] = {
        "slurm": SlurmDispatcher,
        "local": LocalDispatcher,
    }
    dispatcher_settings_types: dict[str, type[BaseSettings]] = {
        "slurm": SlurmDispatcherSettings,
        "local": BaseSettings,
    }

    def __init__(
        self,
        dispatcher_type: Literal["auto", "slurm"] | None = "auto",
        dispatcher_settings: dict[str, Any] = {},
    ) -> None:
        determined_dispatcher_type = determine_dispatcher(dispatcher_type) or "local"

        self.dispatcher_settings = self.dispatcher_settings_types[
            determined_dispatcher_type](**dispatcher_settings)

        self.dispatcher = self.dispatcher_types[determined_dispatcher_type](
            dispatcher_settings=self.dispatcher_settings)

    async def dispatch(
        self,
        job_name: str,
        work_directory: Path,
        pipeout_dir: Path,
        task: list[str],
        timeout: int,
    ) -> ReturnCode:

        code = await self.dispatcher.dispatch(job_name=job_name,
                                              task=task,
                                              work_directory=work_directory,
                                              pipeout_dir=pipeout_dir,
                                              timeout=timeout)

        return code
