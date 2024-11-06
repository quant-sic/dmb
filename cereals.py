"""Handle multiple platforms for the dependencies with rye.

Use the `cereals` key in pyproject toml to collect configurations, like this

[cereals.cpu]
dependencies = [ "torch==2.3.1+cpu",]
extra_index_urls = ["https://download.pytorch.org/whl/cpu"]

[cereals.cu121]
dependencies = ["torch==2.3.1+cu121"]
extra_index_urls = ["https://download.pytorch.org/whl/cu121"]

the different sets of additional dependencies can be declared like this

[project.optional-dependencies]
dev= [
    "jupyterlab>=4.0",
]
ci = [
    "pytest-cov>=5.0.0",
]

it's assumed that there is at least the dev section present.

This script can be used to generate lock files for all platforms
and then sync the venv to the chosen platform.
For doing the lock-file generation it needs the tomlkit package.
This is imported inside the functions, so that some functionality
can be used without tomlkit installed. (important for the CI pipeline)
"""

import os
import shutil
import subprocess
from itertools import chain, repeat
from pathlib import Path
import typer
import tomlkit
from attrs import field, frozen, define
from tomlkit import TOMLDocument


@frozen
class Cereal:
    """A cereal section in the pyproject.toml file.
    
    Attributes:
        dependencies: The dependencies for the cereal.
        extra_index_urls: The extra index urls for the cereal.
    """

    dependencies: list[str]
    extra_index_urls: list[str] = []


@define
class CerealTOMLDocument:
    """A class to handle the pyproject.toml file for cereals.
    
    Attributes:
        toml_file_path: The path to the toml file.
    """

    toml_file_path: Path

    @property
    def toml_document(self) -> TOMLDocument:
        """Load the toml file and return the document."""
        with open(self.toml_file_path) as f:
            return tomlkit.load(f)

    def write(self, data: TOMLDocument) -> None:
        """Write the data to the toml file."""
        with open(self.toml_file_path, "w") as f:
            tomlkit.dump(data, f)

    @property
    def cereals(self) -> dict[str, Cereal]:
        """Return the cereals from the toml file."""
        cereals = self.toml_document.get("cereals", {})
        return {key: Cereal(**value) for key, value in cereals.items()}

    @property
    def base_dependencies(self) -> list[str]:
        """Return the base dependencies from the toml file."""
        return self.toml_document["project"]["dependencies"]

    @base_dependencies.setter
    def base_dependencies(self, value: list[str]) -> None:
        """Set the base dependencies in the toml file."""
        new_toml_document = self.toml_document
        new_toml_document["project"]["dependencies"] = value
        new_toml_document["project"]["dependencies"].multiline(True)
        self.write(new_toml_document)

    @property
    def optional_dependencies(self) -> dict[str, list[str]]:
        """Return the optional dependencies from the toml file."""
        return self.toml_document["project"]["optional-dependencies"]

    def set_cereal(self, key: str) -> None:
        """Set the cereal in the toml file. Unset all other cereals."""
        for unset_key in self.cereals:
            self.unset_cereal(unset_key)

        cereal = self.cereals[key]
        self.base_dependencies = self.base_dependencies + cereal.dependencies

    def unset_cereal(self, key: str) -> None:
        """Unset a cereal in the toml file."""
        cereal = self.cereals[key]
        self.base_dependencies = [
            dep for dep in self.base_dependencies
            if dep not in cereal.dependencies
        ]


def build_uv_lock_command(
    extra_index_urls: tuple[str, ...],
    output_path: Path,
    additional_args: tuple[str, ...] = ()) -> list[str]:
    """Build the uv lock command.
    
    Args:
        extra_index_urls: The extra index urls to use.
        output_path: The path to the output lock file.
        additional_args: Additional arguments to pass to uv.
    """
    urls = chain(*zip(repeat("--extra-index-url"), extra_index_urls))
    return [
        os.getenv("UV_PATH", shutil.which("uv")),
        "pip",
        "compile",
        "pyproject.toml",
        "--emit-index-url",
        "--index-strategy",
        "unsafe-first-match",
        *urls,
        *additional_args,
        "-o",
        output_path,
    ]


app = typer.Typer()


@app.command(name="lock")
def lock(
        cereal_name: str = "linux-cpu",
        uv_arguments: list[str] = [],
        pyproject_toml_path: Path = Path("./pyproject.toml"),
        requirements_directory: Path = Path("./requirements"),
) -> None:
    """Generate lock files for the given cereal.
    
    Args:
        cereal_name: The name of the cereal to generate lock files for.
            Defaults to "linux-cpu".
        uv_arguments: Additional arguments to pass to uv.
            Defaults to [].
        pyproject_toml_path: The path to the pyproject.toml file.
            Defaults to Path("./pyproject.toml").
        requirements_directory: The directory to save the lock files.
            Defaults to Path("./requirements").
    """
    document = CerealTOMLDocument(pyproject_toml_path)

    if cereal_name not in document.cereals:
        raise ValueError(
            f"Cereal {cereal_name} not available. Choose from {document.cereals.keys()}"
        )

    document.set_cereal(cereal_name)

    lock_files_directory = requirements_directory / cereal_name
    lock_files_directory.mkdir(exist_ok=True, parents=True)

    # generate base lock file
    subprocess.run(
        build_uv_lock_command(
            extra_index_urls=document.cereals[cereal_name].extra_index_urls,
            output_path=lock_files_directory / "requirements-base.lock",
            additional_args=uv_arguments,
        ))

    # generate lock files for optional dependencies
    for extra in document.optional_dependencies:
        subprocess.run(
            build_uv_lock_command(
                extra_index_urls=document.cereals[cereal_name].
                extra_index_urls,
                output_path=lock_files_directory /
                f"requirements-{extra}.lock",
                additional_args=["--extra", extra] + uv_arguments,
            ))

    # generate lock file for all dependencies
    subprocess.run(
        build_uv_lock_command(
            extra_index_urls=document.cereals[cereal_name].extra_index_urls,
            output_path=lock_files_directory / "requirements-all.lock",
            additional_args=uv_arguments +
            list(chain(
                *zip(repeat("--extra"), document.optional_dependencies)))))

    # unset the cereal
    document.unset_cereal(cereal_name)


@app.command(name="set")
def set_cereal(
    cereal_name: str = "linux-cpu",
    pyproject_toml_path: Path = Path("./pyproject.toml")
) -> None:
    """Set a cereal in the pyproject.toml file.
    
    Args:
        cereal_name: The name of the cereal to set. Defaults to "linux-cpu".
        pyproject_toml_path: The path to the pyproject.toml file.
            Defaults to Path("./pyproject.toml").
    """
    document = CerealTOMLDocument(pyproject_toml_path)
    document.set_cereal(cereal_name)


if __name__ == "__main__":
    app()
