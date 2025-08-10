"""Load simulation data from a directory containing simulation directories and
save it to a dataset directory."""

import json
from pathlib import Path

import numpy as np
import torch
from attrs import frozen
from tqdm import tqdm
from typer import Typer

from dmb.data.bose_hubbard_2d.nn_input import get_nn_input
from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulation
from dmb.logging import create_logger

log = create_logger(__name__)

app = Typer()


def get_dataset_sample_name(simulation_dir: Path) -> str:
    """Get the name of a dataset sample from a simulation directory."""

    return (
        simulation_dir.name
        if not simulation_dir.name == "tune"
        else simulation_dir.parent.name + "_tune"
    )


@frozen
class FilterStrategy:
    """A strategy for filtering simulation directories."""

    clean: bool
    max_density_error: float | None
    recalculate_errors: bool
    reevaluate: bool
    dataset_samples_dir_path: Path

    @staticmethod
    def _get_simulation_valid(sim_dir: Path, reevaluate: bool) -> bool:
        """Check if a simulation directory is valid."""

        try:
            sim = WormSimulation.from_dir(sim_dir)
        except (OSError, KeyError, ValueError) as e:
            log.error(f"Could not load simulation from {sim_dir} with error {e}")
            return False

        if "clean" in sim.record and not sim.record["clean"] and not reevaluate:
            return False

        if "clean" in sim.record and sim.record["clean"] and not reevaluate:
            return True

        valid = sim.valid

        # general purpose validation
        sim.record["clean"] = valid

        return valid

    @staticmethod
    def _filter_by_error(
        sim_dir: Path, max_density_error: float, recalculate_errors: bool
    ) -> bool:
        """Filter a simulation directory by error."""

        try:
            simulation = WormSimulation.from_dir(sim_dir)
        except (OSError, KeyError, ValueError):
            log.error(f"Could not load simulation from {sim_dir}")
            return False

        try:
            if len(simulation.record["steps"]) == 0:
                simulation.record["steps"] = [{"error": None, "tau_max": None}]

            error_key = (
                "error"
                if "error" in simulation.record["steps"][-1]
                else "max_density_error"
            )

            if recalculate_errors:
                log.info(f"Recalculating errors for {simulation.save_dir}")

                simulation.record["steps"][-1][error_key] = simulation.max_density_error
                simulation.record["steps"][-1]["tau_max"] = simulation.max_tau_int

            result: bool = (
                simulation.record["steps"][-1][error_key] <= max_density_error
            ) and (simulation.record["steps"][-1]["tau_max"] > 0)
            return result

        except (IndexError, TypeError, KeyError) as e:
            log.error(f"Error {e} During error filtering for {simulation}")
            return False

    def _check_exists(self, sim_dir: Path) -> bool:
        """Check if a simulation directory already exists in the dataset."""

        dataset_samples_name = get_dataset_sample_name(sim_dir)
        dataset_sample_dir = self.dataset_samples_dir_path / dataset_samples_name
        if (
            dataset_sample_dir.exists()
            and (dataset_sample_dir / "inputs.pt").exists()
            and (dataset_sample_dir / "outputs.pt").exists()
            and (dataset_sample_dir / "metadata.json").exists()
        ):
            return True

        return False

    def filter(self, sim_dir: Path) -> bool:
        """Filter a simulation directory."""

        filter_result = True

        if not self.reevaluate and self._check_exists(sim_dir):
            return False

        if self.clean:
            filter_result &= self._get_simulation_valid(
                sim_dir, reevaluate=self.reevaluate
            )

        if self.max_density_error is not None:
            filter_result &= self._filter_by_error(
                sim_dir,
                max_density_error=self.max_density_error,
                recalculate_errors=self.recalculate_errors,
            )

        return filter_result


def load_sample(
    simulation_dir: Path,
    observables: list[str],
    reload: bool = False,
    overwrite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load a sample from a simulation directory."""

    inputs_path = simulation_dir / "inputs.pt"
    outputs_path = simulation_dir / "outputs.pt"
    errors_path = simulation_dir / "errors.pt"

    try:
        sim = WormSimulation.from_dir(simulation_dir)
        saved_observables = sim.record["saved_observables"]

    except:  # pylint: disable=bare-except # noqa: E722
        reload = True
    else:
        if not set(observables).issubset(set(saved_observables)):
            reload = True

    if (
        not inputs_path.exists()
        or not outputs_path.exists()
        or not errors_path.exists()
        or reload
    ):
        sim = WormSimulation.from_dir(simulation_dir)

        saved_observables = (
            sim.observables.observable_names["primary"]
            + sim.observables.observable_names["derived"]
        )

        expectation_values = [
            sim.observables.get_expectation_value(obs_type, obs_name)
            for obs_type in ["primary", "derived"]
            for obs_name in sim.observables.observable_names[obs_type]
        ]
        expanded_expectation_values = [
            (
                np.full(
                    shape=(sim.input_parameters.Lx, sim.input_parameters.Ly),
                    fill_value=obs,
                )
                if (obs is not None and obs.ndim == 0)
                else obs
            )
            for obs in expectation_values
        ]
        # stack observables
        outputs = torch.stack(
            [torch.from_numpy(obs) for obs in expanded_expectation_values],
            dim=0,
        )

        errors = [
            sim.observables.get_error_analysis(obs_type, obs_name)["error"]
            for obs_type in ["primary", "derived"]
            for obs_name in sim.observables.observable_names[obs_type]
        ]
        expanded_errors = [
            (
                np.full(
                    shape=(sim.input_parameters.Lx, sim.input_parameters.Ly),
                    fill_value=err,
                )
                if (err is not None and err.ndim == 0)
                else err
            )
            for err in errors
        ]
        errors = torch.stack(
            [torch.from_numpy(err) for err in expanded_errors],
            dim=0,
        )

        inputs = get_nn_input(
            sim.input_parameters.mu,
            sim.input_parameters.U_on,
            sim.input_parameters.V_nn,
            cb_projection=True,
            target_density=sim.observables.get_expectation_value("primary", "density"),
        )

        # save to .npy files
        if overwrite or not inputs_path.exists():
            torch.save(inputs, inputs_path)

        if overwrite or not outputs_path.exists():
            torch.save(outputs, outputs_path)

        if overwrite or not errors_path.exists():
            log.info(f"Saving errors to {errors_path}")
            torch.save(errors, errors_path)

        # save saved_observables
        sim.record["saved_observables"] = saved_observables

    else:
        inputs = torch.load(inputs_path, weights_only=True, map_location="cpu")
        outputs = torch.load(outputs_path, weights_only=True, map_location="cpu")
        errors = torch.load(errors_path, weights_only=True, map_location="cpu")

        # load saved_observables
        sim = WormSimulation.from_dir(simulation_dir)
        saved_observables = sim.record["saved_observables"]

        # filter observables
        outputs = outputs[[saved_observables.index(obs) for obs in observables]]
        errors = errors[[saved_observables.index(obs) for obs in observables]]

    metadata = {
        "max_density_error": sim.max_density_error,
        "L": sim.input_parameters.Lx,
        "J": sim.input_parameters.t_hop[0, 0, 0],
        "U_on": sim.input_parameters.U_on[0, 0],
        "mu": sim.input_parameters.mu_offset,
        "V_nn": sim.input_parameters.V_nn[0, 0, 0],
    }

    return inputs, outputs, errors, metadata


@app.command()
def load_dataset_simulations(  # pylint: disable=dangerous-default-value
    simulations_dir: Path,
    dataset_save_path: Path,
    include_tune_dirs: bool = False,
    clean: bool = True,
    reevaluate: bool = False,
    max_density_error: float | None = None,
    recalculate_errors: bool = False,
    delete_unreadable: bool = False,
    observables: list[str] = [
        "density",
        "density_density_corr_0",
        "density_density_corr_1",
        "density_density_corr_2",
        "density_density_corr_3",
        "density_squared",
        "density_max",
        "density_min",
        "density_variance",
    ],
) -> None:
    """Load simulation data from a directory containing simulation directories
    and save it to a dataset directory.

    Args:
        simulations_dir: Path to the directory containing the simulation
            directories.
        dataset_save_path: Path to the directory to save the dataset.
        include_tune_dirs: Include the tune directories in the dataset.
        clean: Clean the simulation directories before loading the dataset.
        reevaluate: Reload the simulation data even if it has already been saved.
        max_density_error: Maximum density error to include in the dataset.
        recalculate_errors: Recalculate the errors for the simulations.
        delete_unreadable: Delete unreadable simulation directories.
        observables: list of observables to include in the dataset.
    """
    log.info(
        "Loading dataset simulations with the following parameters:\n\n"
        f"simulations_dir = {simulations_dir}\n"
        f"dataset_save_path = {dataset_save_path}\n"
        f"include_tune_dirs = {include_tune_dirs}\n"
        f"clean = {clean}\n"
        f"reevaluate = {reevaluate}\n"
        f"max_density_error = {max_density_error}\n"
        f"recalculate_errors = {recalculate_errors}\n"
        f"delete_unreadable = {delete_unreadable}\n"
        f"observables = {observables}\n"
    )

    dataset_save_path.mkdir(exist_ok=True, parents=True)
    with open(dataset_save_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "observables": observables,
                "max_density_error": max_density_error,
            },
            f,
        )

    all_simulation_directories = list(reversed(sorted(simulations_dir.glob("*"))))

    if include_tune_dirs:
        all_simulation_directories += [
            directory / "tune" for directory in all_simulation_directories
        ]

    log.info(f"Found {len(all_simulation_directories)} simulation directories.")

    samples_dir = dataset_save_path / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)

    filter_strategy = FilterStrategy(
        clean=clean,
        max_density_error=max_density_error,
        recalculate_errors=recalculate_errors,
        reevaluate=reevaluate,
        dataset_samples_dir_path=samples_dir,
    )

    for sim_dir in tqdm(filter(filter_strategy.filter, all_simulation_directories)):
        inputs, outputs, errors, metadata = load_sample(
            sim_dir, list(observables), reload=reevaluate, overwrite=False
        )

        if (inputs[0] == 0).all():
            log.warning(f"Skipping simulation {sim_dir} with zero inputs. ")
            continue

        sample_save_path = samples_dir / (
            sim_dir.name
            if not sim_dir.name == "tune"
            else sim_dir.parent.name + "_tune"
        )
        sample_save_path.mkdir(exist_ok=True, parents=True)

        log.info(f"Saving sample to {sample_save_path} with observables {observables}")

        torch.save(inputs, sample_save_path / "inputs.pt")
        torch.save(outputs, sample_save_path / "outputs.pt")
        torch.save(errors, sample_save_path / "errors.pt")

        with open(sample_save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)


if __name__ == "__main__":
    app()
