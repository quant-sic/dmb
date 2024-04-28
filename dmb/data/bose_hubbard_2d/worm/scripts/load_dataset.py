import itertools
import shutil
from pathlib import Path

import numpy as np
import torch

from dmb.data.bose_hubbard_2d.nn_input import get_nn_input
from dmb.data.bose_hubbard_2d.worm.simulation import WormSimulation
from dmb.logging import create_logger
import json
import argparse

log = create_logger(__name__)


def get_simulation_valid(
    simulation_dir: Path,
    redo: bool = False,
    check_readable: bool = True,
) -> bool:

    if check_readable:
        try:
            sim = WormSimulation.from_dir(simulation_dir)
        except (OSError, KeyError, ValueError):
            log.error(f"Could not load simulation from {simulation_dir}")
            return False

    if "clean" in sim.record and not sim.record["clean"] and not redo:
        return False
    elif "clean" in sim.record and sim.record["clean"] and not redo:
        return True
    else:
        valid = sim.valid

    # general purpose validation
    sim.record["clean"] = valid

    return valid


def filter_by_error(
    sim_dir: Path,
    max_density_error: float,
    recalculate_errors: bool = False,
) -> bool:
    try:
        simulation = WormSimulation.from_dir(sim_dir)
    except (OSError, KeyError, ValueError):
        log.error(f"Could not load simulation from {sim_dir}")
        return False

    try:
        if recalculate_errors:
            log.info(f"Recalculating errors for {simulation.save_dir}")
            if len(simulation.record["steps"]) == 0:
                simulation.record["steps"] = [{"error": None, "tau_max": None}]

            simulation.record["steps"][-1]["error"] = simulation.max_density_error
            simulation.record["steps"][-1]["tau_max"] = simulation.max_tau_int

        return (simulation.record["steps"][-1]["error"] <= max_density_error) and (
            simulation.record["steps"][-1]["tau_max"] > 0
        )
    except (IndexError, TypeError, KeyError) as e:
        log.error(f"Error {e} During error filtering for {simulation}")
        return False


def clean_sim_dirs(
    sim_dirs,
    redo=False,
    verbose=False,
    max_density_error: float | None = None,
    recalculate_errors: bool = False,
    delete_unreadable: bool = False,
    check_readable: bool = True,
):

    valid_sim_dirs = [
        get_simulation_valid(
            sim_dir,
            redo=redo,
            check_readable=check_readable,
        )
        for sim_dir in sim_dirs
    ]

    if delete_unreadable:
        for sim_dir, valid in zip(sim_dirs, valid_sim_dirs):
            if not valid:
                log.info(f"Deleting {sim_dir}")
                shutil.rmtree(sim_dir)

    sim_dirs = list(
        itertools.compress(
            sim_dirs,
            valid_sim_dirs,
        )
    )

    if max_density_error is not None:
        sim_dirs = list(
            itertools.compress(
                sim_dirs,
                [
                    filter_by_error(
                        sim_dir,
                        max_density_error=max_density_error,
                        recalculate_errors=recalculate_errors,
                    )
                    for sim_dir in sim_dirs
                ],
            )
        )

    return sim_dirs


def load_sample(simulation_dir, observables, reload=False):

    inputs_path = simulation_dir / "inputs.pt"
    outputs_path = simulation_dir / "outputs.pt"

    try:
        sim = WormSimulation.from_dir(simulation_dir)
        saved_observables = sim.record["saved_observables"]
    except Exception:
        reload = True

    if not inputs_path.exists() or not outputs_path.exists() or reload:
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
                if obs.ndim == 0
                else obs
            )
            for obs in expectation_values
        ]
        # stack observables
        outputs = torch.stack(
            [torch.from_numpy(obs) for obs in expanded_expectation_values],
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
        torch.save(inputs, inputs_path)
        torch.save(outputs, outputs_path)

        # save saved_observables
        sim.record["saved_observables"] = saved_observables

    else:
        inputs = torch.load(inputs_path)
        outputs = torch.load(outputs_path)

        # load saved_observables
        sim = WormSimulation.from_dir(simulation_dir)
        saved_observables = sim.record["saved_observables"]

        # filter observables
        outputs = outputs[[saved_observables.index(obs) for obs in observables]]

    metadata = {
        "max_density_error": sim.max_density_error,
        "L": sim.input_parameters.Lx,
        "J": sim.input_parameters.t_hop[0, 0, 0],
        "U_on": sim.input_parameters.U_on[0, 0],
        "mu": sim.input_parameters.mu_offset,
        "V_nn": sim.input_parameters.V_nn[0, 0, 0],
    }

    return inputs, outputs, metadata


def load_dataset_simulations(
    simulations_dir: Path,
    dataset_save_path: Path,
    include_tune_dirs: bool = False,
    clean: bool = True,
    reload: bool = True,
    verbose: bool = True,
    max_density_error: float = 0.015,
    recalculate_errors: bool = False,
    delete_unreadable: bool = False,
    check_readable: bool = True,
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
):
    """Load simulation data from a directory containing simulation directories
    and save it to a dataset directory.

    Args:
        simulations_dir: Path to the directory containing the simulation
            directories.
        dataset_save_path: Path to the directory to save the dataset.
        include_tune_dirs: Include the tune directories in the dataset.
        clean: Clean the simulation directories before loading the dataset.
        reload: Reload the simulation data even if it has already been saved.
        verbose: Print verbose output.
        max_density_error: Maximum density error to include in the dataset.
        recalculate_errors: Recalculate the errors for the simulations.
        delete_unreadable: Delete unreadable simulation directories.
        check_readable: Check if the simulation is readable.
        observables: List of observables to include in the dataset.
    """
    dataset_save_path.mkdir(exist_ok=True, parents=True)
    with open(dataset_save_path / "metadata.json", "w") as f:
        json.dump(
            {
                "observables": observables,
                "max_density_error": max_density_error,
            },
            f,
        )

    all_simulation_directories = sorted(simulations_dir.glob("*"))

    if include_tune_dirs:
        all_simulation_directories += [
            all_simulation_directories / "tune"
            for directory in all_simulation_directories
        ]

    log.info(f"Found {len(all_simulation_directories)} simulation directories.")

    if clean:
        clean_simulation_directories = clean_sim_dirs(
            all_simulation_directories,
            redo=reload,
            verbose=verbose,
            max_density_error=max_density_error,
            recalculate_errors=recalculate_errors,
            delete_unreadable=delete_unreadable,
            check_readable=check_readable,
        )

    else:
        clean_simulation_directories = all_simulation_directories

    log.info(f"Found {len(clean_simulation_directories)} valid simulation directories.")

    samples_dir = dataset_save_path / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)

    for sim_dir in clean_simulation_directories:
        inputs, outputs, metadata = load_sample(sim_dir, observables, reload=reload)

        sample_save_path = samples_dir / sim_dir.name
        sample_save_path.mkdir(exist_ok=True, parents=True)

        torch.save(inputs, sample_save_path / "inputs.pt")
        torch.save(outputs, sample_save_path / "outputs.pt")

        with open(sample_save_path / "metadata.json", "w") as f:
            json.dump(metadata, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulations-dir",
        type=Path,
        required=True,
        help="Path to the directory containing the simulation directories.",
    )
    parser.add_argument(
        "--dataset-save-path",
        type=Path,
        required=True,
        help="Path to the directory to save the dataset.",
    )
    parser.add_argument(
        "--max-density-error",
        type=float,
        default=0.015,
        help="Maximum density error to include in the dataset.",
    )
    parser.add_argument(
        "--include-tune-dirs",
        action="store_true",
        help="Include the tune directories in the dataset.",
    )
    args = parser.parse_args()

    load_dataset_simulations(args.simulations_dir, args.dataset_save_path)
