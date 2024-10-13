import concurrent.futures
import itertools
from copy import deepcopy
from logging import Logger
from typing import Callable

import numpy as np
from attrs import define
from auto_correlation import DerivedAnalysis, GammaPathologicalError, \
    PrimaryAnalysis

from dmb.logging import create_logger

from .output import WormOutput

log = create_logger(__name__)


def reshape_if_not_none(results: dict[str, np.ndarray] | None, attribute: str,
                        shape: tuple[int, ...]) -> np.ndarray | None:
    """Reshape the attribute of the results if it is not None."""
    if results is None:
        return None

    attribute_value = np.array(getattr(results, attribute))

    if attribute_value is None:
        return None

    if np.prod(attribute_value.shape) == np.prod(shape):
        return attribute_value.reshape(shape)
    else:
        return attribute_value


def ulli_wolff_mc_error_analysis(
    samples: np.ndarray,
    timeout: int = 300,
    logging_instance: Logger = log,
    derived_quantity: Callable = None,
) -> dict[str, np.ndarray | None]:
    """Perform the Ulli Wolff Monte Carlo error analysis on the given samples.

    Args:
        samples: The samples to perform the analysis on.
            Shape: (number_of_samples, Lx, Ly)
        timeout: The maximum time to wait for the analysis to finish.
        logging_instance: The logger to use for logging.
        derived_quantity: The function to apply to the samples before performing the analysis.
            Acts on the last two dimensions of the samples and returns a scalar.

    Returns:
        The errors of the observable, or None if the analysis failed.
    """
    sample_shape = samples.shape[1:]
    number_individual_observables = int(np.prod(sample_shape))
    individual_observable_names = [
        "".join([f"{int(idx/shape_i)}" for shape_i in sample_shape[:-1]] +
                [f"{idx% sample_shape[-1]}"])
        for idx in range(number_individual_observables)
    ] if number_individual_observables > 1 else ["0"]

    samples_reshaped = samples.reshape(1, samples.shape[0],
                                       number_individual_observables)

    def function_shape_adjuster(function: Callable) -> Callable:
        """Adjust the shape of input data to the function."""

        def wrapper(data: np.ndarray) -> np.ndarray:
            return function(data.reshape(*data.shape[:-1], *sample_shape))

        return wrapper

    if derived_quantity is None:
        analysis = PrimaryAnalysis(
            data=samples_reshaped,
            rep_sizes=[len(samples)],
            name=individual_observable_names,
        )
        # calculate the mean of the observable
        mean_analysis_results = analysis.mean()
        variance = np.array(samples.var(axis=0))

    else:
        analysis = DerivedAnalysis(
            data=samples_reshaped,
            rep_sizes=[len(samples)],
            name=individual_observable_names,
        )
        # calculate the mean of the observable
        analysis.mean()
        mean_analysis_results = analysis.apply(
            function_shape_adjuster(derived_quantity))
        variance = np.array(derived_quantity(samples).var(axis=0))

    try:

        def run_analysis():
            return analysis.errors()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_analysis)
            error_analysis_results = future.result(timeout=timeout)

    except GammaPathologicalError as e:
        logging_instance.warning(f"GammaPathologicalError: {e}")
        error_analysis_results = None

    except concurrent.futures.TimeoutError:
        logging_instance.warning(
            f"TimeoutError: The Ulli Wolff Error analysis timed out after {timeout} seconds."
        )
        error_analysis_results = None

    expectation_value = reshape_if_not_none(mean_analysis_results, "value",
                                            sample_shape)

    error = reshape_if_not_none(error_analysis_results, "dvalue", sample_shape)
    naive_error = reshape_if_not_none(error_analysis_results, "naive_err",
                                      sample_shape)
    tau_int = reshape_if_not_none(error_analysis_results, "tau_int",
                                  sample_shape)
    tau_int_error = reshape_if_not_none(error_analysis_results, "dtau_int",
                                        sample_shape)
    error_on_error = reshape_if_not_none(error_analysis_results, "ddvalue",
                                         sample_shape)

    return {
        "expectation_value": expectation_value,
        "variance": variance,
        "error": error,
        "naive_error": naive_error,
        "tau_int": tau_int,
        "tau_int_error": tau_int_error,
        "error_on_error": error_on_error,
    }


@define
class DensityDerivedObservable:
    """Class for computing observables and errors derived from the density."""

    name: str
    sample_function: callable = None
    derived_quantity: callable = None
    max_number_of_samples: int = 10000

    _error_previous_samples: np.ndarray = None
    _error_previous_results: dict[str, np.ndarray] = None

    def _subsample(self, samples: np.ndarray) -> np.ndarray:
        """Subsample the given samples if there are too many."""
        if samples.shape[0] > self.max_number_of_samples:
            skip = samples.shape[0] // self.max_number_of_samples
            samples = samples[::skip]

        return samples

    def expectation_value(self, output: WormOutput) -> np.ndarray | None:
        """Return the expectation value of the observable."""
        densities = output.densities

        if densities is None:
            return None

        samples = self._subsample(densities)

        mapped_samples = (self.sample_function(samples)
                          if self.sample_function else samples)
        derived_quantity = (self.derived_quantity(mapped_samples)
                            if self.derived_quantity else mapped_samples)

        return np.array(derived_quantity.mean(axis=0))

    def error_analysis(self,
                       output: WormOutput) -> dict[str, np.ndarray | None]:
        """Return the error of the observable."""
        densities = output.densities

        if densities is None:
            return {
                "expectation_value": None,
                "variance": None,
                "error": None,
                "naive_error": None,
                "tau_int": None,
                "tau_int_error": None,
                "error_on_error": None,
            }

        samples = self._subsample(densities)

        if np.array_equal(samples, self._error_previous_samples):
            return self._error_previous_results

        mapped_samples = (self.sample_function(samples)
                          if self.sample_function else samples)

        results = ulli_wolff_mc_error_analysis(
            samples=mapped_samples, derived_quantity=self.derived_quantity)

        # store the results and samples for future reference
        self._error_previous_results = results
        self._error_previous_samples = deepcopy(samples)

        return results


class SimulationObservables:
    """Class for computing observables from a simulation."""

    primary_observables: dict[str, callable] = {}
    derived_observables: dict[str, callable] = {}
    observable_names: dict[str, list[str]] = {
        "primary": [],
        "derived": [],
    }

    def __init__(self, output: WormOutput):
        self.output = output

    @classmethod
    def register_primary(cls, name) -> callable:
        """Register an observable with the class.

        Args:
            name: The name of the observable.

        Returns:
            A decorator that registers the function as an observable.
        """

        def wrapper(sample_function):
            cls.primary_observables[name] = DensityDerivedObservable(
                name=name,
                sample_function=sample_function,
            )
            cls.observable_names["primary"].append(name)

            return sample_function

        return wrapper

    @classmethod
    def register_derived(cls, name) -> callable:
        """Register a derived observable with the class.

        Args:
            name: The name of the observable.

        Returns:
            A decorator that registers the function as an observable.
        """

        def wrapper(derived_quantity):
            cls.derived_observables[name] = DensityDerivedObservable(
                name=name,
                derived_quantity=derived_quantity,
            )
            cls.observable_names["derived"].append(name)

            return derived_quantity

        return wrapper

    def get_expectation_value(
        self,
        observable_type: str,
        name: str,
    ) -> dict[str, np.ndarray | None]:
        """Return the observable with the given name."""
        if observable_type == "primary":
            return self.primary_observables[name].expectation_value(
                self.output)
        elif observable_type == "derived":
            return self.derived_observables[name].expectation_value(
                self.output)
        else:
            raise ValueError(f"Invalid observable type: {observable_type}")

    def get_error_analysis(
        self,
        observable_type: str,
        name: str,
    ) -> dict[str, np.ndarray | None]:
        """Return the error of the observable with the given name."""
        if observable_type == "primary":
            return self.primary_observables[name].error_analysis(self.output)
        elif observable_type == "derived":
            return self.derived_observables[name].error_analysis(self.output)
        else:
            raise ValueError(f"Invalid observable type: {observable_type}")

    def __contains__(self, key):
        """Return whether the observable with the given key is present."""
        return key in itertools.chain(self.primary_observables.keys(),
                                      self.derived_observables.keys())


# sample_function = None
SimulationObservables.register_primary("density")(sample_function=None)


@SimulationObservables.register_primary("density_density_corr_0")
def get_density_density_corr_0(samples: np.ndarray) -> np.ndarray:
    return np.roll(samples, axis=-2, shift=1) * samples


@SimulationObservables.register_primary("density_density_corr_1")
def get_density_density_corr_1(samples: np.ndarray) -> np.ndarray:
    return np.roll(samples, axis=-1, shift=1) * samples


@SimulationObservables.register_primary("density_density_corr_2")
def get_density_density_corr_2(samples: np.ndarray) -> np.ndarray:
    return np.roll(np.roll(samples, axis=-2, shift=1), axis=2,
                   shift=1) * samples


@SimulationObservables.register_primary("density_density_corr_3")
def get_density_density_corr_3(samples: np.ndarray) -> np.ndarray:
    return np.roll(np.roll(samples, axis=-2, shift=-1), axis=-1,
                   shift=1) * samples


@SimulationObservables.register_primary("density_squared")
def get_density_squared(samples: np.ndarray) -> np.ndarray:
    return samples**2


@SimulationObservables.register_primary("density_max")
def get_density_max(samples: np.ndarray) -> float:
    return samples.max(axis=(-1, -2))


@SimulationObservables.register_primary("density_min")
def get_density_min(samples: np.ndarray) -> float:
    return samples.min(axis=(-1, -2))


@SimulationObservables.register_primary("density_variance")
def get_variance(samples: np.ndarray) -> float:
    return samples.var(axis=(-1, -2))
