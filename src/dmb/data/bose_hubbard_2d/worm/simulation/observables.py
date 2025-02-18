"""Module for computing observables from a simulation."""

import concurrent.futures
import itertools
from copy import deepcopy
from logging import Logger
from typing import Any, Callable, cast

import numpy as np
from attrs import define

from auto_correlation import Analysis, DerivedAnalysis, \
    GammaPathologicalError, PrimaryAnalysis
from dmb.logging import create_logger

from .output import Output

log = create_logger(__name__)


def reshape_if_not_none(
    results: dict[str, np.ndarray] | None, attribute: str, shape: tuple[int, ...]
) -> np.ndarray | None:
    """Reshape the attribute of the results if it is not None."""
    if results is None:
        return None

    attribute_value = getattr(results, attribute, None)

    if attribute_value is None:
        return None

    attribute_value_array: np.ndarray = np.array(attribute_value)

    if np.prod(attribute_value_array.shape) == np.prod(shape):
        reshaped_array: np.ndarray = attribute_value_array.reshape(shape)
        return reshaped_array

    return attribute_value_array


def ulli_wolff_mc_error_analysis(
    samples: np.ndarray,
    timeout: int = 300,
    logging_instance: Logger = log,
    derived_quantity: Callable | None = None,
) -> dict[str, np.ndarray | None]:
    """Perform the Ulli Wolff Monte Carlo error analysis on the given samples.

    Args:
        samples: The samples to perform the analysis on.
            Shape: (number_of_samples, Lx, Ly)
        timeout: The maximum time to wait for the analysis to finish.
        logging_instance: The logger to use for logging.
        derived_quantity: The function to apply to the samples before performing
            the analysis. Acts on the last two dimensions of the
            samples and returns a scalar.

    Returns:
        The errors of the observable, or None if the analysis failed.
    """
    sample_shape = samples.shape[1:]
    number_individual_observables = int(np.prod(sample_shape))
    individual_observable_names = (
        [
            "".join(
                [f"{int(idx/shape_i)}" for shape_i in sample_shape[:-1]]
                + [f"{idx% sample_shape[-1]}"]
            )
            for idx in range(number_individual_observables)
        ]
        if number_individual_observables > 1
        else ["0"]
    )

    samples_reshaped = samples.reshape(
        1, samples.shape[0], number_individual_observables
    )

    def function_shape_adjuster(
        function: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Adjust the shape of input data to the function."""

        def wrapper(data: np.ndarray) -> np.ndarray:
            return function(data.reshape(*data.shape[:-1], *sample_shape))

        return wrapper

    if derived_quantity is None:
        analysis: Analysis = PrimaryAnalysis(
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
            function_shape_adjuster(derived_quantity)
        )
        variance = np.array(derived_quantity(samples).var(axis=0))

    try:

        def run_analysis() -> Any:
            return analysis.errors()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_analysis)
            error_analysis_results = future.result(timeout=timeout)

    except GammaPathologicalError as e:
        logging_instance.warning(f"GammaPathologicalError: {e}")
        error_analysis_results = None

    except concurrent.futures.TimeoutError:
        logging_instance.warning(
            "TimeoutError: The Ulli Wolff Error analysis timed out after "
            f"{timeout} seconds."
        )
        error_analysis_results = None

    expectation_value = reshape_if_not_none(
        mean_analysis_results, "value", sample_shape
    )

    error = reshape_if_not_none(error_analysis_results, "dvalue", sample_shape)
    naive_error = reshape_if_not_none(error_analysis_results, "naive_err", sample_shape)
    tau_int = reshape_if_not_none(error_analysis_results, "tau_int", sample_shape)
    tau_int_error = reshape_if_not_none(
        error_analysis_results, "dtau_int", sample_shape
    )
    error_on_error = reshape_if_not_none(
        error_analysis_results, "ddvalue", sample_shape
    )

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
    sample_function: Callable | None = None
    derived_quantity: Callable | None = None
    max_number_of_samples: int = 10000

    _error_previous_samples: np.ndarray | None = None
    _error_previous_results: dict[str, np.ndarray | None] | None = None

    def _subsample(self, samples: np.ndarray) -> np.ndarray:
        """Subsample the given samples if there are too many."""
        if samples.shape[0] > self.max_number_of_samples:
            skip = samples.shape[0] // self.max_number_of_samples
            samples = samples[::skip]

        return samples

    def expectation_value(self, output: Output) -> np.ndarray | None:
        """Return the expectation value of the observable."""
        densities = output.densities

        if densities is None:
            return None

        samples = self._subsample(densities)

        mapped_samples = (
            self.sample_function(samples) if self.sample_function else samples
        )
        derived_quantity = (
            self.derived_quantity(mapped_samples)
            if self.derived_quantity
            else mapped_samples
        )

        return np.array(derived_quantity.mean(axis=0))

    def error_analysis(self, output: Output) -> dict[str, np.ndarray | None]:
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

        if self._error_previous_samples is not None and np.array_equal(
            samples, self._error_previous_samples
        ):
            return cast(dict[str, np.ndarray | None], self._error_previous_results)

        mapped_samples = (
            self.sample_function(samples) if self.sample_function else samples
        )

        results = ulli_wolff_mc_error_analysis(
            samples=mapped_samples, derived_quantity=self.derived_quantity
        )

        # store the results and samples for future reference
        self._error_previous_results = results
        self._error_previous_samples = deepcopy(samples)

        return results


class SimulationObservables:
    """Class for computing observables from a simulation."""

    primary_observables: dict[str, DensityDerivedObservable] = {}
    derived_observables: dict[str, DensityDerivedObservable] = {}
    observable_names: dict[str, list[str]] = {
        "primary": [],
        "derived": [],
    }

    def __init__(self, output: Output):
        self.output = output

    @classmethod
    def register_primary(cls, name: str) -> Callable:
        """Register an observable with the class.

        Args:
            name: The name of the observable.

        Returns:
            A decorator that registers the function as an observable.
        """

        def wrapper(sample_function: Callable) -> Callable:
            cls.primary_observables[name] = DensityDerivedObservable(
                name=name,
                sample_function=sample_function,
            )
            cls.observable_names["primary"].append(name)

            return sample_function

        return wrapper

    @classmethod
    def register_derived(cls, name: str) -> Callable:
        """Register a derived observable with the class.

        Args:
            name: The name of the observable.

        Returns:
            A decorator that registers the function as an observable.
        """

        def wrapper(derived_quantity: Callable) -> Callable:
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
    ) -> np.ndarray | None:
        """Return the observable with the given name."""
        if observable_type == "primary":
            expectation_value: np.ndarray | None = self.primary_observables[
                name
            ].expectation_value(self.output)
            return expectation_value

        if observable_type == "derived":
            expectation_value = self.derived_observables[name].expectation_value(
                self.output
            )
            return expectation_value

        raise ValueError(f"Invalid observable type: {observable_type}")

    def get_error_analysis(
        self,
        observable_type: str,
        name: str,
    ) -> dict[str, np.ndarray | None]:
        """Return the error of the observable with the given name."""

        if observable_type == "primary":
            return self.primary_observables[name].error_analysis(self.output)

        if observable_type == "derived":
            return self.derived_observables[name].error_analysis(self.output)

        raise ValueError(f"Invalid observable type: {observable_type}")

    def __contains__(self, key: str) -> bool:
        """Return whether the observable with the given key is present."""
        return key in itertools.chain(
            self.primary_observables, self.derived_observables
        )


# sample_function = None
SimulationObservables.register_primary("density")(sample_function=None)


@SimulationObservables.register_primary("density_density_corr_0")
def get_density_density_corr_0(samples: np.ndarray) -> np.ndarray:
    """Return density-density correlation with shift (1, 0)."""

    density_corr_0: np.ndarray = np.roll(samples, axis=-2, shift=1) * samples
    return density_corr_0


@SimulationObservables.register_primary("density_density_corr_1")
def get_density_density_corr_1(samples: np.ndarray) -> np.ndarray:
    """Return density-density correlation with shift (0, 1)."""

    density_corr_1: np.ndarray = np.roll(samples, axis=-1, shift=1) * samples
    return density_corr_1


@SimulationObservables.register_primary("density_density_corr_2")
def get_density_density_corr_2(samples: np.ndarray) -> np.ndarray:
    """Return density-density correlation with shift (1, 1)."""

    density_corr_2: np.ndarray = (
        np.roll(np.roll(samples, axis=-2, shift=1), axis=2, shift=1) * samples
    )
    return density_corr_2


@SimulationObservables.register_primary("density_density_corr_3")
def get_density_density_corr_3(samples: np.ndarray) -> np.ndarray:
    """Return density-density correlation with shift (1, -1)."""

    density_corr_3: np.ndarray = (
        np.roll(np.roll(samples, axis=-2, shift=-1), axis=-1, shift=1) * samples
    )
    return density_corr_3


@SimulationObservables.register_primary("density_squared")
def get_density_squared(samples: np.ndarray) -> np.ndarray:
    """Return the square of the observable along the last two dimensions."""

    samples_squared: np.ndarray = samples**2
    return samples_squared


@SimulationObservables.register_primary("density_max")
def get_density_max(samples: np.ndarray) -> np.ndarray:
    """Return the maximum of the observable along the last two dimensions."""

    _max: np.ndarray = samples.max(axis=(-1, -2))
    return _max


@SimulationObservables.register_primary("density_min")
def get_density_min(samples: np.ndarray) -> np.ndarray:
    """Return the minimum of the observable along the last two dimensions."""

    _min: np.ndarray = samples.min(axis=(-1, -2))
    return _min


@SimulationObservables.register_primary("density_variance")
def get_variance(samples: np.ndarray) -> np.ndarray:
    """Return the variance of the observable along the last two dimensions."""

    var: np.ndarray = samples.var(axis=(-1, -2))
    return var
