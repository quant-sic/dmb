import numpy as np

from dmb.data.bose_hubbard_2d.simulation import SimulationOutput


class WormObservables:
    """Class for computing observables from a simulation."""

    observables: dict[str, callable] = {}
    observable_shapes: dict[str, tuple[str, ...]] = {}

    @classmethod
    def observable_names(cls):
        """Return the names of the observables that can be computed."""
        return cls.observables.keys()

    def __init__(self, output: SimulationOutput):
        self.output = output

    @classmethod
    def register(cls, name, shape) -> callable:
        """Register an observable with the class.

        Args:
            name: The name of the observable.
            shape: The shape of the observable.

        Returns:
            A decorator that registers the function as an observable.
        """

        def wrapper(func):
            cls.observables[name] = func
            cls.observable_shapes[name] = shape
            return func

        return wrapper

    def __getitem__(self, key):
        """Return the value of the observable with the given key."""
        return self.observables[key](self.output)

    def __contains__(self, key):
        """Return whether the observable with the given key is present."""
        return key in self.observables


@WormObservables.register("density_variance", ("Lx", "Ly"))
def get_density_variance(output: SimulationOutput) -> np.ndarray:
    return output.reshape_densities.var(axis=0)


@WormObservables.register("density", ("Lx", "Ly"))
def get_density_mean(output: SimulationOutput) -> np.ndarray:
    return output.reshape_densities.mean(axis=0)


@WormObservables.register("density_density_corr_0", ("Lx", "Ly"))
def get_density_density_corr_0(output: SimulationOutput) -> np.ndarray:
    return (
        np.roll(output.reshape_densities, axis=1, shift=1) * output.reshape_densities
    ).mean(axis=0)


@WormObservables.register("density_density_corr_1", ("Lx", "Ly"))
def get_density_density_corr_1(output: SimulationOutput) -> np.ndarray:
    densities = output.reshape_densities
    return (np.roll(densities, axis=2, shift=1) * densities).mean(axis=0)


@WormObservables.register("density_density_corr_2", ("Lx", "Ly"))
def get_density_density_corr_2(output: SimulationOutput) -> np.ndarray:
    densities = output.reshape_densities
    return (
        np.roll(np.roll(densities, axis=1, shift=1), axis=2, shift=1) * densities
    ).mean(axis=0)


@WormObservables.register("density_density_corr_3", ("Lx", "Ly"))
def get_density_density_corr_3(output: SimulationOutput) -> np.ndarray:
    densities = output.reshape_densities
    return (
        np.roll(np.roll(densities, axis=1, shift=-1), axis=2, shift=1) * densities
    ).mean(axis=0)


@WormObservables.register("density_squared", ("Lx", "Ly"))
def get_density_squared_mean(output: SimulationOutput) -> np.ndarray:
    return (output.reshape_densities**2).mean(axis=0)


@WormObservables.register("density_max", tuple())
def get_density_max(output: SimulationOutput) -> np.ndarray:
    return output.reshape_densities.max(axis=(-1, -2)).mean(axis=0)


@WormObservables.register("density_min", tuple())
def get_density_min(output: SimulationOutput) -> np.ndarray:
    return output.reshape_densities.min(axis=(-1, -2)).mean(axis=0)
