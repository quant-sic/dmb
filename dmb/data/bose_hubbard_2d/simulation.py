from abc import ABC, abstractmethod


class SimulationOutput(ABC):
    @property
    @abstractmethod
    def densities(self):
        pass
