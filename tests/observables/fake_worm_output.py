from dmb.data.bose_hubbard_2d.worm_qmc.worm.outputs import SimulationOutput
import numpy as np


class FakeWormOutput(SimulationOutput):

    def __init__(self, densities):
        self._densities = densities
        self._reshape_densities = densities.reshape(
            densities.shape[0],
            int(np.sqrt(densities.shape[1])),
            int(np.sqrt(densities.shape[1])),
        )

    @property
    def densities(self):
        return self._densities

    @property
    def reshape_densities(self):
        return self._reshape_densities
