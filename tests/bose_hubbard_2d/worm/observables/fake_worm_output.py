import numpy as np


class FakeWormOutput:

    def __init__(self, densities):
        self._densities = densities

    @property
    def densities(self):
        return self._densities
