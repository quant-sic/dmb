import json

import pytest

from dmb.data.bose_hubbard_2d.worm.simulation import WormInputParameters
from dmb.paths import REPO_DATA_ROOT
from tests.data.bose_hubbard_2d.worm.simulation.utils import WormInputParametersDecoder


@pytest.fixture(scope="class", name="input_parameters")
def fixture_input_parameters() -> WormInputParameters:
    with open(REPO_DATA_ROOT / "test/input_parameters.json") as f:
        params = json.load(f, cls=WormInputParametersDecoder)["parameters"]

    return WormInputParameters(**params)
