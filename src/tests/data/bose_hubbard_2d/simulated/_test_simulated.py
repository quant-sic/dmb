import torch

from dmb.data.bose_hubbard_2d.simulated.dataset import RandomLDAMSampler
from dmb.data.bose_hubbard_2d.simulated.fake_phase_diagram_objects import (
    BOSE_HUBBARD_FAKE_ELLIPSOIDS,
    BOSE_HUBBARD_FAKE_GRADIENTS,
)


def test_phasediagram() -> None:
    # test that all densities are positive
    sampler = RandomLDAMSampler(
        ellipsoids=BOSE_HUBBARD_FAKE_ELLIPSOIDS, gradients=BOSE_HUBBARD_FAKE_GRADIENTS
    )

    for _ in range(10000):
        sample = sampler.sample()
        assert torch.all(sample["label"] >= 0)
