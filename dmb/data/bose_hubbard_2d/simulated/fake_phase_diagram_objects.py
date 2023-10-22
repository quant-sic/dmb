import torch
from dmb.data.bose_hubbard_2d.simulated.dataset import Ellipsoid, Gradient

BOSE_HUBBARD_FAKE_ELLIPSOIDS = [
    Ellipsoid(
        center=torch.tensor((0.5, 0, 1), dtype=torch.float32),
        params=torch.tensor((0.5, 0.35, 0.2), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=1.0,
    ),
    Ellipsoid(
        center=torch.tensor((1.5, 0, 1), dtype=torch.float32),
        params=torch.tensor((0.5, 0.2, 0.15), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=1.0,
        max_value=1.0,
    ),
    Ellipsoid(
        center=torch.tensor((2.5, 0, 1), dtype=torch.float32),
        params=torch.tensor((0.5, 0.2, 0.25), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=1.0,
        max_value=2.0,
    ),
    Ellipsoid(
        center=torch.tensor((2.5, 0, 1.5), dtype=torch.float32),
        params=torch.tensor((0.5, 0.7, 0.25), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=3.0,
    ),
    Ellipsoid(
        center=torch.tensor((1.5, 0, 1.5), dtype=torch.float32),
        params=torch.tensor((0.5, 0.5, 0.15), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=2.0,
    ),
    Ellipsoid(
        center=torch.tensor((0.5, 0, 1.5), dtype=torch.float32),
        params=torch.tensor((0.5, 0.3, 0.15), dtype=torch.float32),
        separation_exponent=20.0,
        min_value=0.0,
        max_value=1.0,
    ),
]

BOSE_HUBBARD_FAKE_GRADIENTS = [
    Gradient(
        anchor=torch.tensor((0, 0, 0), dtype=torch.float32),
        anchor_value=0.0,
        direction=torch.tensor((1, 0, 0), dtype=torch.float32),
        direction_value=1,
    )
]
