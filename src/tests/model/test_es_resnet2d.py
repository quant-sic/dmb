from typing import Callable

import torch
from pytest_cases import fixture, parametrize

from dmb.model.modules.es_resnet2d import EsSeResNet2d


def case_D4_group_actions() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "identity": lambda x: x,
        "rotate_90_left": lambda x: torch.rot90(x, 1, [-2, -1]),
        "rotate_180_left": lambda x: torch.rot90(x, 2, [-2, -1]),
        "rotate_270_left": lambda x: torch.rot90(x, 3, [-2, -1]),
        "flip_x": lambda x: torch.flip(x, [-2]),
        "flip_y": lambda x: torch.flip(x, [-1]),
        "reflection_x_y": lambda x: torch.transpose(x, -2, -1),
        "reflection_x_neg_y": lambda x: torch.flip(
            torch.transpose(x, -2, -1), [-2, -1]
        ),
    }


class TestEsSeResNet2d:
    @fixture(name="model", scope="class")
    def model(self) -> EsSeResNet2d:
        return EsSeResNet2d(
            in_channels=4,
            out_channels=7,
            kernel_sizes=[5, 3, 3],
            n_channels=[16, 16, 16],
            dropout=0.0,
            se_squeeze_factor=4,
        )

    @parametrize(argnames="name,action", argvalues=case_D4_group_actions().items())
    def test_action(
        self,
        model: EsSeResNet2d,
        name: str,
        action: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        x = torch.randn(9, 4, 32, 32)
        x_transformed = action(x).clone()

        y = model(x)
        y_transformed = model(x_transformed)

        assert torch.allclose(action(y), y_transformed, atol=1e-5)
