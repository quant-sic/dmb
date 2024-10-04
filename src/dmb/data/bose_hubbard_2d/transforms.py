from typing import Callable, Literal

import numpy as np
import torch


class GaussianNoise:

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={}, std={})".format(
            self.mean, self.std)


class SquareSymmetryGroupAugmentations:
    """Square symmetry group augmentations.

    This class implements the square symmetry group augmentations for 2D
    images. The following transformations are implemented:
    - identity
    - rotate 90 left
    - rotate 180 left
    - rotate 270 left
    - flip x
    - flip y
    - reflection x=y
    - reflection x=-y
    """

    def __call__(
        self, xy: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(xy, tuple):
            x = xy[0]
            y = xy[1]

        else:
            x = xy
            y = None

        def map_if_not_none(fn: Callable[[torch.Tensor], torch.Tensor],
                            x: torch.Tensor | None) -> torch.Tensor | None:
            if x is None:
                return None
            else:
                return fn(x)

        # with p=1/8 each choose one symmetry transform at random and apply it
        rnd = np.random.rand()

        if rnd < 1 / 8:  # unity
            pass
        elif rnd < 2 / 8:  # rotate 90 left
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.rot90(x, 1, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 3 / 8:  # rotate 180 left
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.rot90(x, 2, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 4 / 8:  # rotate 270 left
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.rot90(x, 3, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 5 / 8:  # flip x
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.flip(x, [-2]), xy),
                (x, y))
        elif rnd < 6 / 8:  # flip y
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.flip(x, [-1]), xy),
                (x, y))
        elif rnd < 7 / 8:  # reflection x=y
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.transpose(x, -2, -1), xy),
                (x, y),
            )
        else:  # reflection x=-y
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.flip(torch.transpose(x, -2, -1), [-2, -1]),
                    xy),
                (x, y),
            )

        if y is None:
            return x
        else:
            return x, y

    def __repr__(self):
        return self.__class__.__name__ + "()"


class TupleWrapperInTransform:

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.transform = transform

    def __call__(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            return self.transform(x[0]), x[1]
        else:
            return self.transform(x)

    def __repr__(self):
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__(
        )


class TupleWrapperOutTransform:

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.transform = transform

    def __call__(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            return x[0], self.transform(x[1])
        else:
            return self.transform(x)

    def __repr__(self):
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__(
        )


class BoseHubbard2dTransforms:

    def __init__(
        self,
        base_augmentations: list[Callable[[torch.Tensor],
                                          torch.Tensor]] = None,
        train_augmentations: list[Callable[[torch.Tensor],
                                           torch.Tensor]] = None,
    ):
        self.base_augmentations = ([] if base_augmentations is None else
                                   base_augmentations)
        self.train_augmentations = ([] if train_augmentations is None else
                                    train_augmentations)
        self._mode = "base"

    @property
    def mode(self) -> Literal["base", "train"]:
        return self._mode

    @mode.setter
    def mode(self, mode: Literal["base", "train"]):
        if mode not in ("base", "train"):
            raise ValueError(
                f"mode must be either 'base' or 'train', but got {mode}")
        self._mode = mode

    def __call__(
        self, x: tuple[torch.Tensor,
                       torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        for transform in self.base_augmentations:
            x = transform(x)

        if self.mode == "train":
            for transform in self.train_augmentations:
                x = transform(x)

        return x

    def __repr__(self):
        return (self.__class__.__name__ + "((\n" +
                ("\t base_augmentations={},\n".format(",".join(
                    map(str, self.base_augmentations)))
                 if self.base_augmentations else "") +
                ("\t train_augmentations={},\n".format(",".join(
                    map(str, self.train_augmentations)))
                 if self.train_augmentations else "") + f"\t mode={self.mode}"
                "\n"
                "\t)"
                "\n"
                ")")
