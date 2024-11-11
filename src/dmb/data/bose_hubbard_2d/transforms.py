"""Transforms for the Bose-Hubbard 2D dataset."""

from typing import Callable, Literal

import torch
from attrs import define, field
from numpy.random import RandomState  # pylint: disable=no-name-in-module

from dmb.data.transforms import DMBTransform, InputOutputDMBTransform


class GaussianNoiseTransform(DMBTransform):
    """Transform that adds Gaussian noise to the input."""

    def __init__(self, mean: float, std: float) -> None:
        """Initialize the transform.

        Args:
            mean: Mean of the Gaussian noise.
            std: Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the input."""
        return x + torch.randn_like(x) * self.std + self.mean

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


@define
class SquareSymmetryGroupTransforms(InputOutputDMBTransform):
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

    random_number_generator: Callable[[], float] = field(default=RandomState(42).rand)

    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # with p=1/8 each choose one symmetry transform at random and apply it
        rnd = self.random_number_generator()

        if rnd < 1 / 8:  # unity
            pass
        elif rnd < 2 / 8:  # rotate 90 left

            def mapping(_x: torch.Tensor) -> torch.Tensor:
                return torch.rot90(_x, 1, [-2, -1])

            x, y = map(mapping, (x, y))

        elif rnd < 3 / 8:  # rotate 180 left

            def mapping(_x: torch.Tensor) -> torch.Tensor:
                return torch.rot90(_x, 2, [-2, -1])

            x, y = map(mapping, (x, y))

        elif rnd < 4 / 8:  # rotate 270 left

            def mapping(_x: torch.Tensor) -> torch.Tensor:
                return torch.rot90(_x, 3, [-2, -1])

            x, y = map(mapping, (x, y))

        elif rnd < 5 / 8:  # flip x

            def mapping(_x: torch.Tensor) -> torch.Tensor:
                return torch.flip(_x, [-2])

            x, y = map(mapping, (x, y))

        elif rnd < 6 / 8:  # flip y

            def mapping(_x: torch.Tensor) -> torch.Tensor:
                return torch.flip(_x, [-1])

            x, y = map(mapping, (x, y))

        elif rnd < 7 / 8:  # reflection x=y

            def mapping(_x: torch.Tensor) -> torch.Tensor:
                return torch.transpose(_x, -2, -1)

            x, y = map(mapping, (x, y))

        else:  # reflection x=-y

            def mapping(_x: torch.Tensor) -> torch.Tensor:
                return torch.flip(torch.transpose(_x, -2, -1), [-2, -1])

            x, y = map(mapping, (x, y))

        return x, y

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class TupleWrapperInTransform(InputOutputDMBTransform):
    """A wrapper for a DMBTransform that only applies the transform to the input."""

    def __init__(self, transform: DMBTransform):
        self.transform = transform

    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), y

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__()


class TupleWrapperOutTransform(InputOutputDMBTransform):
    """A wrapper for a DMBTransform that only applies the transform to the output."""

    def __init__(self, transform: DMBTransform):
        self.transform = transform

    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, self.transform(y)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__()


class BoseHubbard2dTransforms(InputOutputDMBTransform):
    """Transforms for the Bose-Hubbard 2D dataset."""

    def __init__(
        self,
        base_augmentations: list[InputOutputDMBTransform] | None = None,
        train_augmentations: list[InputOutputDMBTransform] | None = None,
    ):
        self.base_augmentations = ([] if base_augmentations is None else
                                   base_augmentations)
        self.train_augmentations = ([] if train_augmentations is None else
                                    train_augmentations)
        self._mode: Literal["base", "train"] = "base"

    @property
    def mode(self) -> Literal["base", "train"]:
        """Return the mode of the transform."""

        return self._mode

    @mode.setter
    def mode(self, mode: Literal["base", "train"]) -> None:
        if mode not in ("base", "train"):
            raise ValueError(f"mode must be either 'base' or 'train', but got {mode}")
        self._mode = mode

    def __call__(self, x: torch.Tensor,
                 y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for transform in self.base_augmentations:
            x, y = transform(x, y)

        if self.mode == "train":
            for transform in self.train_augmentations:
                x, y = transform(x, y)

        return x, y

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return (self.__class__.__name__ + "((\n" +
                ("\t base_augmentations={},\n".format(",".join(
                    map(str, self.base_augmentations))) if self.base_augmentations else
                 "") + ("\t train_augmentations={},\n".format(",".join(
                     map(str, self.train_augmentations)))
                        if self.train_augmentations else "") + f"\t mode={self.mode}"
                "\n"
                "\t)"
                "\n"
                ")")
