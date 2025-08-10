"""Transforms for the Bose-Hubbard 2D dataset."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from attrs import field, frozen

from dmb.data.dataset import DMBData
from dmb.data.transforms import (
    DMBDatasetTransform,
    DMBTransform,
    GroupElement,
    InputOutputDMBTransform,
)
from dmb.logging import create_logger

log = create_logger(__name__)


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


@frozen
class D4Group:
    """Square symmetry group."""

    elements: dict[str, GroupElement] = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Initialize the group elements."""
        object.__setattr__(
            self,
            "elements",
            {
                "identity": GroupElement("identity", self.identity, self.identity),
                "rotate_90_left": GroupElement(
                    "rotate_90_left", self.rotate_90_left, self.rotate_270_left
                ),
                "rotate_180_left": GroupElement(
                    "rotate_180_left", self.rotate_180_left, self.rotate_180_left
                ),
                "rotate_270_left": GroupElement(
                    "rotate_270_left", self.rotate_270_left, self.rotate_90_left
                ),
                "flip_x": GroupElement("flip_x", self.flip_x, self.flip_x),
                "flip_y": GroupElement("flip_y", self.flip_y, self.flip_y),
                "reflection_x_y": GroupElement(
                    "reflection_x_y", self.reflection_x_y, self.reflection_x_y
                ),
                "reflection_x_neg_y": GroupElement(
                    "reflection_x_neg_y",
                    self.reflection_x_neg_y,
                    self.reflection_x_neg_y,
                ),
            },
        )

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        """Identity transformation."""
        return x

    def rotate_90_left(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees left."""
        return torch.rot90(x, 1, [-2, -1])

    def rotate_180_left(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate 180 degrees left."""
        return torch.rot90(x, 2, [-2, -1])

    def rotate_270_left(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate 270 degrees left."""
        return torch.rot90(x, 3, [-2, -1])

    def flip_x(self, x: torch.Tensor) -> torch.Tensor:
        """Flip along the x-axis."""
        return torch.flip(x, [-2])

    def flip_y(self, x: torch.Tensor) -> torch.Tensor:
        """Flip along the y-axis."""
        return torch.flip(x, [-1])

    def reflection_x_y(self, x: torch.Tensor) -> torch.Tensor:
        """Reflect along the x=y line."""
        return torch.transpose(x, -2, -1)

    def reflection_x_neg_y(self, x: torch.Tensor) -> torch.Tensor:
        """Reflect along the x=-y line."""
        return torch.flip(torch.transpose(x, -2, -1), [-2, -1])


class D4GroupTransforms(InputOutputDMBTransform):
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

    def __init__(self, generator: np.random.Generator | None = None) -> None:
        """Initialize the transform."""
        self.d4_group: D4Group = D4Group()

        self.generator = generator or np.random.default_rng(seed=42)

    def __call__(self, dmb_data: DMBData) -> DMBData:
        # with p=1/8 each choose one symmetry transform at random and apply it
        rnd = int(self.generator.uniform(0, 8))
        element_transform = list(self.d4_group.elements.values())[rnd]

        dmb_data_out = DMBData(
            inputs=element_transform.transform(dmb_data.inputs),
            outputs=element_transform.transform(dmb_data.outputs),
            sample_id=dmb_data.sample_id,
            group_elements=dmb_data.group_elements + [element_transform],
        )

        return dmb_data_out

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the transform."""
        return {
            "generator_state": self.generator.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load the state of the transform."""
        self.generator.bit_generator.state = state["generator_state"]
        log.info("Loaded generator state: %s", self.generator.bit_generator.state)


class TupleWrapperInTransform(InputOutputDMBTransform):
    """A wrapper for a DMBTransform that only applies the transform to the input."""

    def __init__(self, transform: DMBTransform):
        self.transform = transform

    def __call__(self, dmb_data: DMBData) -> DMBData:
        return DMBData(
            inputs=self.transform(dmb_data.inputs),
            outputs=dmb_data.outputs,
            sample_id=dmb_data.sample_id,
            group_elements=dmb_data.group_elements,
        )

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__()

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the transform."""
        return self.transform.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load the state of the transform."""
        self.transform.load_state_dict(state)


class TupleWrapperOutTransform(InputOutputDMBTransform):
    """A wrapper for a DMBTransform that only applies the transform to the output."""

    def __init__(self, transform: DMBTransform):
        self.transform = transform

    def __call__(self, dmb_data: DMBData) -> DMBData:
        return DMBData(
            inputs=dmb_data.inputs,
            outputs=self.transform(dmb_data.outputs),
            sample_id=dmb_data.sample_id,
            group_elements=dmb_data.group_elements,
        )

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__()

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the transform."""
        return self.transform.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load the state of the transform."""
        self.transform.load_state_dict(state)


class BoseHubbard2dTransforms(DMBDatasetTransform):
    """Transforms for the Bose-Hubbard 2D dataset."""

    def __init__(
        self,
        base_augmentations: list[InputOutputDMBTransform] | None = None,
        train_augmentations: list[InputOutputDMBTransform] | None = None,
    ):
        self.base_augmentations = (
            [] if base_augmentations is None else base_augmentations
        )
        self.train_augmentations = (
            [] if train_augmentations is None else train_augmentations
        )

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

    def __call__(self, dmb_data: DMBData) -> DMBData:
        for transform in self.base_augmentations:
            dmb_data = transform(dmb_data)

        if self.mode == "train":
            for transform in self.train_augmentations:
                dmb_data = transform(dmb_data)

        return dmb_data

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return (
            self.__class__.__name__
            + "((\n"
            + (
                "\t base_augmentations={},\n".format(
                    ",".join(map(str, self.base_augmentations))
                )
                if self.base_augmentations
                else ""
            )
            + (
                "\t train_augmentations={},\n".format(
                    ",".join(map(str, self.train_augmentations))
                )
                if self.train_augmentations
                else ""
            )
            + f"\t mode={self.mode}"
            "\n"
            "\t)"
            "\n"
            ")"
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the transform."""
        return {
            "base_augmentations": [aug.state_dict() for aug in self.base_augmentations],
            "train_augmentations": [
                aug.state_dict() for aug in self.train_augmentations
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load the state of the transform."""
        for aug, augmentation_state in zip(
            self.base_augmentations, state["base_augmentations"]
        ):
            aug.load_state_dict(augmentation_state)
        for aug, augmentation_state in zip(
            self.train_augmentations, state["train_augmentations"]
        ):
            aug.load_state_dict(augmentation_state)
