"""Implementation of a 2D equivariant Squeeze-and-Excitation ResNet."""

import math
from functools import partial
from typing import Callable

import escnn.nn as enn
import torch
from escnn import gspaces


def conv2d(
    in_type: enn.FieldType,
    out_type: enn.FieldType,
    kernel_size: int,
    bias: bool = False,
    stride: int = 1,
    dilation: int = 1,
    sigma: float | None = None,
    frequencies_cutoff: float | Callable[[float], float] | None = None,
) -> enn.R2Conv:
    """2D convolution with circular padding."""
    return enn.R2Conv(
        in_type,
        out_type,
        kernel_size=kernel_size,
        bias=bias,
        sigma=sigma,
        frequencies_cutoff=frequencies_cutoff,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        padding_mode="circular",
    )


def regular_feature_type(gspace: gspaces.GSpace,
                         planes: int,
                         fixparams: bool = True) -> enn.FieldType:
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0

    N = gspace.fibergroup.order()

    if fixparams:
        planes *= int(math.sqrt(N))

    planes = planes / N

    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace,
                         planes: int,
                         fixparams: bool = True) -> enn.FieldType:
    """ build a trivial feature map with the specified number of channels"""

    if fixparams:
        planes *= int(math.sqrt(gspace.fibergroup.order()))

    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}


class SqueezeExcitation(enn.EquivariantModule):
    """This block implements the Squeeze-and-Excitation block
    from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation``
    correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional):
            ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]):
            ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        edge_type: enn.FieldType,
        inner_type: enn.FieldType,
        activation: Callable[..., enn.EquivariantModule] = enn.ReLU,
        scale_activation: Callable[..., enn.EquivariantModule] | str = "p_sigmoid",
    ) -> None:
        super().__init__()

        self.avgpool = enn.PointwiseAdaptiveAvgPool2D(edge_type, output_size=1)
        self.fc1 = enn.R2Conv(edge_type, inner_type, kernel_size=1)
        self.fc2 = enn.R2Conv(inner_type, edge_type, kernel_size=1)
        self.activation = activation(inner_type)

        if not isinstance(scale_activation, enn.EquivariantModule):
            self.scale_activation = enn.PointwiseNonLinearity(
                edge_type, scale_activation)

    def _scale(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        scale = self._scale(x)
        return enn.GeometricTensor(x.tensor * scale.tensor, x.type, x.coords)

    def evaluate_output_shape(self, input_shape: torch.Size) -> torch.Size:
        return input_shape


class BasicSeBlock(enn.EquivariantModule):
    """Basic block for the Squeeze-and-Excitation ResNet."""

    def __init__(
        self,
        in_type: enn.FieldType,
        se_inner_type: enn.FieldType,
        inner_type: enn.FieldType,
        dropout_rate: float,
        stride: int = 1,
        kernel_size: int = 3,
        out_type: enn.FieldType = None,
        sigma: float | None = None,
        frequencies_cutoff: float | Callable[[float], float] | None = None,
    ):
        super().__init__()

        if out_type is None:
            out_type = in_type

        self.in_type = in_type
        self.out_type = out_type

        assert isinstance(in_type.gspace, gspaces.GSpace2D)

        conv_func = partial(
            conv2d,
            sigma=sigma,
            frequencies_cutoff=frequencies_cutoff,
        )

        self.bn1 = enn.InnerBatchNorm(self.in_type)

        self.se = SqueezeExcitation(self.in_type, inner_type=se_inner_type)

        self.relu1 = enn.ReLU(self.in_type)
        self.conv1 = conv_func(
            self.in_type,
            inner_type,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.bn2 = enn.InnerBatchNorm(inner_type)
        self.relu2 = enn.ReLU(inner_type)

        self.dropout = enn.PointwiseDropout(inner_type, p=dropout_rate)

        self.conv2 = conv_func(inner_type, self.out_type, kernel_size=kernel_size)

        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv_func(self.in_type,
                                      self.out_type,
                                      kernel_size=1,
                                      stride=stride)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        x_n = self.relu1(self.se(self.bn1(x)))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            out = out + self.shortcut(x_n)
        else:
            out = out + x

        return out

    def evaluate_output_shape(self, input_shape: torch.Size) -> torch.Size:
        return input_shape


class EsSeResNet2d(torch.nn.Module):
    """Equivariant steerable Squeeze-and-Excitation ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        n_channels: list[int],
        dropout: float = 0.0,
        frequencies_cutoff: float | Callable[[float], float] | None = None,
        sigma: float | None = None,
        se_squeeze_factor: int = 4,
    ):
        super().__init__()

        self.gspace = gspaces.flipRot2dOnR2(4)  # D4 group

        self.in_type = enn.FieldType(self.gspace,
                                     [self.gspace.trivial_repr] * in_channels)

        out_type = FIELD_TYPE["regular"](self.gspace, n_channels[0], fixparams=True)

        conv_func = partial(
            conv2d,
            sigma=sigma,
            frequencies_cutoff=frequencies_cutoff,
        )

        first_block = enn.SequentialModule(*[
            conv_func(self.in_type, out_type, kernel_size=kernel_sizes[0], bias=False),
        ])

        subsequent_blocks = []
        for i_block in range(len(n_channels) - 1):

            in_type = FIELD_TYPE["regular"](self.gspace,
                                            n_channels[i_block],
                                            fixparams=True)
            se_inner_type = FIELD_TYPE["regular"](self.gspace,
                                                  n_channels[i_block] //
                                                  se_squeeze_factor,
                                                  fixparams=True)
            inner_type = FIELD_TYPE["regular"](self.gspace,
                                               n_channels[i_block],
                                               fixparams=True)
            out_type = FIELD_TYPE["regular"](self.gspace,
                                             n_channels[i_block + 1],
                                             fixparams=True)

            tmp_block = BasicSeBlock(
                in_type=in_type,
                inner_type=inner_type,
                se_inner_type=se_inner_type,
                dropout_rate=dropout,
                stride=1,
                kernel_size=kernel_sizes[i_block],
                out_type=out_type,
                sigma=sigma,
                frequencies_cutoff=frequencies_cutoff,
            )

            subsequent_blocks.append(tmp_block)

        final_type = enn.FieldType(self.gspace,
                                   [self.gspace.trivial_repr] * out_channels)
        last_block = enn.SequentialModule(*[
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type),
            enn.PointwiseDropout(out_type, p=dropout),
            conv_func(out_type, final_type, kernel_size=3, bias=False),
        ])

        self.es_resnet = enn.SequentialModule(first_block, *subsequent_blocks,
                                              last_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""

        # wrap the input tensor in a GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)

        # apply the network
        out: torch.Tensor = self.es_resnet(x).tensor

        return out
