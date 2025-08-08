"""ResNet2d and SeResNet2d modules."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.misc import SqueezeExcitation


def conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = False,
    stride: int = 1,
) -> nn.Conv2d:
    """Return a 2D convolutional layer with circular padding.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the kernel.
        bias: Whether to include bias in the convolution.

    Returns:
        nn.Conv2d: 2D convolutional layer with circular padding.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="circular",
        bias=bias,
        stride=stride,
    )


class SeResnetBlock(nn.Module):
    """SeResNet Basic Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.0,
        squeeze_factor: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self.se = SqueezeExcitation(out_channels, out_channels // squeeze_factor)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BasicBlock module."""

        identity = x

        out: torch.Tensor = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.dropout1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.se(out) + self.shortcut(identity)

        return out


class SeResNet2dv3(nn.Module):
    """SeResNet2d module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        n_channels: list[int],
        strides: list[int] = None,
        dropout: float = 0.0,
        squeeze_factor: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.dropout = dropout

        if strides is None:
            strides = [1] * len(n_channels)

        self.first_block = nn.Sequential(
            conv2d(
                in_channels=in_channels,
                out_channels=n_channels[0],
                kernel_size=kernel_sizes[0],
                stride=strides[0],
            )
        )

        self.se_basic_block_list = nn.ModuleList()
        for block_idx in range(len(n_channels) - 1):
            self.se_basic_block_list.append(
                SeResnetBlock(
                    in_channels=n_channels[block_idx],
                    out_channels=n_channels[block_idx + 1],
                    kernel_size=kernel_sizes[block_idx + 1],
                    stride=strides[block_idx + 1],
                    dropout=dropout,
                    squeeze_factor=squeeze_factor,
                )
            )

        self.last_block = nn.Sequential(
            conv2d(
                in_channels=self.n_channels[-1],
                out_channels=self.out_channels,
                kernel_size=3,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SeResNet2d module."""
        x = self.first_block(x)
        for block in self.se_basic_block_list:
            x = block(x)

        out: torch.Tensor = self.last_block(x)
        return out
