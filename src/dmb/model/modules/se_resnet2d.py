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
        squeeze_factor: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.conv1 = conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.se = SqueezeExcitation(out_channels, out_channels // squeeze_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BasicBlock module."""

        identity = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.permute(0, 3, 2, 1)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.permute(0, 3, 2, 1)

        out = self.se(out) + identity
        out = self.relu2(out)

        return out


class SeResNet2dv2(nn.Module):
    """SeResNet2d module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        n_channels: list[int],
        dropout: float = 0.0,
        squeeze_factor: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.dropout = dropout

        self.first_block = nn.Sequential(
            conv2d(
                in_channels=in_channels,
                out_channels=n_channels[0],
                kernel_size=kernel_sizes[0],
            ),
            nn.BatchNorm2d(n_channels[0]),
            SqueezeExcitation(n_channels[0], n_channels[0] // squeeze_factor),
            nn.ReLU(),
        )

        self.se_basic_block_list = []
        for block_idx in range(len(n_channels) - 1):
            self.se_basic_block_list.append(
                SeResnetBlock(
                    in_channels=n_channels[block_idx],
                    out_channels=n_channels[block_idx + 1],
                    kernel_size=kernel_sizes[block_idx],
                )
            )

        self.last_block = nn.Sequential(
            nn.Dropout2d(p=dropout),
            conv2d(
                in_channels=self.n_channels[-1],
                out_channels=self.out_channels,
                kernel_size=3,
                bias=True,
            ),
        )

        self.resnet = nn.Sequential(
            self.first_block, *self.se_basic_block_list, self.last_block
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SeResNet2d module."""
        out: torch.Tensor = self.resnet(x)
        return out
