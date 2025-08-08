"""ResNet2d and SeResNet2d modules."""

import torch
import torch.nn.functional as F
from torch import nn


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


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, channels: int, reduction_factor: int = 16):
        super().__init__()
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_factor, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_factor, channels, 1),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            # Using 3 input channels for avg, max, and std pooling
            nn.Conv2d(3, 1, 5, padding=2, padding_mode="circular"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = self.channel_attention(x)
        x = x * torch.sigmoid(ca)

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        std_pool = torch.std(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool, std_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * torch.sigmoid(sa)

        return x


class FourierAttention(nn.Module):
    """Fourier-space Attention Module."""

    def __init__(self, channels: int, reduction_factor: int = 16):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_factor, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_factor, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Go to Fourier space
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))

        # Get magnitude for attention (abs is differentiable)
        fft_magnitude = torch.abs(x_fft)

        # Learn attention map in Fourier space
        # The attention network learns to weigh different frequencies
        f_attention = self.attention_net(fft_magnitude)

        # Apply attention in Fourier space
        x_fft_att = x_fft * f_attention

        # Go back to spatial domain
        x_att = torch.fft.irfft2(x_fft_att, s=x.shape[-2:], dim=(-2, -1))

        return x_att


class HybridAttention(nn.Module):
    """Combines CBAM and Fourier-space Attention with a learnable gate."""

    def __init__(self, channels: int, reduction_factor: int = 16):
        super().__init__()
        self.cbam = CBAM(channels, reduction_factor)
        self.fourier_attention = FourierAttention(channels, reduction_factor)

        # Learnable gate parameter to balance the two attention mechanisms
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get attended features from both modules
        x_cbam = self.cbam(x)
        x_fourier = self.fourier_attention(x)

        # Learn a weighted sum of the outputs
        # The sigmoid ensures the gate is between 0 and 1
        gate_val = torch.sigmoid(self.gate)
        return gate_val * x_cbam + (1 - gate_val) * x_fourier


class Block(nn.Module):
    """SeResNet Basic Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.0,
        reduction_factor: int = 4,
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

        self.attention = HybridAttention(
            out_channels, reduction_factor=reduction_factor
        )

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

        out = self.attention(out) + self.shortcut(identity)

        return out


class DMBNet2dv4(nn.Module):
    """SeResNet2d module with Feature Pyramid Network (FPN) style feature fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        n_channels: list[int],
        dropout: float = 0.0,
        reduction_factor: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.dropout = dropout

        # Feature extraction backbone
        self.backbone_blocks = nn.ModuleList()
        current_in_channels = in_channels
        for i, (channels, kernel_size) in enumerate(zip(n_channels, kernel_sizes)):
            if i == 0:
                # First block is just a convolution
                self.backbone_blocks.append(
                    nn.Sequential(
                        conv2d(
                            in_channels=current_in_channels,
                            out_channels=channels,
                            kernel_size=kernel_size,
                        ),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.backbone_blocks.append(
                    Block(
                        in_channels=current_in_channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dropout=dropout,
                        reduction_factor=reduction_factor,
                    )
                )
            current_in_channels = channels

        # FPN lateral connections - 1x1 convs to standardize channel dimensions
        fpn_channels = max(n_channels)  # Use the maximum channels as FPN feature dim
        self.lateral_convs = nn.ModuleList(
            [conv2d(channels, fpn_channels, kernel_size=1) for channels in n_channels]
        )

        # FPN output convolutions - 3x3 convs after feature fusion
        self.fpn_convs = nn.ModuleList(
            [conv2d(fpn_channels, fpn_channels, kernel_size=3) for _ in n_channels]
        )

        # Feature fusion and final output
        self.fusion_conv = conv2d(
            in_channels=fpn_channels * len(n_channels),
            out_channels=fpn_channels,
            kernel_size=3,
        )

        self.final_conv = conv2d(
            in_channels=fpn_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FPN-style DMBNet2d module."""

        # Forward pass through backbone to get multi-scale features
        features = []
        current = x
        for block in self.backbone_blocks:
            current = block(current)
            features.append(current)

        # Apply lateral convolutions to standardize channel dimensions
        lateral_features = []
        for feature, lateral_conv in zip(features, self.lateral_convs):
            lateral_features.append(lateral_conv(feature))

        # Apply FPN convolutions for final feature refinement
        fpn_features = []
        for lateral_feature, fpn_conv in zip(lateral_features, self.fpn_convs):
            fpn_features.append(fpn_conv(lateral_feature))

        # Concatenate all FPN features along channel dimension
        fused_features = torch.cat(fpn_features, dim=1)

        # Apply fusion convolution
        fused = self.fusion_conv(fused_features)

        # Final output convolution
        return self.final_conv(fused)
