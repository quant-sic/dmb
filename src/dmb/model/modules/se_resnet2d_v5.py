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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get attended features from both modules
        x_cbam = self.cbam(x)
        x_fourier_cbam = self.fourier_attention(x_cbam)

        return x_fourier_cbam


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


class WeightedFeatureFusion(nn.Module):
    """Weighted feature fusion for BiFPN."""

    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Weighted fusion of input features."""
        # Normalize weights using softmax
        weights = F.softmax(self.weights, dim=0)

        # Weighted sum of features
        fused = sum(w * x for w, x in zip(weights, inputs))
        return fused


class DMBNet2dv5(nn.Module):
    """SeResNet2d module with Bidirectional Feature Pyramid Network (BiFPN) feature fusion.

    BiFPN introduces bidirectional cross-scale connections and weighted feature fusion
    to efficiently combine multi-scale features. Key improvements over FPN:
    - Top-down and bottom-up pathways for bidirectional information flow
    - Learnable weights for feature fusion at each level
    - Repeated BiFPN blocks for iterative feature refinement
    - Optional layer selection for applying BiFPN only to specified indices
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_sizes: List of kernel sizes for each layer
        n_channels: List of channel counts for each layer
        dropout: Dropout probability
        reduction_factor: Channel reduction factor for attention modules
        bifpn_repeats: Number of BiFPN block repetitions
        bifpn_layer_indices: List of layer indices to apply BiFPN to. If None, applies to all layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        n_channels: list[int],
        dropout: float = 0.0,
        reduction_factor: int = 4,
        bifpn_repeats: int = 2,
        bifpn_layer_indices: list[int] | None = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.n_channels = n_channels
        self.dropout = dropout
        self.bifpn_repeats = bifpn_repeats
        
        # Default to using all layers if no specific indices provided
        if bifpn_layer_indices is None:
            self.bifpn_layer_indices = list(range(len(n_channels)))
        else:
            self.bifpn_layer_indices = bifpn_layer_indices
        
        # Validate indices
        for idx in self.bifpn_layer_indices:
            if idx < 0 or idx >= len(n_channels):
                raise ValueError(f"bifpn_layer_indices contains invalid index {idx}. Must be in range [0, {len(n_channels)-1}]")

        # Feature extraction backbone
        self.backbone_blocks = nn.ModuleList()
        current_in_channels = in_channels
        for i, (channels, kernel_size) in enumerate(zip(n_channels, kernel_sizes)):
            if i == 0:
                # First block is just a convolution (no BN/ReLU to avoid redundancy with pre-activation blocks)
                self.backbone_blocks.append(
                    conv2d(
                        in_channels=current_in_channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
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

        # BiFPN lateral connections - 1x1 convs to standardize channel dimensions
        # Only for specified layer indices
        bifpn_channels = max(
            [n_channels[i] for i in self.bifpn_layer_indices]
        )  # Use the maximum channels from selected layers as BiFPN feature dim
        self.lateral_convs = nn.ModuleDict()
        for i in self.bifpn_layer_indices:
            self.lateral_convs[str(i)] = conv2d(n_channels[i], bifpn_channels, kernel_size=1)

        # BiFPN intermediate convolutions for each level
        self.bifpn_convs = nn.ModuleList()
        for _ in range(bifpn_repeats):
            level_convs = nn.ModuleDict()
            for i in self.bifpn_layer_indices:
                level_convs[str(i)] = conv2d(bifpn_channels, bifpn_channels, kernel_size=3)
            self.bifpn_convs.append(level_convs)

        # Weighted fusion modules for BiFPN
        self.td_weights = nn.ModuleList()  # Top-down weights
        self.bu_weights = nn.ModuleList()  # Bottom-up weights

        for _ in range(bifpn_repeats):
            # Top-down path weights (skip first level as it has no higher level)
            td_level_weights = nn.ModuleDict()
            for i, layer_idx in enumerate(self.bifpn_layer_indices):
                if i == 0:
                    # Highest level: no fusion needed
                    td_level_weights[str(layer_idx)] = None
                else:
                    # Lower levels: fuse with higher level
                    td_level_weights[str(layer_idx)] = WeightedFeatureFusion(2)
            self.td_weights.append(td_level_weights)

            # Bottom-up path weights (skip last level as it has no lower level)
            bu_level_weights = nn.ModuleDict()
            for i, layer_idx in enumerate(self.bifpn_layer_indices):
                if i == len(self.bifpn_layer_indices) - 1:
                    # Lowest level: no fusion needed
                    bu_level_weights[str(layer_idx)] = None
                else:
                    # Higher levels: fuse with lower level and lateral connection
                    bu_level_weights[str(layer_idx)] = WeightedFeatureFusion(3)
            self.bu_weights.append(bu_level_weights)

        # Feature fusion and final output
        self.fusion_conv = conv2d(
            in_channels=bifpn_channels * len(self.bifpn_layer_indices),
            out_channels=bifpn_channels,
            kernel_size=3,
        )

        self.final_conv = conv2d(
            in_channels=bifpn_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BiFPN-style DMBNet2d module."""

        # Forward pass through backbone to get multi-scale features
        backbone_features = {}
        current = x
        for i, block in enumerate(self.backbone_blocks):
            current = block(current)
            if i in self.bifpn_layer_indices:
                backbone_features[i] = current

        # Apply lateral convolutions only to specified layer indices
        lateral_features = {}
        for layer_idx in self.bifpn_layer_indices:
            lateral_features[layer_idx] = self.lateral_convs[str(layer_idx)](backbone_features[layer_idx])

        # BiFPN repeated blocks - only process specified layers
        current_features = lateral_features
        for repeat_idx in range(self.bifpn_repeats):
            # Top-down pathway
            td_features = {}
            
            # Process layers in order of indices
            sorted_indices = sorted(self.bifpn_layer_indices)
            td_features[sorted_indices[0]] = current_features[sorted_indices[0]]  # Highest level unchanged

            for i in range(1, len(sorted_indices)):
                current_idx = sorted_indices[i]
                higher_idx = sorted_indices[i - 1]
                
                # Upsample higher level feature to match current level
                higher_level = F.interpolate(
                    td_features[higher_idx],
                    size=current_features[current_idx].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                # Weighted fusion of upsampled higher level and current lateral
                td_features[current_idx] = self.td_weights[repeat_idx][str(current_idx)](
                    [higher_level, current_features[current_idx]]
                )

                # Apply convolution
                td_features[current_idx] = self.bifpn_convs[repeat_idx][str(current_idx)](td_features[current_idx])

            # Bottom-up pathway
            bu_features = {}
            bu_features[sorted_indices[-1]] = td_features[sorted_indices[-1]]  # Lowest level unchanged

            for i in range(len(sorted_indices) - 2, -1, -1):
                current_idx = sorted_indices[i]
                lower_idx = sorted_indices[i + 1]
                
                # Downsample lower level feature to match current level
                lower_level = F.adaptive_avg_pool2d(
                    bu_features[lower_idx], output_size=td_features[current_idx].shape[-2:]
                )

                # Weighted fusion of downsampled lower level, top-down, and lateral
                bu_features[current_idx] = self.bu_weights[repeat_idx][str(current_idx)](
                    [td_features[current_idx], lower_level, current_features[current_idx]]
                )

                # Apply convolution
                bu_features[current_idx] = self.bifpn_convs[repeat_idx][str(current_idx)](bu_features[current_idx])

            # Update current features for next iteration
            current_features = bu_features

        # Resize all BiFPN features to the same spatial resolution (use the first layer's size)
        first_layer_idx = sorted(self.bifpn_layer_indices)[0]
        target_size = current_features[first_layer_idx].shape[-2:]
        resized_features = []
        for layer_idx in sorted(self.bifpn_layer_indices):
            feature = current_features[layer_idx]
            if feature.shape[-2:] != target_size:
                resized_feature = F.interpolate(
                    feature, size=target_size, mode="bilinear", align_corners=False
                )
            else:
                resized_feature = feature
            resized_features.append(resized_feature)

        # Concatenate all BiFPN features along channel dimension
        fused_features = torch.cat(resized_features, dim=1)

        # Apply fusion convolution
        fused = self.fusion_conv(fused_features)

        # Final output convolution
        return self.final_conv(fused)
