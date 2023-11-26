from typing import Dict, Literal, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.models.segmentation.deeplabv3 import ASPP

from dmb.dmb.model.models.regnet_2d import RegNet400mf
from dmb.dmb.model.models.resnet_2d import ResNet18, SeResNet18


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward_impl(self, x: Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        x = features
        x = self.classifier(x)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if isinstance(x, (tuple, list)):
            out = tuple(self.forward_impl(_x) for _x in x)
        else:
            out = self.forward_impl(x)

        return out


class DeepLabV3Alike(_SimpleSegmentationModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: Literal[
            "regnet_y_400mf", "resnet18", "se_resnet18"
        ] = "regnet_y_400mf",
    ) -> None:
        if backbone == "resnet18":
            backbone = ResNet18(in_channels=in_channels)
            head_in_channels = 512
        elif backbone == "se_resnet18":
            backbone = SeResNet18(in_channels=in_channels)
            head_in_channels = 512
        elif backbone == "regnet_y_400mf":
            backbone = RegNet400mf(in_channels=in_channels)
            head_in_channels = 440
        else:
            raise ValueError(
                "backbone must be either resnet18 or regnet_y_400mf. Got: {}".format(
                    backbone
                )
            )

        head = DeepLabHead(head_in_channels, out_channels)
        super().__init__(backbone, head)
