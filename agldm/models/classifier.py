from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_backbone(pretrained: bool = True) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = ResNet50_Weights.IMAGENET1K_V2
        except Exception:
            weights = None
    try:
        return resnet50(weights=weights)
    except Exception:
        return resnet50(weights=None)


class AttributeClassifier(nn.Module):
    def __init__(self, num_attributes: int, *, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = build_resnet50_backbone(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_attributes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
