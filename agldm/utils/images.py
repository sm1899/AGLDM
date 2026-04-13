from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_zero_one(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp((images + 1.0) * 0.5, 0.0, 1.0)


def normalize_for_resnet(images: torch.Tensor) -> torch.Tensor:
    images = to_zero_one(images)
    mean = IMAGENET_MEAN.to(images.device, images.dtype)
    std = IMAGENET_STD.to(images.device, images.dtype)
    return (images - mean) / std


def save_image_grid(images: torch.Tensor, path: str | Path, *, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(to_zero_one(images), nrow=nrow)
    save_image(grid, path)
