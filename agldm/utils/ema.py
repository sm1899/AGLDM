from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad = False

    def to(self, device: str) -> "ExponentialMovingAverage":
        self.shadow.to(device)
        return self

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters(), strict=True):
                shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
            for shadow_buffer, model_buffer in zip(self.shadow.buffers(), model.buffers(), strict=True):
                shadow_buffer.data.copy_(model_buffer.data)
