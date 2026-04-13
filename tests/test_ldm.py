from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from agldm.training.ldm import compute_self_consistency_loss, train_ldm


class DummyVQVAE(nn.Module):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


class DummyClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))


def test_self_consistency_loss_backpropagates() -> None:
    predicted_x0 = torch.randn(2, 3, 16, 16, requires_grad=True)
    attrs = torch.zeros(2, 3)
    loss = compute_self_consistency_loss(predicted_x0, attrs, DummyVQVAE(), DummyClassifier())
    loss.backward()
    assert predicted_x0.grad is not None


def test_train_ldm_requires_frozen_checkpoints() -> None:
    fixtures = Path(__file__).resolve().parent / "fixtures" / "runtime"
    config = {
        "data": {
            "manifest_path": str(fixtures / "manifest.jsonl"),
            "image_size": 256,
            "num_workers": 0,
            "attribute_dim": 312,
        },
        "model": {
            "vqvae": {"latent_channels": 256},
            "classifier": {"pretrained": False},
            "text_encoder": {"model_name": "openai/clip-vit-base-patch32"},
            "ldm": {"latent_channels": 256, "base_channels": 32, "num_heads": 4},
        },
        "train": {
            "output_root": str(fixtures),
            "seed": 42,
            "ldm": {"batch_size": 1, "eval_batch_size": 1, "epochs": 1, "lr": 1e-4},
        },
    }
    with pytest.raises(FileNotFoundError, match="requires frozen checkpoints"):
        train_ldm(config)
