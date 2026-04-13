from __future__ import annotations

import torch

from agldm.models.diffusion import LatentUNet
from agldm.models.vqvae import VQVAE


def test_vqvae_encode_decode_shapes() -> None:
    model = VQVAE(base_channels=16, latent_channels=32, codebook_size=64)
    images = torch.randn(2, 3, 256, 256)
    outputs = model(images)
    assert outputs["recon"].shape == images.shape
    assert outputs["quantized"].shape == (2, 32, 32, 32)


def test_latent_unet_preserves_latent_shape() -> None:
    model = LatentUNet(latent_channels=32, attribute_dim=8, text_dim=16, base_channels=32, conditioning_mode="full")
    latents = torch.randn(2, 32, 32, 32)
    timesteps = torch.randint(0, 1000, (2,))
    attrs = torch.randn(2, 8)
    text_context = torch.randn(2, 5, 16)
    outputs = model(latents, timesteps, text_context=text_context, attributes=attrs)
    assert outputs.shape == latents.shape
