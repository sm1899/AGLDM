from __future__ import annotations

import lpips
import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, channels, height, width = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, channels)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(bsz, height, width, channels).permute(0, 3, 1, 2).contiguous()

        codebook_loss = torch.mean((quantized.detach() - z) ** 2)
        commitment_loss = self.beta * torch.mean((quantized - z.detach()) ** 2)
        quantized = z + (quantized - z).detach()
        return quantized, {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "indices": indices.view(bsz, height, width),
        }


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, base_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 256,
        codebook_size: int = 1024,
        commitment_beta: float = 0.25,
        lpips_weight: float = 1.0,
        adv_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, latent_channels)
        self.decoder = Decoder(in_channels, base_channels, latent_channels)
        self.quantizer = VectorQuantizer(codebook_size, latent_channels, commitment_beta)
        self.perceptual_loss = lpips.LPIPS(net="vgg")
        self.lpips_weight = lpips_weight
        self.adv_weight = adv_weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_quantized(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        quantized, _ = self.quantizer(z)
        return quantized

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, q_stats = self.quantizer(z_e)
        recon = self.decoder(z_q)
        recon_loss = torch.mean((x - recon) ** 2)
        perceptual = self.perceptual_loss(x, recon).mean()
        return {
            "recon": recon,
            "quantized": z_q,
            "indices": q_stats["indices"],
            "recon_loss": recon_loss,
            "lpips_loss": perceptual * self.lpips_weight,
            "codebook_loss": q_stats["codebook_loss"],
            "commitment_loss": q_stats["commitment_loss"],
        }
