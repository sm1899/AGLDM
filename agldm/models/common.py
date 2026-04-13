from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10_000) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        angles = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out
        h = self.out_layers(h)
        return h + self.skip(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.context_proj = nn.Linear(context_dim, channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if context is None:
            return x
        bsz, channels, height, width = x.shape
        residual = x
        x = self.proj_in(self.norm(x))
        tokens = x.flatten(2).transpose(1, 2)
        context = self.context_proj(context)
        attended, _ = self.attn(tokens, context, context, need_weights=False)
        attended = attended.transpose(1, 2).reshape(bsz, channels, height, width)
        return residual + self.proj_out(attended)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
