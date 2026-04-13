from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from agldm.models.common import CrossAttentionBlock, Downsample, ResBlock, SinusoidalTimeEmbedding, Upsample


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    values = buffer.gather(0, timesteps)
    return values.reshape((timesteps.shape[0],) + (1,) * (len(x_shape) - 1))


class LatentUNet(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        attribute_dim: int,
        text_dim: int,
        base_channels: int = 128,
        num_heads: int = 8,
        conditioning_mode: str = "full",
    ) -> None:
        super().__init__()
        self.conditioning_mode = conditioning_mode
        emb_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.attr_embedding = nn.Sequential(
            nn.Linear(attribute_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_conv = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResBlock(base_channels, base_channels, emb_dim)
        self.attn1 = CrossAttentionBlock(base_channels, text_dim, num_heads=num_heads)
        self.downsample1 = Downsample(base_channels)

        self.down2 = ResBlock(base_channels, base_channels * 2, emb_dim)
        self.attn2 = CrossAttentionBlock(base_channels * 2, text_dim, num_heads=num_heads)
        self.downsample2 = Downsample(base_channels * 2)

        self.mid1 = ResBlock(base_channels * 2, base_channels * 4, emb_dim)
        self.mid_attn = CrossAttentionBlock(base_channels * 4, text_dim, num_heads=num_heads)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, emb_dim)

        self.upsample2 = Upsample(base_channels * 4)
        self.up2 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, emb_dim)
        self.upattn2 = CrossAttentionBlock(base_channels * 2, text_dim, num_heads=num_heads)

        self.upsample1 = Upsample(base_channels * 2)
        self.up1 = ResBlock(base_channels * 2 + base_channels, base_channels, emb_dim)
        self.upattn1 = CrossAttentionBlock(base_channels, text_dim, num_heads=num_heads)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    def _resolve_context(
        self,
        text_context: torch.Tensor | None,
        attributes: torch.Tensor | None,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        emb = self.time_embedding(timesteps)
        if self.conditioning_mode != "text_only" and attributes is not None:
            emb = emb + self.attr_embedding(attributes)
        if self.conditioning_mode == "attr_only":
            text_context = None
        return text_context, emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        text_context: torch.Tensor | None,
        attributes: torch.Tensor | None,
    ) -> torch.Tensor:
        text_context, emb = self._resolve_context(text_context, attributes, timesteps)

        x0 = self.in_conv(x)
        x1 = self.down1(x0, emb)
        x1 = self.attn1(x1, text_context)
        x2 = self.downsample1(x1)

        x2 = self.down2(x2, emb)
        x2 = self.attn2(x2, text_context)
        x3 = self.downsample2(x2)

        x3 = self.mid1(x3, emb)
        x3 = self.mid_attn(x3, text_context)
        x3 = self.mid2(x3, emb)

        y = self.upsample2(x3)
        y = torch.cat([y, x2], dim=1)
        y = self.up2(y, emb)
        y = self.upattn2(y, text_context)

        y = self.upsample1(y)
        y = torch.cat([y, x1], dim=1)
        y = self.up1(y, emb)
        y = self.upattn1(y, text_context)
        return self.out(y)


class DDPMScheduler(nn.Module):
    def __init__(self, num_train_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim=0)

        self.num_train_steps = num_train_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_recip_alpha_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alpha_cumprod",
            torch.sqrt(torch.clamp(1.0 / alpha_cumprod - 1.0, min=0.0)),
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_train_steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape
        ) * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return _extract(self.sqrt_recip_alpha_cumprod, timesteps, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alpha_cumprod, timesteps, x_t.shape
        ) * noise

    def ddim_timesteps(self, num_steps: int) -> list[int]:
        stride = max(self.num_train_steps // num_steps, 1)
        return list(range(self.num_train_steps - 1, -1, -stride))


@dataclass(slots=True)
class DDIMState:
    sample: torch.Tensor
    pred_x0: torch.Tensor


def ddim_step(
    scheduler: DDPMScheduler,
    sample: torch.Tensor,
    pred_noise: torch.Tensor,
    timestep: int,
    next_timestep: int,
) -> DDIMState:
    device = sample.device
    t = torch.full((sample.shape[0],), timestep, device=device, dtype=torch.long)
    pred_x0 = scheduler.predict_start_from_noise(sample, t, pred_noise)

    if next_timestep < 0:
        return DDIMState(sample=pred_x0, pred_x0=pred_x0)

    alpha_t = scheduler.alpha_cumprod[timestep]
    alpha_next = scheduler.alpha_cumprod[next_timestep]
    sigma = 0.0
    direction = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma**2, min=0.0)) * pred_noise
    next_sample = torch.sqrt(alpha_next) * pred_x0 + direction
    return DDIMState(sample=next_sample, pred_x0=pred_x0)
