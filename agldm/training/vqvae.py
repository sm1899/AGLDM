from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from agldm.training.common import build_stage_loaders, build_vqvae_from_config, checkpoint_paths, resolve_device
from agldm.utils.checkpointing import save_checkpoint
from agldm.utils.images import save_image_grid
from agldm.utils.logging import get_logger
from agldm.utils.seed import set_seed
from agldm.models.vqvae import PatchDiscriminator

LOGGER = get_logger(__name__)


def _bce_loss(logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    targets = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def _validate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, torch.Tensor]:
    model.eval()
    total = 0.0
    count = 0
    sample_pair: torch.Tensor | None = None
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        loss = outputs["recon_loss"] + outputs["lpips_loss"] + outputs["codebook_loss"] + outputs["commitment_loss"]
        total += loss.item() * images.size(0)
        count += images.size(0)
        if sample_pair is None:
            sample_pair = torch.cat([images[:4], outputs["recon"][:4]], dim=0).cpu()
    return total / max(count, 1), sample_pair if sample_pair is not None else torch.empty(0)


def train_vqvae(config: dict) -> None:
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]["vqvae"]
    output_root = config["train"]["output_root"]

    set_seed(int(config["train"].get("seed", data_config.get("seed", 42))))
    device = resolve_device()
    train_loader, val_loader = build_stage_loaders(data_config, train_config, random_text=False)

    model = build_vqvae_from_config(model_config).to(device)
    disc = PatchDiscriminator(
        in_channels=int(model_config["vqvae"].get("in_channels", 3)),
        base_channels=int(model_config["vqvae"].get("disc_base_channels", 64)),
    ).to(device)

    gen_opt = torch.optim.Adam(model.parameters(), lr=float(train_config["lr"]), betas=(0.5, 0.9))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=float(train_config.get("disc_lr", train_config["lr"])), betas=(0.5, 0.9))
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    ckpts = checkpoint_paths(output_root)
    sample_dir = Path(output_root) / "samples" / "vqvae"

    for epoch in range(int(train_config["epochs"])):
        model.train()
        disc.train()
        progress = tqdm(train_loader, desc=f"vqvae epoch {epoch + 1}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)

            gen_opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                fake_logits = disc(outputs["recon"])
                adv_loss = _bce_loss(fake_logits, True) * float(model_config["vqvae"].get("adv_weight", 0.1))
                gen_loss = (
                    outputs["recon_loss"]
                    + outputs["lpips_loss"]
                    + outputs["codebook_loss"]
                    + outputs["commitment_loss"]
                    + adv_loss
                )
            scaler.scale(gen_loss).backward()
            scaler.step(gen_opt)

            disc_opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                real_logits = disc(images)
                fake_logits = disc(outputs["recon"].detach())
                disc_loss = 0.5 * (_bce_loss(real_logits, True) + _bce_loss(fake_logits, False))
            scaler.scale(disc_loss).backward()
            scaler.step(disc_opt)
            scaler.update()

            progress.set_postfix(gen=float(gen_loss.item()), disc=float(disc_loss.item()))

        val_loss, sample_pair = _validate(model, val_loader, device)
        LOGGER.info("VQ-VAE epoch %s validation loss %.4f", epoch + 1, val_loss)
        if sample_pair.numel() > 0:
            save_image_grid(sample_pair, sample_dir / f"epoch_{epoch + 1:03d}.png", nrow=4)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpts["vqvae"],
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config["vqvae"],
                    "train_config": train_config,
                    "best_val": best_val,
                },
            )
            LOGGER.info("Saved best VQ-VAE checkpoint to %s", ckpts["vqvae"])
