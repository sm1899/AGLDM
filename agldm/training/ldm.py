from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from agldm.models.diffusion import DDPMScheduler, LatentUNet
from agldm.models.text import FrozenCLIPTextEncoder
from agldm.training.common import (
    build_stage_loaders,
    checkpoint_paths,
    load_classifier_checkpoint,
    load_vqvae_checkpoint,
    manifest_attribute_dim,
    resolve_device,
    verify_ldm_dependencies,
)
from agldm.utils.checkpointing import save_checkpoint
from agldm.utils.ema import ExponentialMovingAverage
from agldm.utils.images import normalize_for_resnet, save_image_grid
from agldm.utils.logging import get_logger
from agldm.utils.seed import set_seed

LOGGER = get_logger(__name__)


def compute_self_consistency_loss(
    predicted_x0: torch.Tensor,
    attributes: torch.Tensor,
    vqvae,
    classifier,
) -> torch.Tensor:
    decoded = vqvae.decode(predicted_x0)
    logits = classifier(normalize_for_resnet(decoded))
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, attributes)


@torch.no_grad()
def _validate(
    model: LatentUNet,
    ema_model: LatentUNet,
    scheduler: DDPMScheduler,
    vqvae,
    classifier,
    text_encoder,
    loader,
    device: torch.device,
    *,
    use_classifier_guidance: bool,
    lambda_attr: float,
) -> tuple[float, torch.Tensor]:
    model.eval()
    ema_model.eval()
    total = 0.0
    count = 0
    samples: torch.Tensor | None = None
    for batch in loader:
        images = batch["image"].to(device)
        attrs = batch["attributes"].to(device)
        z0 = vqvae.encode_quantized(images)
        noise = torch.randn_like(z0)
        timesteps = scheduler.sample_timesteps(images.size(0), device)
        zt = scheduler.q_sample(z0, timesteps, noise)
        text_context = text_encoder.encode(batch["text"], device)["last_hidden_state"]
        pred_noise = ema_model(zt, timesteps, text_context=text_context, attributes=attrs)
        noise_loss = torch.nn.functional.mse_loss(pred_noise, noise)
        loss = noise_loss
        predicted_x0 = scheduler.predict_start_from_noise(zt, timesteps, pred_noise)
        if use_classifier_guidance:
            loss = loss + lambda_attr * compute_self_consistency_loss(predicted_x0, attrs, vqvae, classifier)
        total += loss.item() * images.size(0)
        count += images.size(0)
        if samples is None:
            decoded = vqvae.decode(predicted_x0[:4]).cpu()
            samples = torch.cat([images[:4].cpu(), decoded], dim=0)
    return total / max(count, 1), samples if samples is not None else torch.empty(0)


def train_ldm(config: dict) -> None:
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]["ldm"]
    output_root = config["train"]["output_root"]
    ckpts = checkpoint_paths(output_root)
    verify_ldm_dependencies(ckpts["vqvae"], ckpts["classifier"])

    set_seed(int(config["train"].get("seed", data_config.get("seed", 42))))
    device = resolve_device()
    train_loader, val_loader = build_stage_loaders(data_config, train_config, random_text=True)
    attribute_dim = int(data_config.get("attribute_dim", manifest_attribute_dim(data_config["manifest_path"])))

    vqvae = load_vqvae_checkpoint(ckpts["vqvae"], model_config, device)
    classifier = load_classifier_checkpoint(ckpts["classifier"], model_config, attribute_dim, device)
    text_encoder = FrozenCLIPTextEncoder(model_config["text_encoder"].get("model_name", "openai/clip-vit-base-patch32")).to(device)

    ldm_cfg = model_config["ldm"]
    conditioning_mode = train_config.get("conditioning_mode", ldm_cfg.get("conditioning_mode", "full"))
    use_classifier_guidance = bool(train_config.get("use_classifier_guidance", True))
    lambda_attr = float(train_config.get("lambda_attr", 1.0))

    model = LatentUNet(
        latent_channels=int(ldm_cfg.get("latent_channels", model_config["vqvae"].get("latent_channels", 256))),
        attribute_dim=attribute_dim,
        text_dim=text_encoder.hidden_size,
        base_channels=int(ldm_cfg.get("base_channels", 128)),
        num_heads=int(ldm_cfg.get("num_heads", 8)),
        conditioning_mode=conditioning_mode,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_steps=int(ldm_cfg.get("num_train_steps", 1000)),
        beta_start=float(ldm_cfg.get("beta_start", 1e-4)),
        beta_end=float(ldm_cfg.get("beta_end", 2e-2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    ema = ExponentialMovingAverage(model, decay=float(train_config.get("ema_decay", 0.9999))).to(str(device))

    best_val = float("inf")
    sample_dir = Path(output_root) / "samples" / "ldm"

    for epoch in range(int(train_config["epochs"])):
        model.train()
        progress = tqdm(train_loader, desc=f"ldm epoch {epoch + 1}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            attrs = batch["attributes"].to(device)

            with torch.no_grad():
                z0 = vqvae.encode_quantized(images)
                text_context = text_encoder.encode(batch["text"], device)["last_hidden_state"]

            noise = torch.randn_like(z0)
            timesteps = scheduler.sample_timesteps(images.size(0), device)
            zt = scheduler.q_sample(z0, timesteps, noise)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred_noise = model(zt, timesteps, text_context=text_context, attributes=attrs)
                noise_loss = torch.nn.functional.mse_loss(pred_noise, noise)
                loss = noise_loss
                if use_classifier_guidance:
                    predicted_x0 = scheduler.predict_start_from_noise(zt, timesteps, pred_noise)
                    self_loss = compute_self_consistency_loss(predicted_x0, attrs, vqvae, classifier)
                    loss = loss + lambda_attr * self_loss
                else:
                    self_loss = torch.tensor(0.0, device=device)

            scaler.scale(loss).backward()
            if float(train_config.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            progress.set_postfix(loss=float(loss.item()), self=float(self_loss.item()))

        val_loss, sample_pair = _validate(
            model,
            ema.shadow,
            scheduler,
            vqvae,
            classifier,
            text_encoder,
            val_loader,
            device,
            use_classifier_guidance=use_classifier_guidance,
            lambda_attr=lambda_attr,
        )
        LOGGER.info("LDM epoch %s validation loss %.4f", epoch + 1, val_loss)
        if sample_pair.numel() > 0:
            save_image_grid(sample_pair, sample_dir / f"epoch_{epoch + 1:03d}.png", nrow=4)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpts["ldm"],
                {
                    "state_dict": model.state_dict(),
                    "ema_state_dict": ema.shadow.state_dict(),
                    "model_config": ldm_cfg,
                    "train_config": train_config,
                    "best_val": best_val,
                    "conditioning_mode": conditioning_mode,
                    "use_classifier_guidance": use_classifier_guidance,
                },
            )
            LOGGER.info("Saved best LDM checkpoint to %s", ckpts["ldm"])
