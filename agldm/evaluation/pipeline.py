from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.utils import save_image

from agldm.data.datasets import build_dataloader
from agldm.evaluation.metrics import ClipScorer, InceptionMetrics, compute_fid, compute_inception_score
from agldm.models.diffusion import DDPMScheduler, LatentUNet, ddim_step
from agldm.models.text import FrozenCLIPTextEncoder
from agldm.training.common import (
    checkpoint_paths,
    load_classifier_checkpoint,
    load_vqvae_checkpoint,
    manifest_attribute_dim,
    resolve_device,
)
from agldm.utils.checkpointing import load_checkpoint
from agldm.utils.images import normalize_for_resnet, save_image_grid, to_zero_one
from agldm.utils.logging import get_logger, write_json
from agldm.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _load_ldm_checkpoint(path: str | Path, model_config: dict[str, Any], attribute_dim: int, text_dim: int, device: torch.device) -> tuple[LatentUNet, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=device)
    merged_cfg = dict(model_config["ldm"])
    merged_cfg.update(payload.get("model_config", {}))
    model = LatentUNet(
        latent_channels=int(merged_cfg.get("latent_channels", model_config["vqvae"].get("latent_channels", 256))),
        attribute_dim=attribute_dim,
        text_dim=text_dim,
        base_channels=int(merged_cfg.get("base_channels", 128)),
        num_heads=int(merged_cfg.get("num_heads", 8)),
        conditioning_mode=str(payload.get("conditioning_mode", merged_cfg.get("conditioning_mode", "full"))),
    ).to(device)
    state = payload.get("ema_state_dict", payload["state_dict"])
    model.load_state_dict(state)
    model.eval()
    return model, payload


@torch.no_grad()
def sample_latents(
    model: LatentUNet,
    scheduler: DDPMScheduler,
    *,
    batch_size: int,
    latent_shape: tuple[int, int, int],
    text_context: torch.Tensor | None,
    attributes: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    sample = torch.randn((batch_size,) + latent_shape, device=device)
    timesteps = scheduler.ddim_timesteps(num_steps)
    for index, timestep in enumerate(timesteps):
        next_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        timestep_batch = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        pred_noise = model(sample, timestep_batch, text_context=text_context, attributes=attributes)
        state = ddim_step(scheduler, sample, pred_noise, timestep, next_timestep)
        sample = state.sample
    return sample


def _save_batch_images(images: torch.Tensor, batch: dict[str, Any], output_dir: Path, start_index: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for offset, image in enumerate(images):
        class_name = batch["class_name"][offset].replace("/", "_")
        file_name = f"{start_index + offset:05d}_{class_name}.png"
        path = output_dir / file_name
        save_image(to_zero_one(image), path)
        saved_paths.append(path)
    return saved_paths


def _resolve_eval_runs(train_config: dict[str, Any], default_ldm: Path) -> dict[str, Path]:
    eval_cfg = train_config.get("eval", {})
    ablation_checkpoints = eval_cfg.get("ablation_checkpoints")
    if ablation_checkpoints:
        return {name: Path(path) for name, path in ablation_checkpoints.items()}
    return {"default": default_ldm}


def sample_and_evaluate(config: dict[str, Any]) -> None:
    data_config = config["data"]
    model_config = config["model"]
    train_root = config["train"]["output_root"]
    eval_cfg = config["train"].get("eval", {})
    ckpts = checkpoint_paths(train_root)

    set_seed(int(config["train"].get("seed", data_config.get("seed", 42))))
    device = resolve_device()
    attribute_dim = int(data_config.get("attribute_dim", manifest_attribute_dim(data_config["manifest_path"])))
    batch_size = int(eval_cfg.get("batch_size", 8))
    num_workers = int(data_config.get("num_workers", 4))
    image_size = int(data_config.get("image_size", 256))
    ddim_steps = int(eval_cfg.get("ddim_steps", 50))
    max_items = eval_cfg.get("max_eval_items")

    test_loader = build_dataloader(
        data_config["manifest_path"],
        split="test_unseen",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        random_text=False,
        max_items=max_items,
    )

    vqvae = load_vqvae_checkpoint(ckpts["vqvae"], model_config, device)
    classifier = load_classifier_checkpoint(ckpts["classifier"], model_config, attribute_dim, device)
    text_encoder = FrozenCLIPTextEncoder(model_config["text_encoder"].get("model_name", "openai/clip-vit-base-patch32")).to(device)
    scheduler = DDPMScheduler(
        num_train_steps=int(model_config["ldm"].get("num_train_steps", 1000)),
        beta_start=float(model_config["ldm"].get("beta_start", 1e-4)),
        beta_end=float(model_config["ldm"].get("beta_end", 2e-2)),
    ).to(device)

    inception = InceptionMetrics().to(device)
    clip_scorer = ClipScorer(model_config["text_encoder"].get("model_name", "openai/clip-vit-base-patch32")).to(device)
    latent_channels = int(model_config["ldm"].get("latent_channels", model_config["vqvae"].get("latent_channels", 256)))
    latent_size = image_size // 8

    output_root = Path(train_root) / "eval"
    runs = _resolve_eval_runs(config["train"], ckpts["ldm"])

    for run_name, ldm_path in runs.items():
        LOGGER.info("Running evaluation for %s using checkpoint %s", run_name, ldm_path)
        model, payload = _load_ldm_checkpoint(ldm_path, model_config, attribute_dim, text_encoder.hidden_size, device)
        real_features: list[np.ndarray] = []
        fake_features: list[np.ndarray] = []
        fake_probs: list[np.ndarray] = []
        clip_scores: list[float] = []
        attr_scores: list[float] = []
        preview_batches: list[torch.Tensor] = []
        generated_dir = output_root / run_name / "generated"

        start_index = 0
        for batch in test_loader:
            real_images = batch["image"].to(device)
            attrs = batch["attributes"].to(device)
            text_context = text_encoder.encode(batch["text"], device)["last_hidden_state"]
            latents = sample_latents(
                model,
                scheduler,
                batch_size=real_images.size(0),
                latent_shape=(latent_channels, latent_size, latent_size),
                text_context=text_context,
                attributes=attrs,
                num_steps=ddim_steps,
                device=device,
            )
            fake_images = vqvae.decode(latents)
            _save_batch_images(fake_images.cpu(), batch, generated_dir, start_index)
            start_index += real_images.size(0)

            real_feat, _ = inception(real_images)
            fake_feat, fake_prob = inception(fake_images)
            real_features.append(real_feat.cpu().numpy())
            fake_features.append(fake_feat.cpu().numpy())
            fake_probs.append(fake_prob.cpu().numpy())

            batch_clip = clip_scorer.score(fake_images, batch["text"], device)
            clip_scores.extend(batch_clip.cpu().tolist())

            cls_logits = classifier(normalize_for_resnet(fake_images))
            cls_preds = (torch.sigmoid(cls_logits) >= 0.5).float()
            attr_scores.extend((cls_preds == attrs).float().mean(dim=1).cpu().tolist())

            if len(preview_batches) < 4:
                preview_batches.append(fake_images[:4].cpu())

        real_np = np.concatenate(real_features, axis=0)
        fake_np = np.concatenate(fake_features, axis=0)
        probs_np = np.concatenate(fake_probs, axis=0)
        metrics = {
            "fid": compute_fid(real_np, fake_np),
            "inception_score": compute_inception_score(probs_np),
            "clip_score": float(np.mean(clip_scores)),
            "attribute_consistency": float(np.mean(attr_scores)),
            "conditioning_mode": payload.get("conditioning_mode", "full"),
            "use_classifier_guidance": bool(payload.get("use_classifier_guidance", True)),
        }

        run_root = output_root / run_name
        write_json(run_root / "metrics.json", metrics)
        if preview_batches:
            grid = torch.cat(preview_batches, dim=0)
            save_image_grid(grid, run_root / "sample_grid.png", nrow=4)
        LOGGER.info("Completed %s evaluation: %s", run_name, metrics)
