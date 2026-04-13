from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from agldm.data.datasets import build_dataloader
from agldm.data.types import SampleRecord
from agldm.models.classifier import AttributeClassifier
from agldm.models.vqvae import VQVAE
from agldm.utils.checkpointing import load_checkpoint


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def manifest_attribute_dim(manifest_path: str | Path) -> int:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    return len(first["attributes"])


def iter_manifest_records(manifest_path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(SampleRecord.from_dict(json.loads(line)))
    return records


def build_stage_loaders(
    data_config: dict[str, Any],
    train_config: dict[str, Any],
    *,
    train_split: str = "train_seen",
    val_split: str = "val_seen",
    random_text: bool = True,
) -> tuple[Any, Any]:
    manifest_path = data_config["manifest_path"]
    image_size = int(data_config.get("image_size", 256))
    num_workers = int(data_config.get("num_workers", 4))
    batch_size = int(train_config["batch_size"])
    eval_batch_size = int(train_config.get("eval_batch_size", batch_size))
    max_train_items = train_config.get("max_train_items")
    max_val_items = train_config.get("max_val_items")

    train_loader = build_dataloader(
        manifest_path,
        split=train_split,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        random_text=random_text,
        max_items=max_train_items,
    )
    val_loader = build_dataloader(
        manifest_path,
        split=val_split,
        image_size=image_size,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        shuffle=False,
        random_text=False,
        max_items=max_val_items,
    )
    return train_loader, val_loader


def checkpoint_paths(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "vqvae": root / "vqvae.ckpt",
        "classifier": root / "attr_classifier.ckpt",
        "ldm": root / "ldm.ckpt",
    }


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def build_vqvae_from_config(model_config: dict[str, Any]) -> VQVAE:
    cfg = model_config["vqvae"]
    return VQVAE(
        in_channels=int(cfg.get("in_channels", 3)),
        base_channels=int(cfg.get("base_channels", 64)),
        latent_channels=int(cfg.get("latent_channels", 256)),
        codebook_size=int(cfg.get("codebook_size", 1024)),
        commitment_beta=float(cfg.get("commitment_beta", 0.25)),
        lpips_weight=float(cfg.get("lpips_weight", 1.0)),
        adv_weight=float(cfg.get("adv_weight", 0.1)),
    )


def build_classifier_from_config(model_config: dict[str, Any], attribute_dim: int) -> AttributeClassifier:
    cfg = model_config["classifier"]
    return AttributeClassifier(
        num_attributes=attribute_dim,
        pretrained=bool(cfg.get("pretrained", True)),
    )


def load_vqvae_checkpoint(path: str | Path, model_config: dict[str, Any], device: torch.device) -> VQVAE:
    model = build_vqvae_from_config(model_config)
    payload = load_checkpoint(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return freeze_module(model)


def load_classifier_checkpoint(
    path: str | Path,
    model_config: dict[str, Any],
    attribute_dim: int,
    device: torch.device,
) -> AttributeClassifier:
    model = build_classifier_from_config(model_config, attribute_dim)
    payload = load_checkpoint(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return freeze_module(model)


def verify_ldm_dependencies(vqvae_path: str | Path, classifier_path: str | Path) -> None:
    missing = [str(path) for path in (Path(vqvae_path), Path(classifier_path)) if not Path(path).exists()]
    if missing:
        msg = f"train_ldm requires frozen checkpoints. Missing: {', '.join(missing)}"
        raise FileNotFoundError(msg)
