from __future__ import annotations

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from agldm.models.classifier import AttributeClassifier
from agldm.training.common import build_stage_loaders, checkpoint_paths, manifest_attribute_dim, resolve_device
from agldm.utils.checkpointing import save_checkpoint
from agldm.utils.images import normalize_for_resnet
from agldm.utils.logging import get_logger
from agldm.utils.seed import set_seed

LOGGER = get_logger(__name__)


@torch.no_grad()
def _evaluate(model: AttributeClassifier, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    for batch in loader:
        images = normalize_for_resnet(batch["image"].to(device))
        attrs = batch["attributes"].to(device)
        logits = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, attrs)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_loss += loss.item() * images.size(0)
        total_acc += (preds == attrs).float().mean().item() * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1), total_acc / max(total, 1)


def train_attribute_classifier(config: dict) -> None:
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]["classifier"]
    output_root = config["train"]["output_root"]

    set_seed(int(config["train"].get("seed", data_config.get("seed", 42))))
    device = resolve_device()
    train_loader, val_loader = build_stage_loaders(data_config, train_config, random_text=False)
    attribute_dim = int(data_config.get("attribute_dim", manifest_attribute_dim(data_config["manifest_path"])))

    model = AttributeClassifier(attribute_dim, pretrained=bool(model_config["classifier"].get("pretrained", True))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["lr"]), weight_decay=float(train_config.get("weight_decay", 1e-4)))
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    ckpts = checkpoint_paths(output_root)

    for epoch in range(int(train_config["epochs"])):
        model.train()
        progress = tqdm(train_loader, desc=f"classifier epoch {epoch + 1}", leave=False)
        for batch in progress:
            images = normalize_for_resnet(batch["image"].to(device))
            attrs = batch["attributes"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, attrs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=float(loss.item()))

        val_loss, val_acc = _evaluate(model, val_loader, device)
        LOGGER.info("Classifier epoch %s validation loss %.4f accuracy %.4f", epoch + 1, val_loss, val_acc)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpts["classifier"],
                {
                    "state_dict": model.state_dict(),
                    "attribute_dim": attribute_dim,
                    "model_config": model_config["classifier"],
                    "train_config": train_config,
                    "best_val": best_val,
                    "best_val_accuracy": val_acc,
                },
            )
            LOGGER.info("Saved best classifier checkpoint to %s", ckpts["classifier"])
