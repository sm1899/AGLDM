from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_parent(path)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
