from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Expected mapping at {path}, found {type(data).__name__}."
        raise TypeError(msg)
    return data


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_experiment_config(
    data_config: str | Path,
    model_config: str | Path | None = None,
    train_config: str | Path | None = None,
) -> dict[str, Any]:
    config = {"data": load_yaml(data_config)}
    if model_config is not None:
        config["model"] = load_yaml(model_config)
    if train_config is not None:
        config["train"] = load_yaml(train_config)
    return config


def resolve_path(path: str | Path | None, *, root: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if root is None:
        return candidate
    return Path(root) / candidate
