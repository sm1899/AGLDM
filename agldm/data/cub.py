from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _read_id_map(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            idx_str, value = line.strip().split(" ", maxsplit=1)
            mapping[int(idx_str)] = value
    return mapping


def load_cub_image_index(raw_root: str | Path) -> list[dict[str, str | int]]:
    root = Path(raw_root)
    metadata_root = root / "CUB_200_2011"
    if not metadata_root.exists():
        metadata_root = root

    images = _read_id_map(metadata_root / "images.txt")
    classes = _read_id_map(metadata_root / "classes.txt")
    labels = {int(idx): int(cls) for idx, cls in _iter_numeric_pairs(metadata_root / "image_class_labels.txt")}

    entries: list[dict[str, str | int]] = []
    for image_id, rel_image in images.items():
        class_id = labels[image_id]
        entries.append(
            {
                "image_id": image_id,
                "rel_image": rel_image,
                "image_path": str((metadata_root / "images" / rel_image).resolve()),
                "class_id": class_id,
                "class_name": classes[class_id],
            }
        )
    return entries


def _iter_numeric_pairs(path: Path) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            left, right = line.strip().split()
            pairs.append((int(left), int(right)))
    return pairs


def load_xian_splits(path: str | Path) -> dict[str, np.ndarray]:
    payload = loadmat(Path(path))
    splits: dict[str, np.ndarray] = {}
    for key in ("train_loc", "trainval_loc", "val_loc", "test_seen_loc", "test_unseen_loc"):
        if key in payload:
            splits[key] = payload[key].squeeze().astype(np.int64)
    if "att" in payload:
        splits["att"] = payload["att"].astype(np.float32)
    elif "original_att" in payload:
        splits["att"] = payload["original_att"].astype(np.float32)
    else:
        msg = f"No attribute matrix found in {path}."
        raise KeyError(msg)
    return splits


def build_class_attribute_lookup(
    attribute_matrix: np.ndarray,
    *,
    threshold: float = 0.5,
    binarize: bool = True,
    transpose: bool = True,
) -> dict[int, list[float]]:
    if attribute_matrix.ndim != 2:
        msg = f"Expected a rank-2 attribute matrix, found shape {attribute_matrix.shape}."
        raise ValueError(msg)
    matrix = attribute_matrix.T if transpose else attribute_matrix

    if binarize:
        matrix = (matrix >= threshold).astype(np.float32)
    lookup: dict[int, list[float]] = {}
    for idx, row in enumerate(matrix, start=1):
        lookup[idx] = row.astype(np.float32).tolist()
    return lookup


def load_reed_captions(caption_root: str | Path) -> dict[str, list[str]]:
    root = Path(caption_root)
    if not root.exists():
        msg = f"Caption root does not exist: {root}"
        raise FileNotFoundError(msg)

    grouped: dict[str, list[str]] = defaultdict(list)
    for caption_file in root.rglob("*.txt"):
        rel_stem = caption_file.relative_to(root).with_suffix("").as_posix()
        lines = [line.strip() for line in caption_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        grouped[rel_stem].extend(lines)
        grouped[caption_file.stem].extend(lines)
    return grouped
