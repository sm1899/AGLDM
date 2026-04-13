from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from agldm.data.cub import build_class_attribute_lookup, load_cub_image_index, load_reed_captions, load_xian_splits
from agldm.data.types import SampleRecord
from agldm.utils.logging import get_logger, write_json

LOGGER = get_logger(__name__)


def _resolve_split_map(split_payload: dict[str, Any]) -> dict[int, str]:
    train_ids = set(split_payload.get("train_loc", split_payload["trainval_loc"]).tolist())
    val_ids = set(split_payload.get("val_loc", split_payload.get("test_seen_loc", [])).tolist())
    seen_test_ids = set(split_payload.get("test_seen_loc", []).tolist())
    unseen_ids = set(split_payload["test_unseen_loc"].tolist())

    split_map: dict[int, str] = {}
    for image_id in train_ids:
        split_map[int(image_id)] = "train_seen"
    for image_id in val_ids:
        split_map.setdefault(int(image_id), "val_seen")
    for image_id in seen_test_ids:
        split_map.setdefault(int(image_id), "test_seen")
    for image_id in unseen_ids:
        split_map[int(image_id)] = "test_unseen"
    return split_map


def _load_caption_lookup(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, list[str]]:
    caption_manifest = Path(config["caption_manifest"])
    if caption_manifest.exists():
        payload = json.loads(caption_manifest.read_text(encoding="utf-8"))
        return {key: value for key, value in payload.items()}

    caption_root = config.get("caption_root")
    if caption_root:
        return load_reed_captions(caption_root)

    if not config.get("allow_blip_fallback", True):
        return {}

    captions = _generate_blip_captions(records, config)
    caption_manifest.parent.mkdir(parents=True, exist_ok=True)
    caption_manifest.write_text(json.dumps(captions, indent=2), encoding="utf-8")
    return captions


def _generate_blip_captions(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, list[str]]:
    model_name = config.get("blip_model_name", "Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Generating fallback BLIP captions with %s on %s", model_name, device)

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    captions: dict[str, list[str]] = {}
    for record in records:
        image = Image.open(record["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=32)
        text = processor.decode(generated[0], skip_special_tokens=True).strip()
        key = Path(record["rel_image"]).with_suffix("").as_posix()
        captions[key] = [text]
    return captions


def _match_text(record: dict[str, Any], caption_lookup: dict[str, list[str]]) -> list[str]:
    rel_stem = Path(record["rel_image"]).with_suffix("").as_posix()
    keys = (
        rel_stem,
        Path(rel_stem).name,
        f"{record['class_name']}/{Path(rel_stem).stem}",
    )
    for key in keys:
        if key in caption_lookup and caption_lookup[key]:
            return caption_lookup[key]
    return [f"a photo of {record['class_name'].replace('.', ' ').replace('_', ' ').lower()}"]


def validate_zero_shot_splits(records: list[SampleRecord]) -> None:
    train_classes = {record.class_id for record in records if record.split == "train_seen"}
    unseen_classes = {record.class_id for record in records if record.split == "test_unseen"}
    overlap = train_classes & unseen_classes
    if overlap:
        msg = f"Leakage detected between train_seen and test_unseen: {sorted(overlap)}"
        raise ValueError(msg)


def prepare_cub_data(config: dict[str, Any], *, force: bool = False) -> None:
    manifest_path = Path(config["manifest_path"])
    stats_path = Path(config["stats_path"])
    if manifest_path.exists() and not force:
        LOGGER.info("Manifest already exists at %s. Use --force to rebuild.", manifest_path)
        return

    entries = load_cub_image_index(config["raw_root"])
    split_payload = load_xian_splits(config["xian_split_mat"])
    split_map = _resolve_split_map(split_payload)
    attribute_lookup = build_class_attribute_lookup(
        split_payload["att"],
        threshold=float(config.get("attribute_threshold", 0.5)),
        binarize=bool(config.get("binarize_attributes", True)),
        transpose=bool(config.get("attribute_matrix_transpose", True)),
    )
    caption_lookup = _load_caption_lookup(config, entries)

    records: list[SampleRecord] = []
    for entry in entries:
        if entry["image_id"] not in split_map:
            continue
        texts = _match_text(entry, caption_lookup)
        records.append(
            SampleRecord(
                image=str(entry["image_path"]),
                class_id=int(entry["class_id"]),
                class_name=str(entry["class_name"]),
                split=split_map[int(entry["image_id"])],
                attributes=attribute_lookup[int(entry["class_id"])],
                text=texts[0],
                texts=texts,
            )
        )

    validate_zero_shot_splits(records)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")

    split_counts = Counter(record.split for record in records)
    stats = {
        "num_records": len(records),
        "split_counts": dict(split_counts),
        "num_train_classes": len({record.class_id for record in records if record.split == "train_seen"}),
        "num_test_unseen_classes": len({record.class_id for record in records if record.split == "test_unseen"}),
        "attribute_dim": len(records[0].attributes) if records else 0,
    }
    write_json(stats_path, stats)
    LOGGER.info("Prepared %s records. Stats written to %s", len(records), stats_path)
