from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from agldm.data.types import SampleRecord


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        image_size: int,
        random_text: bool = True,
        max_items: int | None = None,
    ) -> None:
        self.records = self._load_records(manifest_path, split)
        if max_items is not None:
            self.records = self.records[:max_items]
        self.random_text = random_text
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def _load_records(manifest_path: str | Path, split: str) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload["split"] == split:
                    records.append(SampleRecord.from_dict(payload))
        if not records:
            msg = f"No records found for split '{split}' in {manifest_path}."
            raise ValueError(msg)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record.image).convert("RGB")
        text = record.text
        if self.random_text and record.texts:
            text = random.choice(record.texts)
        return {
            "image": self.transform(image),
            "class_id": record.class_id,
            "class_name": record.class_name,
            "split": record.split,
            "attributes": torch.tensor(record.attributes, dtype=torch.float32),
            "text": text,
            "texts": record.texts or [record.text],
            "image_path": record.image,
        }


def collate_records(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "attributes": torch.stack([item["attributes"] for item in batch], dim=0),
        "class_id": torch.tensor([item["class_id"] for item in batch], dtype=torch.long),
        "class_name": [item["class_name"] for item in batch],
        "split": [item["split"] for item in batch],
        "text": [item["text"] for item in batch],
        "texts": [item["texts"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def build_dataloader(
    manifest_path: str | Path,
    *,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    random_text: bool,
    max_items: int | None = None,
) -> DataLoader[dict[str, Any]]:
    dataset = ManifestDataset(
        manifest_path,
        split=split,
        image_size=image_size,
        random_text=random_text,
        max_items=max_items,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_records,
    )
