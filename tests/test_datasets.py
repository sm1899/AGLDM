from __future__ import annotations

from pathlib import Path

from agldm.data.datasets import ManifestDataset, collate_records
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_manifest_dataset_returns_record_schema() -> None:
    dataset = ManifestDataset(FIXTURES / "manifest_single.jsonl", split="train_seen", image_size=32, random_text=False)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["attributes"].shape[0] == 4
    assert sample["text"] == "a small bird"


def test_collate_records_preserves_text_lists() -> None:
    records = []
    for name in ("manifest_double_a.jsonl", "manifest_double_b.jsonl"):
        dataset = ManifestDataset(FIXTURES / name, split="train_seen", image_size=32, random_text=False)
        records.append(dataset[0])

    batch = collate_records(records)
    assert batch["image"].shape[0] == 2
    assert batch["texts"][0] == ["text 0", "alt 0"]
