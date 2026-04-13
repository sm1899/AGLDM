from __future__ import annotations

import numpy as np
import pytest

from agldm.data.cub import build_class_attribute_lookup
from agldm.data.prepare import validate_zero_shot_splits
from agldm.data.types import SampleRecord


def test_build_class_attribute_lookup_transposes_xian_matrix() -> None:
    matrix = np.array([[0.2, 0.8], [0.9, 0.1], [0.6, 0.7]], dtype=np.float32)
    lookup = build_class_attribute_lookup(matrix, threshold=0.5, binarize=True, transpose=True)
    assert lookup[1] == [0.0, 1.0, 1.0]
    assert lookup[2] == [1.0, 0.0, 1.0]


def test_validate_zero_shot_splits_detects_overlap() -> None:
    records = [
        SampleRecord("a.png", 1, "class_1", "train_seen", [1.0], "bird"),
        SampleRecord("b.png", 1, "class_1", "test_unseen", [1.0], "bird"),
    ]
    with pytest.raises(ValueError, match="Leakage detected"):
        validate_zero_shot_splits(records)
