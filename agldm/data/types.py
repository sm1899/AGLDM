from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SampleRecord:
    image: str
    class_id: int
    class_name: str
    split: str
    attributes: list[float]
    text: str
    texts: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SampleRecord":
        return cls(**payload)
