"""Validated deterministic data/packing cursor state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingCursor:
    schema_version: int = 1
    global_step: int = 0
    epoch: int = 0
    source_shard: int = 0
    source_row: int = 0
    sampler_offset: int = 0
    microbatch: int = 0
    gradient_accumulation_position: int = 0
    packed_sequence_count: int = 0

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported cursor schema version")
        for name, value in asdict(self).items():
            if name == "schema_version":
                continue
            if not isinstance(value, int) or not 0 <= value < 2**63:
                raise ValueError(
                    f"cursor field {name} must be a bounded non-negative int"
                )

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingCursor:
        required = set(cls.__dataclass_fields__)
        unknown = set(value) - required
        if unknown:
            raise ValueError(f"unknown cursor fields: {sorted(unknown)}")
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing cursor fields: {sorted(missing)}")
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in value.values()
        ):
            raise ValueError("cursor fields must be integers")
        cursor = cls(**dict(value))
        cursor.validate()
        return cursor


def save_cursor(cursor: TrainingCursor, path: Path) -> None:
    path.write_text(
        json.dumps(cursor.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_cursor(path: Path, *, max_bytes: int = 64_000) -> TrainingCursor:
    if path.stat().st_size > max_bytes:
        raise ValueError("cursor metadata exceeds size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("cursor must be a JSON object")
    return TrainingCursor.from_mapping(value)
