"""Validated prompt/completion datasets with stable logical example identity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import pyarrow as pa
import pyarrow.parquet as pq


def _canonical_bytes(value: Any) -> bytes:
    try:
        from modelblame.util.canonical_json import canonical_json_bytes

        return canonical_json_bytes(value)
    except ImportError:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _example_id(record: Mapping[str, Any]) -> str:
    try:
        from modelblame.data.identity import compute_example_id

        return compute_example_id(record)
    except ImportError:
        return hashlib.sha256(_canonical_bytes(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexedExample:
    example_id: str
    prompt: str
    completion: str
    labels: Any
    metadata: Mapping[str, Any]
    sample_weight: float
    source: str
    source_row: int
    reserved_noop: bool = False

    @property
    def logical_record(self) -> dict[str, Any]:
        value = {
            "prompt": self.prompt,
            "completion": self.completion,
            "labels": self.labels,
            "metadata": dict(self.metadata),
            "sample_weight": self.sample_weight,
        }
        if self.reserved_noop:
            value["reserved_noop"] = True
        return value


class IndexedDataset(Sequence[IndexedExample]):
    schema_version = 1

    def __init__(
        self,
        examples: Iterable[IndexedExample],
        *,
        source_path: Path | None = None,
        source_hash: str | None = None,
    ) -> None:
        self._examples = tuple(examples)
        if not self._examples:
            raise ValueError("training dataset is empty")
        self.source_path = source_path
        self.source_hash = source_hash
        for expected_row, example in enumerate(self._examples):
            if example.source_row < 0 or not example.example_id:
                raise ValueError("invalid indexed example")
            if example.sample_weight < 0 or example.sample_weight > 1_000:
                raise ValueError("sample_weight must be in [0,1000]")
            if expected_row != example.source_row:
                raise ValueError("source rows must be contiguous and ordered")

    def __len__(self) -> int:
        return len(self._examples)

    @overload
    def __getitem__(self, index: int) -> IndexedExample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[IndexedExample, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> IndexedExample | tuple[IndexedExample, ...]:
        return self._examples[index]

    def __iter__(self) -> Iterator[IndexedExample]:
        return iter(self._examples)

    @property
    def semantic_fingerprint(self) -> str:
        content = {
            "schema_version": self.schema_version,
            "examples": [example.logical_record for example in self._examples],
        }
        return hashlib.sha256(_canonical_bytes(content)).hexdigest()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(
                {
                    "schema_version": self.schema_version,
                    "semantic_fingerprint": self.semantic_fingerprint,
                    "source_hash": self.source_hash,
                }
            )
        ).hexdigest()

    @property
    def duplicate_groups(self) -> dict[str, tuple[int, ...]]:
        groups: dict[str, list[int]] = {}
        for example in self._examples:
            groups.setdefault(example.example_id, []).append(example.source_row)
        return {key: tuple(rows) for key, rows in groups.items() if len(rows) > 1}

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        source: str = "memory",
        prompt_field: str = "prompt",
        completion_field: str = "completion",
        labels_fields: tuple[str, ...] | None = None,
        metadata_fields: tuple[str, ...] | None = None,
        sample_weight_field: str | None = "sample_weight",
        reserved_noop_field: str | None = None,
    ) -> IndexedDataset:
        examples: list[IndexedExample] = []
        for row_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"record {row_index} must be an object")
            prompt = record.get(prompt_field)
            completion = record.get(completion_field)
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(
                    f"record {row_index} requires string prompt/completion"
                )
            if not completion:
                raise ValueError(f"record {row_index} completion must not be empty")
            labels = (
                record.get("labels")
                if labels_fields is None
                else {name: record[name] for name in labels_fields if name in record}
            )
            metadata = (
                record.get("metadata", {})
                if metadata_fields is None
                else {name: record[name] for name in metadata_fields if name in record}
            )
            if not isinstance(metadata, Mapping):
                raise ValueError(f"record {row_index} metadata must be an object")
            weight = float(
                record.get(sample_weight_field, 1.0)
                if sample_weight_field is not None
                else 1.0
            )
            if not math.isfinite(weight):
                raise ValueError(f"record {row_index} sample weight must be finite")
            reserved_noop = False
            if reserved_noop_field is not None:
                raw_reserved = record.get(reserved_noop_field, False)
                if not isinstance(raw_reserved, bool):
                    raise ValueError(
                        f"record {row_index} reserved no-op flag must be boolean"
                    )
                reserved_noop = raw_reserved
            logical_record = {
                "prompt": prompt,
                "completion": completion,
                "labels": labels,
                "metadata": dict(metadata),
                "sample_weight": weight,
                "reserved_noop": reserved_noop,
            }
            examples.append(
                IndexedExample(
                    example_id=_example_id(logical_record),
                    prompt=prompt,
                    completion=completion,
                    labels=labels,
                    metadata=dict(metadata),
                    sample_weight=weight,
                    source=source,
                    source_row=row_index,
                    reserved_noop=reserved_noop,
                )
            )
        return cls(examples)

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        max_file_bytes: int = 2**31,
        source: str | None = None,
        prompt_field: str = "prompt",
        completion_field: str = "completion",
        labels_fields: tuple[str, ...] | None = None,
        metadata_fields: tuple[str, ...] | None = None,
        sample_weight_field: str | None = "sample_weight",
        reserved_noop_field: str | None = None,
    ) -> IndexedDataset:
        source_path = path.resolve(strict=True)
        size = source_path.stat().st_size
        if size > max_file_bytes:
            raise ValueError("dataset exceeds configured size limit")
        source_hash = _file_hash(source_path)
        suffix = source_path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            records: list[Mapping[str, Any]] = []
            with source_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL line {line_number} is not an object")
                    records.append(value)
        elif suffix in {".parquet", ".pq"}:
            table = pq.read_table(source_path)
            records = table.to_pylist()
        else:
            raise ValueError("dataset must be JSONL or Parquet")
        indexed = cls.from_records(
            records,
            source=source or source_path.name,
            prompt_field=prompt_field,
            completion_field=completion_field,
            labels_fields=labels_fields,
            metadata_fields=metadata_fields,
            sample_weight_field=sample_weight_field,
            reserved_noop_field=reserved_noop_field,
        )
        return cls(indexed, source_path=source_path, source_hash=source_hash)

    def write_parquet(self, path: Path) -> None:
        rows = [
            {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "completion": example.completion,
                "labels_json": json.dumps(
                    example.labels,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "metadata_json": json.dumps(
                    dict(example.metadata),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "sample_weight": example.sample_weight,
                "reserved_noop": example.reserved_noop,
                "source": example.source,
                "source_row": example.source_row,
            }
            for example in self._examples
        ]
        schema = pa.schema(
            [
                ("example_id", pa.string()),
                ("prompt", pa.string()),
                ("completion", pa.string()),
                ("labels_json", pa.string()),
                ("metadata_json", pa.string()),
                ("sample_weight", pa.float64()),
                ("reserved_noop", pa.bool_()),
                ("source", pa.string()),
                ("source_row", pa.int64()),
            ]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), path, compression="zstd"
        )

    @classmethod
    def from_index_parquet(cls, path: Path) -> IndexedDataset:
        rows = pq.read_table(path).to_pylist()
        examples = []
        for row in rows:
            logical = {
                "prompt": row["prompt"],
                "completion": row["completion"],
                "labels": json.loads(row["labels_json"]),
                "metadata": json.loads(row["metadata_json"]),
                "sample_weight": float(row["sample_weight"]),
                "reserved_noop": bool(row.get("reserved_noop", False)),
            }
            if _example_id(logical) != row["example_id"]:
                raise ValueError("indexed dataset example hash mismatch")
            examples.append(
                IndexedExample(
                    example_id=row["example_id"],
                    prompt=row["prompt"],
                    completion=row["completion"],
                    labels=logical["labels"],
                    metadata=logical["metadata"],
                    sample_weight=logical["sample_weight"],
                    source=row["source"],
                    source_row=int(row["source_row"]),
                    reserved_noop=logical["reserved_noop"],
                )
            )
        return cls(examples)
