"""Streaming, validated Parquet ledger for exact training-event replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from modelblame.checkpoint.cursor import TrainingCursor
from modelblame.data.packing import PackedMicrobatch

LEDGER_SCHEMA_VERSION = 1
MAX_SEQUENCE_LENGTH = 1_048_576
MAX_BATCH_SIZE = 65_536


def _canonical_bytes(value: Any) -> bytes:
    try:
        from modelblame.util.canonical_json import canonical_json_bytes

        return canonical_json_bytes(value)
    except ImportError:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RecordedBatchEvent:
    schema_version: int
    global_step: int
    microbatch_index: int
    packed_sequence_ids: tuple[str, ...]
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[bool, ...], ...]
    loss_weights: tuple[tuple[float, ...], ...]
    padding_tokens: tuple[int, ...]
    occurrence_spans: tuple[tuple[Mapping[str, Any], ...], ...]
    original_loss_denominator: float
    cursor_after: Mapping[str, int]
    hyperparameters: Mapping[str, Any]
    recorded_loss: float | None = None
    output_hash: str | None = None
    event_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_SCHEMA_VERSION:
            raise ValueError("unsupported recorded batch schema")
        if self.global_step < 0 or self.microbatch_index < 0:
            raise ValueError("negative step or microbatch")
        cursor = TrainingCursor.from_mapping(self.cursor_after)
        if cursor.global_step != self.global_step:
            raise ValueError("event cursor global step differs from event")
        batch_size = len(self.input_ids)
        if not 0 < batch_size <= MAX_BATCH_SIZE:
            raise ValueError("invalid recorded batch size")
        if not (
            len(self.packed_sequence_ids)
            == len(self.attention_mask)
            == len(self.loss_weights)
            == len(self.padding_tokens)
            == len(self.occurrence_spans)
            == batch_size
        ):
            raise ValueError("recorded batch fields have different batch dimensions")
        seen: set[str] = set()
        denominator = 0.0
        for position, token_ids in enumerate(self.input_ids):
            if not 1 < len(token_ids) <= MAX_SEQUENCE_LENGTH:
                raise ValueError("invalid recorded sequence length")
            if not (
                len(self.attention_mask[position])
                == len(self.loss_weights[position])
                == len(token_ids)
            ):
                raise ValueError("recorded sequence fields have different lengths")
            if any(token < 0 or token >= 2**31 for token in token_ids):
                raise ValueError("recorded token ID is out of bounds")
            if any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in self.loss_weights[position]
            ):
                raise ValueError(
                    "recorded loss weights must be finite and non-negative"
                )
            if self.padding_tokens[position] != sum(
                not value for value in self.attention_mask[position]
            ):
                raise ValueError("padding metadata does not match attention mask")
            nonpadding = len(token_ids) - self.padding_tokens[position]
            expected_attention = tuple(
                index < nonpadding for index in range(len(token_ids))
            )
            if tuple(self.attention_mask[position]) != expected_attention:
                raise ValueError("padding must be a contiguous sequence suffix")
            denominator += sum(
                float(value) for value in self.loss_weights[position][1:]
            )
            previous_end = 0
            for span in self.occurrence_spans[position]:
                occurrence_id = str(span.get("occurrence_id", ""))
                if not occurrence_id or occurrence_id in seen:
                    raise ValueError("empty or duplicate occurrence ID inside event")
                seen.add(occurrence_id)
                start = int(span.get("token_start", -1))
                end = int(span.get("token_end", -1))
                if not 0 <= start < end <= len(token_ids):
                    raise ValueError("occurrence span is outside packed sequence")
                if start < previous_end:
                    raise ValueError("occurrence spans overlap or are out of order")
                if span.get("packed_sequence_id") != self.packed_sequence_ids[position]:
                    raise ValueError("occurrence references the wrong packed sequence")
                previous_end = end
                span_length = end - start
                if (
                    len(span.get("prompt_token_mask", ())) != span_length
                    or len(span.get("completion_token_mask", ())) != span_length
                ):
                    raise ValueError("occurrence token masks do not match its span")
                if any(
                    bool(prompt) and bool(completion)
                    for prompt, completion in zip(
                        span["prompt_token_mask"],
                        span["completion_token_mask"],
                        strict=True,
                    )
                ):
                    raise ValueError("prompt and completion masks overlap")
        if abs(denominator - self.original_loss_denominator) > 1e-5:
            raise ValueError("original loss denominator does not match token weights")
        if self.recorded_loss is not None and not math.isfinite(self.recorded_loss):
            raise ValueError("recorded loss must be finite")
        if self.event_hash and self.event_hash != self.calculate_hash():
            raise ValueError("recorded batch event hash mismatch")

    def hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "global_step": self.global_step,
            "microbatch_index": self.microbatch_index,
            "packed_sequence_ids": list(self.packed_sequence_ids),
            "input_ids": [list(row) for row in self.input_ids],
            "attention_mask": [list(row) for row in self.attention_mask],
            "loss_weights": [list(row) for row in self.loss_weights],
            "padding_tokens": list(self.padding_tokens),
            "occurrence_spans": [list(row) for row in self.occurrence_spans],
            "original_loss_denominator": self.original_loss_denominator,
            "cursor_after": dict(self.cursor_after),
            "hyperparameters": dict(self.hyperparameters),
        }

    def calculate_hash(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.hash_payload())).hexdigest()

    @classmethod
    def from_packed(
        cls,
        packed: PackedMicrobatch,
        *,
        recorded_loss: float | None = None,
        output_hash: str | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> RecordedBatchEvent:
        weights = tuple(sequence.loss_weights for sequence in packed.sequences)
        denominator = sum(sum(row[1:]) for row in weights)
        initial = cls(
            schema_version=LEDGER_SCHEMA_VERSION,
            global_step=packed.global_step,
            microbatch_index=packed.microbatch_index,
            packed_sequence_ids=tuple(
                sequence.packed_sequence_id for sequence in packed.sequences
            ),
            input_ids=tuple(sequence.input_ids for sequence in packed.sequences),
            attention_mask=tuple(
                sequence.attention_mask for sequence in packed.sequences
            ),
            loss_weights=weights,
            padding_tokens=tuple(
                sequence.padding_tokens for sequence in packed.sequences
            ),
            occurrence_spans=tuple(
                tuple(span.to_dict() for span in sequence.occurrences)
                for sequence in packed.sequences
            ),
            original_loss_denominator=float(denominator),
            cursor_after=dict(packed.cursor_after),
            hyperparameters=dict(hyperparameters or {}),
            recorded_loss=recorded_loss,
            output_hash=output_hash,
        )
        return cls(**{**asdict(initial), "event_hash": initial.calculate_hash()})

    def with_result(self, *, loss: float, output_hash: str) -> RecordedBatchEvent:
        return RecordedBatchEvent(
            **{
                **asdict(self),
                "recorded_loss": float(loss),
                "output_hash": output_hash,
                "event_hash": self.event_hash or self.calculate_hash(),
            }
        )


@dataclass(frozen=True, slots=True)
class StepRecord:
    global_step: int
    completed_step: int
    learning_rate: float
    mean_loss: float
    event_hashes: tuple[str, ...]
    model_hash: str


@dataclass(frozen=True, slots=True)
class MetricRecord:
    global_step: int
    name: str
    value: float


_BATCH_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("global_step", pa.int64()),
        ("microbatch_index", pa.int32()),
        ("packed_sequence_ids", pa.list_(pa.string())),
        ("input_ids", pa.list_(pa.list_(pa.int32()))),
        ("attention_mask", pa.list_(pa.list_(pa.bool_()))),
        ("loss_weights", pa.list_(pa.list_(pa.float64()))),
        ("padding_tokens", pa.list_(pa.int32())),
        ("occurrence_spans_json", pa.string()),
        ("original_loss_denominator", pa.float64()),
        ("cursor_after_json", pa.string()),
        ("hyperparameters_json", pa.string()),
        ("recorded_loss", pa.float64()),
        ("output_hash", pa.string()),
        ("event_hash", pa.string()),
    ]
)

_OCCURRENCE_SCHEMA = pa.schema(
    [
        ("occurrence_id", pa.string()),
        ("example_id", pa.string()),
        ("global_step", pa.int64()),
        ("microbatch_index", pa.int32()),
        ("batch_position", pa.int32()),
        ("packed_sequence_id", pa.string()),
        ("token_start", pa.int32()),
        ("token_end", pa.int32()),
        ("source", pa.string()),
        ("source_row", pa.int64()),
        ("epoch", pa.int64()),
        ("original_loss_weight", pa.float64()),
        ("prompt_token_mask", pa.list_(pa.bool_())),
        ("completion_token_mask", pa.list_(pa.bool_())),
        ("left_truncated_tokens", pa.int32()),
        ("right_truncated_tokens", pa.int32()),
    ]
)

_STEP_SCHEMA = pa.schema(
    [
        ("global_step", pa.int64()),
        ("completed_step", pa.int64()),
        ("learning_rate", pa.float64()),
        ("mean_loss", pa.float64()),
        ("event_hashes", pa.list_(pa.string())),
        ("model_hash", pa.string()),
    ]
)

_METRIC_SCHEMA = pa.schema(
    [
        ("global_step", pa.int64()),
        ("name", pa.string()),
        ("value", pa.float64()),
    ]
)


class LedgerWriter:
    """Append-only Parquet writer; final table names appear only on close."""

    def __init__(self, root: Path, *, row_group_size: int = 128) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.row_group_size = row_group_size
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        nonce = uuid.uuid4().hex
        self._paths = {
            name: root / f".{name}.{nonce}.inprogress"
            for name in ("steps", "batches", "occurrences", "metrics")
        }
        self._schemas = {
            "steps": _STEP_SCHEMA,
            "batches": _BATCH_SCHEMA,
            "occurrences": _OCCURRENCE_SCHEMA,
            "metrics": _METRIC_SCHEMA,
        }
        self._writers = {
            name: pq.ParquetWriter(path, self._schemas[name], compression="zstd")
            for name, path in self._paths.items()
        }
        self._buffers: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self._writers
        }
        self._counts = {name: 0 for name in self._writers}
        self._closed = False
        self._write_seconds = 0.0

    def _append(self, table: str, row: dict[str, Any]) -> None:
        started = time.perf_counter()
        if self._closed:
            raise RuntimeError("ledger writer is closed")
        try:
            self._buffers[table].append(row)
            if len(self._buffers[table]) >= self.row_group_size:
                self._flush(table)
        finally:
            self._write_seconds += time.perf_counter() - started

    def _flush(self, table: str) -> None:
        rows = self._buffers[table]
        if not rows:
            return
        arrow_table = pa.Table.from_pylist(rows, schema=self._schemas[table])
        self._writers[table].write_table(arrow_table)
        self._counts[table] += len(rows)
        rows.clear()

    def append_batch(self, event: RecordedBatchEvent) -> None:
        event.__post_init__()
        self._append(
            "batches",
            {
                "schema_version": event.schema_version,
                "global_step": event.global_step,
                "microbatch_index": event.microbatch_index,
                "packed_sequence_ids": list(event.packed_sequence_ids),
                "input_ids": [list(row) for row in event.input_ids],
                "attention_mask": [list(row) for row in event.attention_mask],
                "loss_weights": [list(row) for row in event.loss_weights],
                "padding_tokens": list(event.padding_tokens),
                "occurrence_spans_json": json.dumps(
                    event.occurrence_spans,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "original_loss_denominator": event.original_loss_denominator,
                "cursor_after_json": json.dumps(
                    event.cursor_after, sort_keys=True, separators=(",", ":")
                ),
                "hyperparameters_json": json.dumps(
                    event.hyperparameters, sort_keys=True, separators=(",", ":")
                ),
                "recorded_loss": event.recorded_loss,
                "output_hash": event.output_hash,
                "event_hash": event.event_hash or event.calculate_hash(),
            },
        )
        for spans in event.occurrence_spans:
            for span in spans:
                self._append("occurrences", dict(span))

    def append_step(self, record: StepRecord) -> None:
        self._append(
            "steps", {**asdict(record), "event_hashes": list(record.event_hashes)}
        )

    def append_metric(self, record: MetricRecord) -> None:
        self._append("metrics", asdict(record))

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("ledger writer already closed")
        started = time.perf_counter()
        try:
            for name in self._writers:
                self._flush(name)
                self._writers[name].close()
            for name, temporary in self._paths.items():
                destination = self.root / f"{name}.parquet"
                if destination.exists():
                    raise FileExistsError(f"ledger table already exists: {destination}")
                os.replace(temporary, destination)
            manifest = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "counts": dict(self._counts),
                "hashes": {
                    f"{name}.parquet": _hash_file(self.root / f"{name}.parquet")
                    for name in self._writers
                },
            }
            temporary_manifest = self.root / f".manifest.{uuid.uuid4().hex}.inprogress"
            temporary_manifest.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, self.root / "manifest.json")
            self._closed = True
            return manifest
        except Exception:
            for writer in self._writers.values():
                with suppress(Exception):
                    writer.close()
            raise
        finally:
            self._write_seconds += time.perf_counter() - started

    @property
    def write_seconds(self) -> float:
        """Measured time spent appending, flushing, and publishing the ledger."""

        return self._write_seconds

    def __enter__(self) -> LedgerWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            for writer in self._writers.values():
                writer.close()


@dataclass(frozen=True, slots=True)
class LedgerValidationReport:
    valid: bool
    issues: tuple[str, ...]
    counts: Mapping[str, int]


class LedgerReader:
    def __init__(self, root: Path, *, verify_hashes: bool = True) -> None:
        self.root = root
        self.manifest = self._read_manifest()
        if verify_hashes:
            for filename, expected in self.manifest["hashes"].items():
                path = root / filename
                if not path.is_file() or _hash_file(path) != expected:
                    raise ValueError(f"ledger hash mismatch: {filename}")

    def _read_manifest(self) -> Mapping[str, Any]:
        path = self.root / "manifest.json"
        if not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError("ledger manifest is missing or too large")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("invalid ledger manifest")
        if set(value.get("hashes", {})) != {
            "steps.parquet",
            "batches.parquet",
            "occurrences.parquet",
            "metrics.parquet",
        }:
            raise ValueError("ledger manifest table set is invalid")
        return value

    @staticmethod
    def _event_from_row(row: Mapping[str, Any]) -> RecordedBatchEvent:
        spans_value = json.loads(row["occurrence_spans_json"])
        return RecordedBatchEvent(
            schema_version=int(row["schema_version"]),
            global_step=int(row["global_step"]),
            microbatch_index=int(row["microbatch_index"]),
            packed_sequence_ids=tuple(row["packed_sequence_ids"]),
            input_ids=tuple(
                tuple(int(token) for token in item) for item in row["input_ids"]
            ),
            attention_mask=tuple(
                tuple(bool(token) for token in item) for item in row["attention_mask"]
            ),
            loss_weights=tuple(
                tuple(float(token) for token in item) for item in row["loss_weights"]
            ),
            padding_tokens=tuple(int(value) for value in row["padding_tokens"]),
            occurrence_spans=tuple(
                tuple(dict(span) for span in sequence) for sequence in spans_value
            ),
            original_loss_denominator=float(row["original_loss_denominator"]),
            cursor_after=json.loads(row["cursor_after_json"]),
            hyperparameters=json.loads(row["hyperparameters_json"]),
            recorded_loss=(
                None if row["recorded_loss"] is None else float(row["recorded_loss"])
            ),
            output_hash=row["output_hash"],
            event_hash=row["event_hash"],
        )

    def iter_batches(
        self, *, start_step: int = 0, end_step: int | None = None
    ) -> Iterator[RecordedBatchEvent]:
        previous: tuple[int, int] | None = None
        parquet = pq.ParquetFile(self.root / "batches.parquet")
        for record_batch in parquet.iter_batches(batch_size=128):
            for row in record_batch.to_pylist():
                step = int(row["global_step"])
                if step < start_step or (end_step is not None and step >= end_step):
                    continue
                event = self._event_from_row(row)
                key = (event.global_step, event.microbatch_index)
                if previous is not None and key <= previous:
                    raise ValueError("ledger batch events are not strictly ordered")
                previous = key
                yield event

    def iter_occurrences(
        self, *, start_step: int = 0, end_step: int | None = None
    ) -> Iterator[Mapping[str, Any]]:
        parquet = pq.ParquetFile(self.root / "occurrences.parquet")
        for record_batch in parquet.iter_batches(batch_size=1024):
            for row in record_batch.to_pylist():
                step = int(row["global_step"])
                if step >= start_step and (end_step is None or step < end_step):
                    yield row

    def validate(self) -> LedgerValidationReport:
        issues: list[str] = []
        counts = dict(self.manifest["counts"])
        try:
            actual_batches = 0
            actual_occurrences = 0
            expected_occurrence_digest = hashlib.sha256()
            for event in self.iter_batches():
                actual_batches += 1
                actual_occurrences += sum(
                    len(spans) for spans in event.occurrence_spans
                )
                for spans in event.occurrence_spans:
                    for span in spans:
                        expected_occurrence_digest.update(
                            str(span["occurrence_id"]).encode("ascii") + b"\n"
                        )
            occurrence_table_count = 0
            actual_occurrence_digest = hashlib.sha256()
            for row in self.iter_occurrences():
                occurrence_table_count += 1
                actual_occurrence_digest.update(
                    str(row["occurrence_id"]).encode("ascii") + b"\n"
                )
            if actual_batches != counts.get("batches"):
                issues.append("batch count differs from ledger manifest")
            if actual_occurrences != counts.get("occurrences"):
                issues.append("occurrence count differs from ledger manifest")
            if occurrence_table_count != actual_occurrences:
                issues.append("occurrence table count differs from batch spans")
            if actual_occurrence_digest.digest() != expected_occurrence_digest.digest():
                issues.append(
                    "occurrence table order or identity differs from batch spans"
                )
        except Exception as error:
            issues.append(str(error))
        return LedgerValidationReport(not issues, tuple(issues), counts)
