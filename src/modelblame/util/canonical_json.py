"""Canonical JSON encoding and content hashes.

The format intentionally accepts only the JSON data model.  It is not a
general-purpose object serializer and must never acquire pickle-like behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_ITEMS = 2_000_000


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented as safe canonical JSON."""


def _json_value(value: Any, *, _active: set[int] | None = None) -> Any:
    """Convert supported convenience types to the strict JSON data model."""

    if _active is None:
        _active = set()
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are not canonical JSON")
        # JSON has one numeric type.  Normalizing negative zero avoids two byte
        # encodings for values which training configurations treat as equal.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Enum):
        return _json_value(value.value, _active=_active)
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in _active:
            raise CanonicalizationError("cyclic mappings are not canonical JSON")
        _active.add(object_id)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise CanonicalizationError(
                        "canonical JSON object keys must be strings"
                    )
                if key in result:
                    raise CanonicalizationError(f"duplicate object key: {key!r}")
                result[key] = _json_value(item, _active=_active)
            return result
        finally:
            _active.remove(object_id)

    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray | memoryview
    ):
        object_id = id(value)
        if object_id in _active:
            raise CanonicalizationError("cyclic sequences are not canonical JSON")
        _active.add(object_id)
        try:
            return [_json_value(item, _active=_active) for item in value]
        finally:
            _active.remove(object_id)

    raise CanonicalizationError(
        f"unsupported canonical JSON value of type {type(value).__name__}"
    )


def validate_json_tree(
    value: Any,
    *,
    max_depth: int = MAX_JSON_DEPTH,
    max_items: int = MAX_JSON_ITEMS,
) -> None:
    """Reject cyclic, excessively nested, or excessively broad JSON-like data."""

    if max_depth < 0 or max_items < 1:
        raise ValueError("JSON bounds must be positive")
    active: set[int] = set()
    item_count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal item_count
        item_count += 1
        if item_count > max_items:
            raise CanonicalizationError("JSON value exceeds the item limit")
        if depth > max_depth:
            raise CanonicalizationError("JSON value exceeds the nesting limit")
        if item is None or isinstance(item, str | bool | int):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CanonicalizationError("non-finite floats are not allowed")
            return
        if isinstance(item, Mapping):
            object_id = id(item)
            if object_id in active:
                raise CanonicalizationError("cyclic JSON object")
            active.add(object_id)
            try:
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise CanonicalizationError("JSON object keys must be strings")
                    visit(child, depth + 1)
            finally:
                active.remove(object_id)
            return
        if isinstance(item, Sequence) and not isinstance(
            item, str | bytes | bytearray | memoryview
        ):
            object_id = id(item)
            if object_id in active:
                raise CanonicalizationError("cyclic JSON array")
            active.add(object_id)
            try:
                for child in item:
                    visit(child, depth + 1)
            finally:
                active.remove(object_id)
            return
        raise CanonicalizationError(f"value of type {type(item).__name__} is not JSON")

    visit(value, 0)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the deterministic UTF-8 JSON encoding used by artifact hashes."""

    normalized = _json_value(value)
    validate_json_tree(normalized)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:  # Defensive: normalization is strict.
        raise CanonicalizationError(str(error)) from error
    return text.encode("utf-8")


def canonical_json_text(value: Any) -> str:
    """Return :func:`canonical_json_bytes` decoded as UTF-8."""

    return canonical_json_bytes(value).decode("utf-8")


def content_hash(value: Any) -> str:
    """Return a lowercase SHA-256 digest of a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest of raw bytes."""

    return hashlib.sha256(value).hexdigest()


def hash_file(path: str | Path, *, max_bytes: int | None = None) -> str:
    """Hash a regular file without loading it into memory.

    ``max_bytes`` is a defensive read bound for untrusted artifacts.  Symlinks
    are accepted only when their resolved target is a regular file; callers
    that need root confinement should first use ``resolve_within_root``.
    """

    source = Path(path)
    stat = source.stat()
    if not source.is_file():
        raise ValueError(f"not a regular file: {source}")
    if max_bytes is not None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if stat.st_size > max_bytes:
            raise ValueError(f"file exceeds the {max_bytes}-byte limit: {source}")
    digest = hashlib.sha256()
    bytes_read = 0
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            bytes_read += len(block)
            if max_bytes is not None and bytes_read > max_bytes:
                raise ValueError(f"file exceeds the {max_bytes}-byte limit: {source}")
            digest.update(block)
    return digest.hexdigest()
