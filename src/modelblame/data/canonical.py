"""Versioned canonicalization of logical prompt/completion examples."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from modelblame.util.canonical_json import (
    CanonicalizationError,
    canonical_json_bytes,
    validate_json_tree,
)

EXAMPLE_CANONICALIZATION_VERSION = 1


class InvalidExampleError(ValueError):
    """Raised when a dataset row cannot become a logical training example."""


def _normalize_semantic_value(value: Any, *, _active: set[int] | None = None) -> Any:
    if _active is None:
        _active = set()
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidExampleError(
                "semantic values cannot contain non-finite floats"
            )
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in _active:
            raise InvalidExampleError("semantic values cannot contain cycles")
        _active.add(object_id)
        try:
            result: dict[str, Any] = {}
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise InvalidExampleError("semantic mapping keys must be strings")
                key = unicodedata.normalize("NFC", raw_key)
                if key in result:
                    raise InvalidExampleError(
                        "Unicode normalization produced a duplicate mapping key"
                    )
                result[key] = _normalize_semantic_value(item, _active=_active)
            return result
        finally:
            _active.remove(object_id)
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray | memoryview
    ):
        object_id = id(value)
        if object_id in _active:
            raise InvalidExampleError("semantic values cannot contain cycles")
        _active.add(object_id)
        try:
            return [_normalize_semantic_value(item, _active=_active) for item in value]
        finally:
            _active.remove(object_id)
    raise InvalidExampleError(
        f"unsupported semantic value of type {type(value).__name__}"
    )


def _field_names(values: Iterable[str], *, role: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(item, str) or not item for item in result):
        raise InvalidExampleError(f"{role} field names must be non-empty strings")
    if len(result) != len(set(result)):
        raise InvalidExampleError(f"{role} field names must be unique")
    return result


def canonicalize_example(
    record: Mapping[str, Any],
    *,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    labels_fields: Iterable[str] = ("labels",),
    metadata_fields: Iterable[str] | None = None,
    sample_weight_field: str | None = "sample_weight",
) -> dict[str, Any]:
    """Select and normalize fields that can affect training semantics.

    Undeclared transport fields are intentionally ignored.  With
    ``metadata_fields=None``, a conventional nested ``metadata`` object is
    included; passing an explicit iterable selects those top-level row fields.
    Whitespace and line endings are preserved because they may affect tokens.
    """

    if not isinstance(record, Mapping):
        raise InvalidExampleError("an example must be a mapping")
    for field_name, role in (
        (prompt_field, "prompt"),
        (completion_field, "completion"),
    ):
        if not isinstance(field_name, str) or not field_name:
            raise InvalidExampleError(f"{role} field name must be non-empty")
        if field_name not in record:
            raise InvalidExampleError(f"missing required {role} field {field_name!r}")
        if not isinstance(record[field_name], str):
            raise InvalidExampleError(f"{role} must be a string")
    if not record[completion_field]:
        raise InvalidExampleError("completion must not be empty")

    label_names = _field_names(labels_fields, role="labels")
    if metadata_fields is None:
        metadata: Any = record.get("metadata", {})
    else:
        metadata_names = _field_names(metadata_fields, role="metadata")
        metadata = {name: record[name] for name in metadata_names if name in record}
    if not isinstance(metadata, Mapping):
        raise InvalidExampleError("metadata must be a mapping")
    if label_names == ("labels",):
        # Missing and explicit-null conventional labels both mean that the SFT
        # harness derives labels from the completion tokens.
        labels: Any = record.get("labels")
    else:
        labels = {name: record.get(name) for name in label_names}

    raw_weight: Any = 1.0
    if sample_weight_field is not None:
        if not isinstance(sample_weight_field, str) or not sample_weight_field:
            raise InvalidExampleError("sample-weight field name must be non-empty")
        raw_weight = record.get(sample_weight_field, 1.0)
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
        raise InvalidExampleError("sample weight must be a finite number")
    weight = float(raw_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1_000:
        raise InvalidExampleError("sample weight must be finite and in [0, 1000]")

    canonical = {
        "canonicalization_version": EXAMPLE_CANONICALIZATION_VERSION,
        "prompt": _normalize_semantic_value(record[prompt_field]),
        "completion": _normalize_semantic_value(record[completion_field]),
        "labels": _normalize_semantic_value(labels),
        "metadata": _normalize_semantic_value(metadata),
        "sample_weight": weight,
    }
    try:
        validate_json_tree(canonical)
    except CanonicalizationError as error:
        raise InvalidExampleError(str(error)) from error
    return canonical


def canonical_example_bytes(
    record: Mapping[str, Any],
    **field_options: Any,
) -> bytes:
    """Canonical bytes used as the input to a stable example ID."""

    return canonical_json_bytes(canonicalize_example(record, **field_options))
