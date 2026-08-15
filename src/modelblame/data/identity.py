"""Stable identities for logical examples and concrete training occurrences."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modelblame.data.canonical import canonical_example_bytes
from modelblame.util.canonical_json import canonical_json_bytes

IDENTITY_SCHEMA_VERSION = 1
EXAMPLE_ID_PATTERN = re.compile(r"^ex_[0-9a-f]{64}$")
OCCURRENCE_ID_PATTERN = re.compile(r"^occ_[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PackedTokenSpan:
    """Half-open token interval occupied by one occurrence."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise TypeError("token offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("packed token span must satisfy 0 <= start < end")

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


def compute_example_id(
    record: Mapping[str, Any],
    **field_options: Any,
) -> str:
    """Return ``ex_`` plus SHA-256 of the versioned logical record."""

    digest = hashlib.sha256(
        canonical_example_bytes(record, **field_options)
    ).hexdigest()
    return f"ex_{digest}"


def _span(value: PackedTokenSpan | Sequence[int]) -> PackedTokenSpan:
    if isinstance(value, PackedTokenSpan):
        return value
    if isinstance(value, str | bytes) or len(value) != 2:
        raise ValueError("packed_token_span must contain exactly start and end")
    start, end = value
    if isinstance(start, bool) or isinstance(end, bool):
        raise TypeError("token offsets must be integers")
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("token offsets must be integers")
    return PackedTokenSpan(start, end)


def compute_occurrence_id(
    *,
    run_id: str,
    global_step: int,
    microbatch_index: int,
    batch_position: int,
    packed_token_span: PackedTokenSpan | Sequence[int],
    example_id: str,
) -> str:
    """Identify one exact appearance of an example in a recorded microbatch."""

    if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
        raise ValueError("run_id must be a non-empty string of at most 256 characters")
    if any(character.isspace() for character in run_id):
        raise ValueError("run_id cannot contain whitespace")
    if not EXAMPLE_ID_PATTERN.fullmatch(example_id):
        raise ValueError("example_id is not a canonical ModelBlame example ID")
    coordinates = {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "run_id": run_id,
        "global_step": _nonnegative_int(global_step, "global_step"),
        "microbatch_index": _nonnegative_int(microbatch_index, "microbatch_index"),
        "batch_position": _nonnegative_int(batch_position, "batch_position"),
        "packed_token_span": _span(packed_token_span).as_dict(),
        "example_id": example_id,
    }
    digest = hashlib.sha256(canonical_json_bytes(coordinates)).hexdigest()
    return f"occ_{digest}"


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be in [0, 2^63 - 1]")
    return value


def is_example_id(value: str) -> bool:
    return bool(EXAMPLE_ID_PATTERN.fullmatch(value))


def is_occurrence_id(value: str) -> bool:
    return bool(OCCURRENCE_ID_PATTERN.fullmatch(value))


def group_logical_duplicates(
    records: Iterable[Mapping[str, Any]], **field_options: Any
) -> dict[str, tuple[int, ...]]:
    """Return example IDs mapped to every input row index, preserving duplicates."""

    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[compute_example_id(record, **field_options)].append(index)
    return {key: tuple(indices) for key, indices in sorted(groups.items())}


# Short names are convenient at ledger call sites while compute_* remain explicit.
example_id = compute_example_id
occurrence_id = compute_occurrence_id
