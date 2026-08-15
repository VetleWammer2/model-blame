"""Deterministic occurrence grouping policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    group_id: str
    occurrence_ids: tuple[str, ...]
    policy: str


def group_occurrences(
    rows: Iterable[Mapping[str, Any]], *, policy: str
) -> list[CandidateGroup]:
    buckets: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        occurrence_id = str(row["occurrence_id"])
        if policy == "occurrence":
            key = occurrence_id
        elif policy == "example":
            key = str(row["example_id"])
        elif policy == "source":
            key = str(row.get("source", ""))
        elif policy == "time_window":
            key = str(row.get("time_window", row.get("step", "")))
        elif policy.startswith("metadata:"):
            field = policy.partition(":")[2]
            metadata = row.get("metadata", {})
            if not isinstance(metadata, Mapping) or field not in metadata:
                raise ValueError(f"candidate row lacks metadata field {field!r}")
            key = str(metadata[field])
        else:
            raise ValueError(f"unsupported grouping policy: {policy!r}")
        buckets[key].append(occurrence_id)
    return [
        CandidateGroup(
            group_id=f"{policy}:{key}",
            occurrence_ids=tuple(sorted(set(occurrence_ids))),
            policy=policy,
        )
        for key, occurrence_ids in sorted(buckets.items())
    ]
