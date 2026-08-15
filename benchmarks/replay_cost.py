"""Measured replay-cost summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast


@dataclass(frozen=True, slots=True)
class ReplayCost:
    experiments: int
    cache_hits: int
    steps_replayed: int
    wall_seconds: float
    compute_seconds: float
    prefix_reuse: int
    final_patch_size: int


def summarize(
    rows: Iterable[Mapping[str, object]], *, final_patch_size: int
) -> ReplayCost:
    values = list(rows)

    def as_int(value: object) -> int:
        return int(cast(str | bytes | bytearray | SupportsIndex, value))

    def as_float(value: object) -> float:
        return float(cast(str | SupportsFloat, value))

    return ReplayCost(
        experiments=len(values),
        cache_hits=sum(bool(row.get("cache_hit", False)) for row in values),
        steps_replayed=sum(as_int(row.get("steps_replayed", 0)) for row in values),
        wall_seconds=sum(as_float(row.get("wall_seconds", 0.0)) for row in values),
        compute_seconds=sum(
            as_float(row.get("compute_seconds", 0.0)) for row in values
        ),
        prefix_reuse=sum(bool(row.get("prefix_reuse", False)) for row in values),
        final_patch_size=final_patch_size,
    )
