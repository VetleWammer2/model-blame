"""Temporal proximity baseline for transition windows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from modelblame.attribution.base import (
    AttributionIndex,
    CandidateEvent,
    CandidateScore,
    ranked_scores,
)


def temporal_scores(
    events: Sequence[CandidateEvent], windows: Sequence[tuple[int, int]]
) -> tuple[float, ...]:
    if not windows:
        raise ValueError("temporal attribution requires at least one transition window")
    normalized = [(min(a, b), max(a, b)) for a, b in windows]
    values: list[float] = []
    for event in events:
        distance = min(
            0
            if start <= event.step < end
            else min(abs(event.step - start), abs(event.step - end))
            for start, end in normalized
        )
        values.append(1.0 / (1.0 + distance))
    return tuple(values)


class TemporalGenerator:
    method_id = "temporal"

    def build_index(
        self,
        events: Sequence[CandidateEvent],
        *,
        run_hash: str,
        behavior_contract_hash: str,
        checkpoint_hashes: Sequence[str],
        config: dict[str, Any],
    ) -> AttributionIndex:
        raw_windows = config.get("windows")
        if not isinstance(raw_windows, list):
            raise ValueError("temporal config requires windows")
        windows = [(int(item[0]), int(item[1])) for item in raw_windows]
        scores = ranked_scores(
            events,
            temporal_scores(events, windows),
            method_id=self.method_id,
            method_config=config,
        )
        return AttributionIndex(
            method_id=self.method_id,
            run_hash=run_hash,
            behavior_contract_hash=behavior_contract_hash,
            checkpoint_hashes=tuple(checkpoint_hashes),
            method_config=config,
            projection_seed=None,
            scores=scores,
        )

    def rank(self, index: AttributionIndex, *, limit: int) -> list[CandidateScore]:
        return list(index.scores[:limit])
