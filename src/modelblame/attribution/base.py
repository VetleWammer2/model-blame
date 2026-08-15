"""Shared candidate-attribution records and index contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    occurrence_id: str
    example_id: str
    step: int
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.occurrence_id or not self.example_id:
            raise ValueError("candidate events require occurrence_id and example_id")
        if self.step < 0:
            raise ValueError("candidate step must be non-negative")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    occurrence_id: str
    example_id: str
    step: int
    source: str
    method_id: str
    raw_score: float
    rank: int
    method_config: dict[str, Any] = field(default_factory=dict)
    projection_seed: int | None = None
    duplicate_groups: tuple[str, ...] = ()
    cluster_memberships: tuple[str, ...] = ()
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("candidate rank is one-based")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AttributionIndex:
    method_id: str
    run_hash: str
    behavior_contract_hash: str
    checkpoint_hashes: tuple[str, ...]
    method_config: dict[str, Any]
    projection_seed: int | None
    scores: tuple[CandidateScore, ...]

    @property
    def content_hash(self) -> str:
        return _hash_json(
            {
                "method_id": self.method_id,
                "run_hash": self.run_hash,
                "behavior_contract_hash": self.behavior_contract_hash,
                "checkpoint_hashes": self.checkpoint_hashes,
                "method_config": self.method_config,
                "projection_seed": self.projection_seed,
                "scores": [score.to_dict() for score in self.scores],
            }
        )

    def write(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "content_hash": self.content_hash,
            "method_id": self.method_id,
            "run_hash": self.run_hash,
            "behavior_contract_hash": self.behavior_contract_hash,
            "checkpoint_hashes": list(self.checkpoint_hashes),
            "method_config": self.method_config,
            "projection_seed": self.projection_seed,
            "scores": [score.to_dict() for score in self.scores],
        }
        destination.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class CandidateGenerator(Protocol):
    method_id: str

    def build_index(
        self,
        events: Sequence[CandidateEvent],
        *,
        run_hash: str,
        behavior_contract_hash: str,
        checkpoint_hashes: Sequence[str],
        config: dict[str, Any],
    ) -> AttributionIndex: ...

    def rank(self, index: AttributionIndex, *, limit: int) -> list[CandidateScore]: ...


def ranked_scores(
    events: Sequence[CandidateEvent],
    raw_scores: Sequence[float],
    *,
    method_id: str,
    method_config: dict[str, Any] | None = None,
    projection_seed: int | None = None,
) -> tuple[CandidateScore, ...]:
    """Build a stable descending ranking, breaking ties by occurrence ID."""

    if len(events) != len(raw_scores):
        raise ValueError("events and scores must have equal length")
    rows = sorted(
        zip(events, raw_scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].occurrence_id),
    )
    return tuple(
        CandidateScore(
            occurrence_id=event.occurrence_id,
            example_id=event.example_id,
            step=event.step,
            source=event.source,
            method_id=method_id,
            raw_score=float(score),
            rank=rank,
            method_config=dict(method_config or {}),
            projection_seed=projection_seed,
            duplicate_groups=(event.example_id,),
        )
        for rank, (event, score) in enumerate(rows, start=1)
    )
