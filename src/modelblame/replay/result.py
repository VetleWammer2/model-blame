"""Replay state and evidence-grade vocabulary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ReplayGrade(StrEnum):
    BITWISE = "BITWISE"
    NUMERIC = "NUMERIC"
    STATISTICAL = "STATISTICAL"
    FAILED = "FAILED"
    UNAUDITED = "UNAUDITED"


class ReplayResultState(StrEnum):
    COMPLETED = "COMPLETED"
    REPLAY_DIVERGED = "REPLAY_DIVERGED"
    TARGET_PASSED = "TARGET_PASSED"
    TARGET_FAILED = "TARGET_FAILED"
    CONTROLS_FAILED = "CONTROLS_FAILED"
    ENVIRONMENT_FAILED = "ENVIRONMENT_FAILED"
    TIMEOUT = "TIMEOUT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class ComponentComparison:
    component: str
    bitwise_equal: bool
    numeric_equal: bool
    max_absolute_difference: float | None = None
    max_relative_difference: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayAuditResult:
    schema_version: int
    run_id: str
    from_step: int
    to_step: int
    source_checkpoint_hash: str
    target_checkpoint_hash: str
    replay_grade: ReplayGrade
    state: ReplayResultState
    components: tuple[ComponentComparison, ...]
    replayed_event_hashes: tuple[str, ...]
    recorded_losses_equal: bool
    output_hashes_equal: bool
    environment_scope: Mapping[str, Any]
    behavior_scores: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["replay_grade"] = self.replay_grade.value
        value["state"] = self.state.value
        return value
