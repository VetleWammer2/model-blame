"""Detect every observed behavior transition instead of assuming monotonicity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise

from modelblame.behavior.evaluate import BehaviorState


@dataclass(frozen=True, slots=True)
class TimelinePoint:
    step: int
    score: float
    state: BehaviorState
    checkpoint_hash: str
    confidence_low: float | None = None
    confidence_high: float | None = None


@dataclass(frozen=True, slots=True)
class TransitionWindow:
    start_step: int
    end_step: int
    from_state: BehaviorState
    to_state: BehaviorState
    kind: str
    occurrence_ids: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    hyperparameter_changes: tuple[str, ...] = ()


def detect_transitions(
    points: Iterable[TimelinePoint],
    *,
    occurrences_by_interval: Mapping[tuple[int, int], Iterable[str]] | None = None,
    sources_by_interval: Mapping[tuple[int, int], Iterable[str]] | None = None,
) -> list[TransitionWindow]:
    """Return adjacent state changes, uncertainty spans, and unobserved gaps."""

    ordered = sorted(points, key=lambda point: point.step)
    if any(a.step == b.step for a, b in pairwise(ordered)):
        raise ValueError("timeline contains duplicate checkpoint steps")
    windows: list[TransitionWindow] = []
    for left, right in pairwise(ordered):
        key = (left.step, right.step)
        occurrences = (
            tuple(sorted(occurrences_by_interval.get(key, ())))
            if occurrences_by_interval
            else ()
        )
        sources = (
            tuple(sorted(set(sources_by_interval.get(key, ()))))
            if sources_by_interval
            else ()
        )
        if (
            left.state is BehaviorState.UNCERTAIN
            or right.state is BehaviorState.UNCERTAIN
        ):
            kind = "UNSTABLE"
        elif left.state == right.state:
            continue
        elif (
            left.state is BehaviorState.ABSENT and right.state is BehaviorState.PRESENT
        ):
            kind = "ABSENT_TO_PRESENT"
        else:
            kind = "PRESENT_TO_ABSENT"
        windows.append(
            TransitionWindow(
                start_step=left.step,
                end_step=right.step,
                from_state=left.state,
                to_state=right.state,
                kind=kind,
                occurrence_ids=occurrences,
                sources=sources,
            )
        )
    return windows


def validate_monotonicity(points: Iterable[TimelinePoint], direction: str) -> bool:
    """Validate a declared monotonic direction against every observed point."""

    states = [point.state for point in sorted(points, key=lambda point: point.step)]
    if BehaviorState.UNCERTAIN in states:
        return False
    encoded = [int(state is BehaviorState.PRESENT) for state in states]
    if direction == "absent_to_present":
        return all(a <= b for a, b in pairwise(encoded))
    if direction == "present_to_absent":
        return all(a >= b for a, b in pairwise(encoded))
    raise ValueError(
        "monotonic direction must be absent_to_present or present_to_absent"
    )
