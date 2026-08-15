"""Exact subset enumeration for the Causal Origin Benchmark.

This module never substitutes attribution for replay: callers provide a replay
function that executes the declared intervention for every subset.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

from modelblame.reducer.engine import ReplayObservation


@dataclass(frozen=True, slots=True)
class LandscapePoint:
    groups: tuple[str, ...]
    accepted: bool
    target_effect: float
    controls_passed: bool
    experiment_hash: str


@dataclass(frozen=True, slots=True)
class ExhaustiveLandscape:
    candidate_groups: tuple[str, ...]
    points: tuple[LandscapePoint, ...]
    global_minimum_size: int | None
    globally_minimal_sets: tuple[tuple[str, ...], ...]

    def write(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def enumerate_subsets(
    groups: Iterable[str],
    replay: Callable[[frozenset[str]], ReplayObservation],
) -> ExhaustiveLandscape:
    ordered = tuple(sorted(set(groups)))
    if not ordered or len(ordered) > 12:
        raise ValueError(
            "exhaustive benchmark requires between 1 and 12 candidate groups"
        )
    points: list[LandscapePoint] = []
    for size in range(len(ordered) + 1):
        for combination in combinations(ordered, size):
            observation = replay(frozenset(combination))
            points.append(
                LandscapePoint(
                    groups=combination,
                    accepted=observation.accepted,
                    target_effect=observation.target_effect,
                    controls_passed=observation.controls_passed,
                    experiment_hash=observation.experiment_hash,
                )
            )
    accepted = [point for point in points if point.accepted]
    minimum = min((len(point.groups) for point in accepted), default=None)
    globally_minimal = tuple(
        point.groups for point in accepted if len(point.groups) == minimum
    )
    return ExhaustiveLandscape(
        candidate_groups=ordered,
        points=tuple(points),
        global_minimum_size=minimum,
        globally_minimal_sets=globally_minimal,
    )
