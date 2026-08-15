"""Bounded deterministic beam search for interaction-sensitive response surfaces."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BeamEvaluation:
    subset: frozenset[str]
    accepted: bool
    objective: float


def bounded_beam_search(
    items: Iterable[str],
    evaluate: Callable[[frozenset[str]], BeamEvaluation],
    *,
    beam_width: int,
    max_evaluations: int,
) -> tuple[BeamEvaluation | None, list[BeamEvaluation]]:
    """Search increasing subset sizes with stable objective-aware pruning."""

    ordered = tuple(sorted(set(items)))
    if not ordered:
        return None, []
    if beam_width < 1 or max_evaluations < 1:
        raise ValueError("beam width and evaluation budget must be positive")
    evaluated: list[BeamEvaluation] = []
    frontier = [frozenset({item}) for item in ordered]
    seen: set[frozenset[str]] = set()
    best: BeamEvaluation | None = None
    while frontier and len(evaluated) < max_evaluations:
        layer_results: list[BeamEvaluation] = []
        for subset in sorted(
            frontier, key=lambda value: (len(value), tuple(sorted(value)))
        ):
            if subset in seen or len(evaluated) >= max_evaluations:
                continue
            seen.add(subset)
            result = evaluate(subset)
            evaluated.append(result)
            layer_results.append(result)
            if result.accepted and (
                best is None
                or len(result.subset) < len(best.subset)
                or (
                    len(result.subset) == len(best.subset)
                    and tuple(sorted(result.subset)) < tuple(sorted(best.subset))
                )
            ):
                best = result
        if best is not None:
            break
        ranked = sorted(
            layer_results,
            key=lambda result: (-result.objective, tuple(sorted(result.subset))),
        )[:beam_width]
        next_frontier: set[frozenset[str]] = set()
        for result in ranked:
            last = max((ordered.index(item) for item in result.subset), default=-1)
            for item in ordered[last + 1 :]:
                next_frontier.add(result.subset | {item})
        frontier = list(next_frontier)
    return best, evaluated
