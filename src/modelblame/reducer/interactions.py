"""Observed interaction diagnostics; these are scoped, not mechanistic proofs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InteractionSignal:
    kind: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    union: tuple[str, ...]
    detail: str


def detect_non_monotonicity(
    observations: Mapping[frozenset[str], bool],
) -> list[InteractionSignal]:
    """Find direct antagonism and synergy in actually evaluated subsets."""

    subsets = sorted(observations, key=lambda item: (len(item), tuple(sorted(item))))
    signals: list[InteractionSignal] = []
    for index, left in enumerate(subsets):
        for right in subsets[index + 1 :]:
            if left & right:
                continue
            union = left | right
            if union not in observations:
                continue
            left_pass = observations[left]
            right_pass = observations[right]
            union_pass = observations[union]
            if left_pass and right_pass and not union_pass:
                signals.append(
                    InteractionSignal(
                        kind="ANTAGONISTIC",
                        left=tuple(sorted(left)),
                        right=tuple(sorted(right)),
                        union=tuple(sorted(union)),
                        detail="both disjoint patches pass but their union fails",
                    )
                )
            elif not left_pass and not right_pass and union_pass:
                signals.append(
                    InteractionSignal(
                        kind="SYNERGISTIC",
                        left=tuple(sorted(left)),
                        right=tuple(sorted(right)),
                        union=tuple(sorted(union)),
                        detail="neither disjoint patch passes but their union does",
                    )
                )
    return signals


def pairwise_effect_diagnostics(
    elements: tuple[str, ...],
    effects: Mapping[frozenset[str], float],
) -> list[dict[str, object]]:
    """Compute pair interaction residuals where singleton effects were measured."""

    diagnostics: list[dict[str, object]] = []
    for left_index, left in enumerate(elements):
        for right in elements[left_index + 1 :]:
            left_key = frozenset({left})
            right_key = frozenset({right})
            pair_key = frozenset({left, right})
            if (
                left_key not in effects
                or right_key not in effects
                or pair_key not in effects
            ):
                continue
            residual = effects[pair_key] - effects[left_key] - effects[right_key]
            diagnostics.append(
                {
                    "left": left,
                    "right": right,
                    "interaction_residual": residual,
                    "kind": "synergistic"
                    if residual > 0
                    else "antagonistic"
                    if residual < 0
                    else "additive",
                }
            )
    return diagnostics
