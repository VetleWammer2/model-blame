"""Transparent candidate fusion that preserves individual method rankings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from modelblame.attribution.base import CandidateScore


def union_rankings(
    rankings: Mapping[str, Sequence[CandidateScore]],
) -> list[CandidateScore]:
    """Union rankings in method/name/rank order without discarding records."""

    return [
        item
        for method in sorted(rankings)
        for item in sorted(
            rankings[method], key=lambda score: (score.rank, score.occurrence_id)
        )
    ]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[CandidateScore]], *, constant: int = 60
) -> list[tuple[str, float, dict[str, int]]]:
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores: defaultdict[str, float] = defaultdict(float)
    ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for method in sorted(rankings):
        for item in rankings[method]:
            scores[item.occurrence_id] += 1.0 / (constant + item.rank)
            ranks[item.occurrence_id][method] = item.rank
    return sorted(
        (
            (occurrence_id, score, ranks[occurrence_id])
            for occurrence_id, score in scores.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )


def quota_fusion(
    rankings: Mapping[str, Sequence[CandidateScore]], quotas: Mapping[str, int]
) -> list[CandidateScore]:
    """Select declared per-method quotas while retaining each method record."""

    output: list[CandidateScore] = []
    for method in sorted(quotas):
        quota = quotas[method]
        if quota < 0:
            raise ValueError("selection quotas must be non-negative")
        if method not in rankings:
            raise ValueError(f"quota names unknown method {method!r}")
        output.extend(rankings[method][:quota])
    return output
