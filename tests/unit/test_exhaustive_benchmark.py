from __future__ import annotations

import pytest
from benchmarks.attribution import precision_recall_at_k, spearman_rank_correlation
from benchmarks.exhaustive_landscape import enumerate_subsets

from modelblame.reducer.engine import ReplayObservation


def test_exhaustive_landscape_finds_all_global_minima() -> None:
    def replay(subset: frozenset[str]) -> ReplayObservation:
        accepted = {"a", "b"} <= subset or {"c", "d"} <= subset
        return ReplayObservation(
            accepted,
            float(accepted),
            True,
            "TARGET_PASSED" if accepted else "TARGET_FAILED",
            "e" * 64,
        )

    landscape = enumerate_subsets(["a", "b", "c", "d"], replay)
    assert len(landscape.points) == 16
    assert landscape.global_minimum_size == 2
    assert landscape.globally_minimal_sets == (("a", "b"), ("c", "d"))


def test_attribution_metrics() -> None:
    assert precision_recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == (0.5, 0.5)
    assert spearman_rank_correlation(
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": 2.0, "b": 4.0, "c": 6.0},
    ) == pytest.approx(1.0)
