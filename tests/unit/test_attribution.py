from __future__ import annotations

import torch

from modelblame.attribution.base import CandidateEvent, ranked_scores
from modelblame.attribution.bm25 import bm25_scores
from modelblame.attribution.fusion import reciprocal_rank_fusion
from modelblame.attribution.projection import project_vector
from modelblame.attribution.temporal import temporal_scores
from modelblame.attribution.tracin import tracin_cp_score
from modelblame.attribution.trajectory_sketch import (
    adam_diagonal_precondition,
    trajectory_sketch_score,
)


def events() -> list[CandidateEvent]:
    return [
        CandidateEvent("o1", "e1", 5, "a", "Veloria capital Nareth"),
        CandidateEvent("o2", "e2", 40, "b", "Nareth river"),
    ]


def test_temporal_and_bm25_baselines() -> None:
    assert (
        temporal_scores(events(), [(0, 10)])[0]
        > temporal_scores(events(), [(0, 10)])[1]
    )
    lexical = bm25_scores([event.text for event in events()], "Veloria capital")
    assert lexical[0] > lexical[1]


def test_projection_is_seeded() -> None:
    vector = torch.arange(12, dtype=torch.float64)
    assert torch.equal(
        project_vector(vector, dimension=5, seed=12),
        project_vector(vector, dimension=5, seed=12),
    )
    assert not torch.equal(
        project_vector(vector, dimension=5, seed=12),
        project_vector(vector, dimension=5, seed=13),
    )


def test_tracin_matches_direct_analytic_dot_product() -> None:
    training = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([2.0, 3.0])}
    behavior = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([4.0, 5.0])}
    expected = 0.1 * 11.0 + 0.2 * 23.0
    assert (
        tracin_cp_score(training, behavior, learning_rates={"a": 0.1, "b": 0.2})
        == expected
    )


def test_trajectory_precondition_and_score_without_projection_error() -> None:
    gradient = torch.tensor([2.0, 4.0], dtype=torch.float64)
    moment = torch.tensor([1.0, 4.0], dtype=torch.float64)
    preconditioned = adam_diagonal_precondition(
        gradient, moment, beta2=0.0, step=1, epsilon=1e-12
    )
    assert torch.allclose(preconditioned, torch.tensor([2.0, 2.0], dtype=torch.float64))
    score = trajectory_sketch_score(
        {"a": gradient},
        {"a": torch.tensor([1.0, 1.0], dtype=torch.float64)},
        {"a": moment},
        steps={"a": 1},
        projection_dimension=2,
        projection_seed=3,
        beta2=0.0,
        epsilon=1e-12,
    )
    assert isinstance(score, float)


def test_rrf_preserves_method_disagreement() -> None:
    left = ranked_scores(events(), [2.0, 1.0], method_id="left")
    right = ranked_scores(events(), [1.0, 2.0], method_id="right")
    fused = reciprocal_rank_fusion({"left": left, "right": right})
    assert len(fused) == 2
    assert fused[0][2].keys() == {"left", "right"}
