from __future__ import annotations

from modelblame.reducer.ddmin import ddmin
from modelblame.reducer.engine import CausalReducer, ReplayObservation
from modelblame.reducer.interactions import detect_non_monotonicity
from modelblame.reducer.minimality import MinimalityGrade


def test_ddmin_finds_one_minimal_set() -> None:
    result = ddmin({"a", "b", "c", "d"}, lambda subset: {"b", "d"} <= subset)
    assert result == {"b", "d"}


def test_interaction_detection_finds_synergy_and_antagonism() -> None:
    observations = {
        frozenset({"a"}): False,
        frozenset({"b"}): False,
        frozenset({"a", "b"}): True,
        frozenset({"c"}): True,
        frozenset({"d"}): True,
        frozenset({"c", "d"}): False,
    }
    assert {signal.kind for signal in detect_non_monotonicity(observations)} == {
        "SYNERGISTIC",
        "ANTAGONISTIC",
    }


def test_reducer_executes_replays_and_certifies_one_minimality() -> None:
    def replay(subset: frozenset[str]) -> ReplayObservation:
        passed = {"b", "d"} <= subset
        return ReplayObservation(
            passed,
            float(len(subset & {"b", "d"})),
            True,
            "TARGET_PASSED" if passed else "TARGET_FAILED",
        )

    result = CausalReducer(replay, replay_budget=30).reduce(
        ["a", "b", "c", "d"], monotonicity_declared=True
    )
    assert result.selected == ("b", "d")
    assert result.minimality_grade is MinimalityGrade.ONE_MINIMAL
    assert result.replay_count > 1
    assert all(not test.accepted for test in result.one_minimality_tests)


def test_global_minimum_only_with_exhaustive_declaration() -> None:
    def replay(subset: frozenset[str]) -> ReplayObservation:
        passed = subset == {"x"} or len(subset) > 1
        return ReplayObservation(
            passed, float(passed), True, "TARGET_PASSED" if passed else "TARGET_FAILED"
        )

    result = CausalReducer(replay, replay_budget=20).reduce(
        ["x", "y"], monotonicity_declared=True, exhaustive_space_evaluated=True
    )
    assert result.selected == ("x",)
    assert result.minimality_grade is MinimalityGrade.GLOBAL_MINIMUM


def test_reducer_drops_control_damaging_candidate_from_failed_union() -> None:
    def replay(subset: frozenset[str]) -> ReplayObservation:
        target_passed = {"b", "d"} <= subset
        controls_passed = "z" not in subset
        accepted = target_passed and controls_passed
        return ReplayObservation(
            accepted,
            float(len(subset & {"b", "d"})),
            controls_passed,
            "TARGET_PASSED"
            if accepted
            else "CONTROLS_FAILED"
            if target_passed
            else "TARGET_FAILED",
        )

    result = CausalReducer(replay, replay_budget=30).reduce(["a", "b", "d", "z"])

    assert result.selected == ("b", "d")
    assert result.minimality_grade is MinimalityGrade.ONE_MINIMAL
    assert any(
        experiment.occurrence_ids == ("a", "b", "d") and experiment.observation.accepted
        for experiment in result.experiments
    )
