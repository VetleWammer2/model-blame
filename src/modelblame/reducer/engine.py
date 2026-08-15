"""Replay-budgeted causal reduction with interaction-aware fallback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

from modelblame.reducer.beam import BeamEvaluation, bounded_beam_search
from modelblame.reducer.ddmin import ddmin
from modelblame.reducer.interactions import InteractionSignal, detect_non_monotonicity
from modelblame.reducer.minimality import (
    MinimalityGrade,
    MinimalityTest,
    test_one_minimality,
)


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    accepted: bool
    target_effect: float
    controls_passed: bool
    status: str
    experiment_hash: str = ""


@dataclass(frozen=True, slots=True)
class TestedSubset:
    occurrence_ids: tuple[str, ...]
    observation: ReplayObservation


@dataclass(frozen=True, slots=True)
class ReductionResult:
    selected: tuple[str, ...]
    accepted: bool
    minimality_grade: MinimalityGrade
    replay_budget: int
    replay_count: int
    budget_exhausted: bool
    monotonicity_basis: str
    experiments: tuple[TestedSubset, ...]
    one_minimality_tests: tuple[MinimalityTest, ...]
    interaction_signals: tuple[InteractionSignal, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["minimality_grade"] = self.minimality_grade.value
        return value


class ReplayBudgetExhaustedError(RuntimeError):
    pass


class CausalReducer:
    """Minimize occurrence sets using only executed replay observations."""

    def __init__(
        self,
        replay: Callable[[frozenset[str]], ReplayObservation],
        *,
        replay_budget: int,
        beam_width: int = 8,
    ) -> None:
        if replay_budget < 1:
            raise ValueError("replay_budget must be positive")
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self._replay = replay
        self._budget = replay_budget
        self._beam_width = beam_width
        self._cache: dict[frozenset[str], ReplayObservation] = {}
        self._order: list[frozenset[str]] = []

    @property
    def remaining(self) -> int:
        return self._budget - len(self._order)

    def _evaluate(self, subset: frozenset[str]) -> ReplayObservation:
        subset = frozenset(subset)
        if subset in self._cache:
            return self._cache[subset]
        if self.remaining <= 0:
            raise ReplayBudgetExhaustedError("causal reducer replay budget exhausted")
        observation = self._replay(subset)
        if observation.accepted and not observation.controls_passed:
            raise ValueError(
                "replay observation cannot be accepted when controls failed"
            )
        if not observation.experiment_hash:
            payload = {
                "subset": sorted(subset),
                "accepted": observation.accepted,
                "effect": observation.target_effect,
                "controls": observation.controls_passed,
                "status": observation.status,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            observation = ReplayObservation(
                accepted=observation.accepted,
                target_effect=observation.target_effect,
                controls_passed=observation.controls_passed,
                status=observation.status,
                experiment_hash=digest,
            )
        self._cache[subset] = observation
        self._order.append(subset)
        return observation

    def reduce(
        self,
        candidates: Iterable[str],
        *,
        monotonicity_declared: bool = False,
        exhaustive_space_evaluated: bool = False,
    ) -> ReductionResult:
        initial = frozenset(candidates)
        if not initial:
            raise ValueError("causal reduction requires candidates")
        selected: frozenset[str] | None = None
        budget_exhausted = False
        try:
            initial_result = self._evaluate(initial)
            if initial_result.accepted:
                selected = initial
                if not monotonicity_declared:
                    for item in sorted(initial)[: self._beam_width]:
                        if self.remaining <= 0:
                            break
                        singleton = frozenset({item})
                        if self._evaluate(singleton).accepted:
                            selected = singleton
                            break
                selected = ddmin(
                    selected, lambda subset: self._evaluate(subset).accepted
                )
            else:
                # A broad patch can remove the target yet fail controls because
                # it contains one harmful candidate. Probe immediate
                # complements before growing singleton beams so this common
                # non-monotonic surface is reachable within a bounded budget.
                for item in sorted(initial):
                    if self.remaining <= 0:
                        break
                    complement = initial - {item}
                    if self._evaluate(complement).accepted:
                        selected = ddmin(
                            complement,
                            lambda subset: self._evaluate(subset).accepted,
                        )
                        break
                if selected is None and self.remaining > 0:
                    best, _ = bounded_beam_search(
                        initial,
                        lambda subset: BeamEvaluation(
                            subset=subset,
                            accepted=(observation := self._evaluate(subset)).accepted,
                            objective=observation.target_effect,
                        ),
                        beam_width=self._beam_width,
                        max_evaluations=max(1, self.remaining),
                    )
                    selected = best.subset if best else None
        except ReplayBudgetExhaustedError:
            budget_exhausted = True

        accepted_seen = [
            subset
            for subset, observation in self._cache.items()
            if observation.accepted
        ]
        if accepted_seen:
            smallest_seen = min(
                accepted_seen,
                key=lambda subset: (len(subset), tuple(sorted(subset))),
            )
            if selected is None or len(smallest_seen) < len(selected):
                selected = smallest_seen

        booleans = {
            subset: observation.accepted for subset, observation in self._cache.items()
        }
        signals = tuple(detect_non_monotonicity(booleans))
        monotonicity_basis = (
            "INTERACTION_SENSITIVE"
            if signals
            else "DECLARED"
            if monotonicity_declared
            else "EMPIRICALLY_PROBED"
        )
        # Observed interactions invalidate a silent monotonicity assumption. A
        # bounded beam pass may still find a smaller accepted set.
        if selected is not None and signals and self.remaining > 0:
            try:
                best, _ = bounded_beam_search(
                    selected,
                    lambda subset: BeamEvaluation(
                        subset=subset,
                        accepted=(observation := self._evaluate(subset)).accepted,
                        objective=observation.target_effect,
                    ),
                    beam_width=self._beam_width,
                    max_evaluations=max(1, self.remaining),
                )
                if best is not None and len(best.subset) <= len(selected):
                    selected = best.subset
            except ReplayBudgetExhaustedError:
                budget_exhausted = True

        minimality_tests: tuple[MinimalityTest, ...] = ()
        one_minimal = False
        if selected is not None:
            try:
                one_minimal, minimality_tests = test_one_minimality(
                    selected, lambda subset: self._evaluate(subset).accepted
                )
            except ReplayBudgetExhaustedError:
                budget_exhausted = True

        if selected is None:
            grade = MinimalityGrade.NO_ACCEPTED_PATCH
        elif exhaustive_space_evaluated and one_minimal:
            grade = MinimalityGrade.GLOBAL_MINIMUM
        elif one_minimal:
            grade = MinimalityGrade.ONE_MINIMAL
        elif budget_exhausted:
            grade = MinimalityGrade.BUDGET_MINIMAL
        else:
            grade = MinimalityGrade.UNREDUCED
        experiments = tuple(
            TestedSubset(tuple(sorted(subset)), self._cache[subset])
            for subset in self._order
        )
        return ReductionResult(
            selected=tuple(sorted(selected or ())),
            accepted=selected is not None and self._cache[selected].accepted,
            minimality_grade=grade,
            replay_budget=self._budget,
            replay_count=len(self._order),
            budget_exhausted=budget_exhausted,
            monotonicity_basis=monotonicity_basis,
            experiments=experiments,
            one_minimality_tests=minimality_tests,
            interaction_signals=signals,
        )
