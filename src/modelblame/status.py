"""Shared status and scientific-claim vocabulary.

These are deliberately separate axes.  In particular, an attribution rank is
not a replay result, and a replay result is not automatically a causal claim.
"""

from __future__ import annotations

from enum import StrEnum


class DeterminismMode(StrEnum):
    STRICT = "strict"
    BEST_EFFORT = "best-effort"
    OFF = "off"


class ReplayGrade(StrEnum):
    BITWISE = "BITWISE"
    NUMERIC = "NUMERIC"
    STATISTICAL = "STATISTICAL"
    FAILED = "FAILED"
    UNAUDITED = "UNAUDITED"

    @property
    def permits_verified_claim(self) -> bool:
        """Whether executed replay may support a scoped causal claim."""

        return self in {self.BITWISE, self.NUMERIC, self.STATISTICAL}


class ReplayResultState(StrEnum):
    COMPLETED = "COMPLETED"
    REPLAY_DIVERGED = "REPLAY_DIVERGED"
    TARGET_PASSED = "TARGET_PASSED"
    TARGET_FAILED = "TARGET_FAILED"
    CONTROLS_FAILED = "CONTROLS_FAILED"
    ENVIRONMENT_FAILED = "ENVIRONMENT_FAILED"
    TIMEOUT = "TIMEOUT"
    INCONCLUSIVE = "INCONCLUSIVE"


class BehaviorState(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNCERTAIN = "UNCERTAIN"


class CausalClaim(StrEnum):
    ATTRIBUTED = "ATTRIBUTED"
    COUNTERFACTUAL_EFFECT = "COUNTERFACTUAL_EFFECT"
    NECESSARY_IN_CONTEXT = "NECESSARY_IN_CONTEXT"
    SUFFICIENT_ON_BASELINE = "SUFFICIENT_ON_BASELINE"
    INCONCLUSIVE = "INCONCLUSIVE"


class MinimalityGrade(StrEnum):
    GLOBAL_MINIMUM = "GLOBAL_MINIMUM"
    ONE_MINIMAL = "ONE_MINIMAL"
    BUDGET_MINIMAL = "BUDGET_MINIMAL"
    UNREDUCED = "UNREDUCED"
    NO_ACCEPTED_PATCH = "NO_ACCEPTED_PATCH"


class HoldoutResultState(StrEnum):
    SEALED = "SEALED"
    SEARCH_PASSED = "SEARCH_PASSED"
    HOLDOUT_PASSED = "HOLDOUT_PASSED"
    HOLDOUT_FAILED = "HOLDOUT_FAILED"
