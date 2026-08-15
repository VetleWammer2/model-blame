from modelblame.status import (
    CausalClaim,
    MinimalityGrade,
    ReplayGrade,
    ReplayResultState,
)


def test_status_axes_remain_distinct() -> None:
    assert CausalClaim.ATTRIBUTED.value == "ATTRIBUTED"
    assert ReplayResultState.TARGET_PASSED.value == "TARGET_PASSED"
    assert MinimalityGrade.ONE_MINIMAL.value == "ONE_MINIMAL"
    assert CausalClaim.ATTRIBUTED.value not in ReplayResultState


def test_only_audited_replay_grades_permit_verified_claims() -> None:
    assert ReplayGrade.BITWISE.permits_verified_claim
    assert ReplayGrade.NUMERIC.permits_verified_claim
    assert ReplayGrade.STATISTICAL.permits_verified_claim
    assert not ReplayGrade.FAILED.permits_verified_claim
    assert not ReplayGrade.UNAUDITED.permits_verified_claim
