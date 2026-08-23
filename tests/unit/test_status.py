from modelblame.evidence.certificate import EvidenceCertificate
from modelblame.evidence.claims import CausalClaim as EvidenceCausalClaim
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


def test_v1_schema_does_not_advertise_unbound_bidirectional_grade() -> None:
    unavailable = "BIDIRECTIONAL_CAUSAL_EVIDENCE"
    assert unavailable not in {claim.value for claim in CausalClaim}
    assert unavailable not in {claim.value for claim in EvidenceCausalClaim}
    schema = EvidenceCertificate.model_json_schema()
    assert unavailable not in schema["$defs"]["CausalClaim"]["enum"]
