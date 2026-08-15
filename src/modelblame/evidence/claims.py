"""Explicit causal-claim vocabulary; no scalar confidence shortcut."""

from enum import StrEnum


class CausalClaim(StrEnum):
    ATTRIBUTED = "ATTRIBUTED"
    COUNTERFACTUAL_EFFECT = "COUNTERFACTUAL_EFFECT"
    NECESSARY_IN_CONTEXT = "NECESSARY_IN_CONTEXT"
    SUFFICIENT_ON_BASELINE = "SUFFICIENT_ON_BASELINE"
    BIDIRECTIONAL_CAUSAL_EVIDENCE = "BIDIRECTIONAL_CAUSAL_EVIDENCE"
    INCONCLUSIVE = "INCONCLUSIVE"


STRONG_CLAIMS = {
    CausalClaim.NECESSARY_IN_CONTEXT,
    CausalClaim.SUFFICIENT_ON_BASELINE,
    CausalClaim.BIDIRECTIONAL_CAUSAL_EVIDENCE,
}


CLAIM_SCOPE = (
    "Under the recorded training procedure, environment scope, intervention "
    "semantics, and behavioral probes, ablating these occurrences produced "
    "the measured counterfactual effect."
)
