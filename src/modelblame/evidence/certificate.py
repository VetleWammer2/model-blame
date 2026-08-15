"""Strict evidence-certificate schema and scientific consistency checks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelblame.evidence.claims import CLAIM_SCOPE, STRONG_CLAIMS, CausalClaim
from modelblame.reducer.minimality import MinimalityGrade


class HashedIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScoreEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    score: float
    prompt_scores: list[float]
    state: str
    split: str


class ControlEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    original_scores: list[float]
    counterfactual_scores: list[float]
    original_score: float
    counterfactual_score: float
    mean_drift: float = Field(ge=0.0)
    max_item_drift: float = Field(ge=0.0)
    max_mean_drift: float = Field(ge=0.0)
    max_allowed_item_drift: float = Field(ge=0.0)
    passed: bool


class EvidenceCertificate(BaseModel):
    """Versioned, mutation-resistant summary of executed causal evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    modelblame_version: str
    source_run: HashedIdentity
    source_checkpoint_hashes: list[str]
    counterfactual_checkpoint_hash: str
    adapter: HashedIdentity
    training_code_identity: dict[str, Any]
    environment_identity: dict[str, Any]
    dataset_fingerprint: str
    tokenizer_fingerprint: str
    behavior_contract_hash: str
    control_contract_hashes: list[str]
    patch_hash: str
    intervention_semantics: dict[str, Any]
    candidate_methods: list[str]
    candidate_method_configurations: dict[str, dict[str, Any]]
    candidate_counts: dict[str, int]
    replay_budget: int = Field(ge=1)
    replay_experiment_hashes: list[str]
    original_behavior_result: ScoreEvidence
    counterfactual_behavior_result: ScoreEvidence
    sealed_holdout_result: dict[str, Any]
    control_results: list[ControlEvidence]
    effect_sizes: dict[str, float]
    confidence_intervals: dict[str, dict[str, float]]
    replay_grade: Literal["BITWISE", "NUMERIC", "STATISTICAL", "FAILED", "UNAUDITED"]
    causal_claim_grade: CausalClaim = Field(strict=False)
    minimality_grade: MinimalityGrade = Field(strict=False)
    complete_subset_space_evaluated: bool = False
    one_minimality_tests: list[dict[str, Any]]
    interaction_diagnostics: list[dict[str, Any]]
    unsupported_assumptions: list[str]
    warnings: list[str]
    generated_artifact_hashes: dict[str, str]
    claim_scope: str = CLAIM_SCOPE
    training_text_included: bool = False

    @field_validator(
        "source_checkpoint_hashes",
        "control_contract_hashes",
        "replay_experiment_hashes",
    )
    @classmethod
    def validate_hash_lists(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    "certificate hash lists require lowercase SHA-256 values"
                )
        return values

    @field_validator(
        "counterfactual_checkpoint_hash",
        "dataset_fingerprint",
        "tokenizer_fingerprint",
        "behavior_contract_hash",
        "patch_hash",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("certificate identities require lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def validate_claim_consistency(self) -> EvidenceCertificate:
        if self.causal_claim_grade in STRONG_CLAIMS and self.replay_grade in {
            "FAILED",
            "UNAUDITED",
        }:
            raise ValueError("strong causal claims require an audited replay")
        if self.causal_claim_grade in STRONG_CLAIMS:
            if not self.control_results or not all(
                result.passed for result in self.control_results
            ):
                raise ValueError(
                    "strong causal claims require all declared controls to pass"
                )
            if self.sealed_holdout_result.get("status") != "PASSED":
                raise ValueError("strong causal claims require a passed sealed holdout")
        if self.causal_claim_grade is CausalClaim.BIDIRECTIONAL_CAUSAL_EVIDENCE:
            directions = self.sealed_holdout_result.get("directions", {})
            if (
                directions.get("removal") != "PASSED"
                or directions.get("addition") != "PASSED"
            ):
                raise ValueError(
                    "bidirectional evidence requires independently passed directions"
                )
        if self.minimality_grade is MinimalityGrade.ONE_MINIMAL:
            if not self.one_minimality_tests or any(
                test.get("accepted") is not False for test in self.one_minimality_tests
            ):
                raise ValueError(
                    "ONE_MINIMAL requires every single-removal test to fail"
                )
        if (
            self.minimality_grade is MinimalityGrade.GLOBAL_MINIMUM
            and not self.complete_subset_space_evaluated
        ):
            raise ValueError("GLOBAL_MINIMUM requires complete subset-space evidence")
        return self
