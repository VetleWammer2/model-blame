"""Declarative, hashable behavior contracts with sealed holdout separation."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelblame.config.validation import read_structured_file
from modelblame.util.canonical_json import content_hash
from modelblame.util.paths import UnsafePathError, validate_relative_path
from modelblame.util.safe_regex import UnsafeRegexError, compile_safe_regex

ContractId = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$"),
]
BoundedText = Annotated[str, Field(min_length=1, max_length=1_000_000)]


class StrictBehaviorModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class TokenLogProbabilityScorer(StrictBehaviorModel):
    type: Literal["token_log_probability"]
    token: BoundedText | None = None


class SequenceLogProbabilityScorer(StrictBehaviorModel):
    type: Literal["sequence_log_probability"]
    completion: BoundedText | None = None


class SequenceNllScorer(StrictBehaviorModel):
    type: Literal["sequence_nll"]
    completion: BoundedText | None = None


class SequenceLogProbabilityMarginScorer(StrictBehaviorModel):
    type: Literal["sequence_logprob_margin"]
    preferred: BoundedText | None = None
    alternative: BoundedText | None = None


class MultipleChoiceMarginScorer(StrictBehaviorModel):
    type: Literal["multiple_choice_margin"]
    choices: tuple[BoundedText, ...] | None = None
    correct: int | None = Field(default=None, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def _complete_global_choices(self) -> Self:
        if (self.choices is None) != (self.correct is None):
            raise ValueError("multiple-choice scorer needs both choices and correct")
        if self.choices is not None:
            if len(self.choices) < 2:
                raise ValueError("multiple-choice scorer needs at least two choices")
            if self.correct is None or self.correct >= len(self.choices):
                raise ValueError("multiple-choice correct index is outside choices")
        return self


class GreedyExactMatchScorer(StrictBehaviorModel):
    type: Literal["greedy_exact_match_rate"]
    expected: BoundedText | None = None
    max_new_tokens: int = Field(default=32, ge=1, le=4096)


class GreedyRegexMatchScorer(StrictBehaviorModel):
    type: Literal["greedy_regex_match_rate"]
    pattern: Annotated[str, Field(min_length=1, max_length=8192)] | None = None
    max_new_tokens: int = Field(default=32, ge=1, le=4096)

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                compile_safe_regex(value)
            except UnsafeRegexError as error:
                raise ValueError(f"unsafe regular expression: {error}") from error
        return value


class ScalarLossScorer(StrictBehaviorModel):
    type: Literal["scalar_loss"]
    completion: BoundedText | None = None


ScorerConfig = Annotated[
    TokenLogProbabilityScorer
    | SequenceLogProbabilityScorer
    | SequenceNllScorer
    | SequenceLogProbabilityMarginScorer
    | MultipleChoiceMarginScorer
    | GreedyExactMatchScorer
    | GreedyRegexMatchScorer
    | ScalarLossScorer,
    Field(discriminator="type"),
]


class PromptProbe(StrictBehaviorModel):
    """One fully declared probe; scorer-specific values may live here."""

    id: ContractId | None = None
    prompt: BoundedText
    token: BoundedText | None = None
    completion: BoundedText | None = None
    preferred: BoundedText | None = None
    alternative: BoundedText | None = None
    choices: tuple[BoundedText, ...] | None = None
    correct: int | None = Field(default=None, ge=0, le=1_000_000)
    expected: BoundedText | None = None
    pattern: Annotated[str, Field(min_length=1, max_length=8192)] | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)

    @model_validator(mode="after")
    def _valid_optional_pair_fields(self) -> Self:
        if (self.choices is None) != (self.correct is None):
            raise ValueError("a multiple-choice probe needs both choices and correct")
        if self.choices is not None:
            if len(self.choices) < 2:
                raise ValueError("a multiple-choice probe needs at least two choices")
            if self.correct is None or self.correct >= len(self.choices):
                raise ValueError("probe correct index is outside choices")
        if self.pattern is not None:
            try:
                compile_safe_regex(self.pattern)
            except UnsafeRegexError as error:
                raise ValueError(f"unsafe regular expression: {error}") from error
        return self


Probe = BoundedText | PromptProbe


def _portable_probe_file(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return validate_relative_path(value)
    except UnsafePathError as error:
        raise ValueError(str(error)) from error


class ProbeSet(StrictBehaviorModel):
    prompts: tuple[Probe, ...] = Field(default=(), max_length=100_000)
    prompts_file: str | None = None

    @field_validator("prompts_file")
    @classmethod
    def _valid_prompts_file(cls, value: str | None) -> str | None:
        return _portable_probe_file(value)

    @model_validator(mode="after")
    def _exactly_one_probe_source(self) -> Self:
        if bool(self.prompts) == bool(self.prompts_file):
            raise ValueError("declare exactly one of prompts or prompts_file")
        return self


class HoldoutProbeSet(ProbeSet):
    sealed: Literal[True] = True


class AggregationKind(StrEnum):
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class AggregationConfig(StrictBehaviorModel):
    type: AggregationKind = AggregationKind.MEAN


class Direction(StrEnum):
    GREATER_IS_PRESENT = "greater_is_present"
    LESS_IS_PRESENT = "less_is_present"


class MonotonicityDeclaration(StrEnum):
    NONE = "none"
    NONDECREASING = "nondecreasing"
    NONINCREASING = "nonincreasing"


class SeedPolicy(StrEnum):
    FIXED = "fixed"
    PAIRED_SEED_SET = "paired_seed_set"


class StatisticsConfig(StrictBehaviorModel):
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_samples: int = Field(default=2000, ge=1, le=10_000_000)
    bootstrap_seed: int = Field(default=0, ge=0, le=2**63 - 1)
    seed_policy: SeedPolicy = SeedPolicy.FIXED


class ControlContract(ProbeSet):
    id: ContractId
    scorer: ScorerConfig
    aggregation: AggregationConfig = AggregationConfig()
    max_mean_drift: float = Field(ge=0.0)
    max_item_drift: float = Field(ge=0.0)

    @field_validator("max_mean_drift", "max_item_drift")
    @classmethod
    def _finite_drift(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("control drift limits must be finite")
        return value

    def control_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class SearchBehaviorContract(StrictBehaviorModel):
    """Reducer-safe view that contains no holdout probes or file path."""

    schema_version: Literal[1]
    id: ContractId
    scorer: ScorerConfig
    aggregation: AggregationConfig
    direction: Direction
    present_threshold: float
    required_effect: float
    search: ProbeSet
    controls: tuple[ControlContract, ...]
    statistics: StatisticsConfig
    monotonicity: MonotonicityDeclaration
    behavior_contract_hash: str


class BehaviorContract(StrictBehaviorModel):
    schema_version: Literal[1] = 1
    id: ContractId
    scorer: ScorerConfig
    aggregation: AggregationConfig = AggregationConfig()
    direction: Direction = Direction.GREATER_IS_PRESENT
    present_threshold: float
    required_effect: float = Field(gt=0.0)
    search: ProbeSet
    holdout: HoldoutProbeSet
    controls: tuple[ControlContract, ...] = ()
    statistics: StatisticsConfig = StatisticsConfig()
    monotonicity: MonotonicityDeclaration = MonotonicityDeclaration.NONE

    @field_validator("present_threshold", "required_effect")
    @classmethod
    def _finite_effect_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("thresholds and effects must be finite")
        return value

    @model_validator(mode="after")
    def _valid_probes_and_controls(self) -> Self:
        control_ids = [control.id for control in self.controls]
        if len(control_ids) != len(set(control_ids)):
            raise ValueError("control IDs must be unique")
        for split_name, split in (("search", self.search), ("holdout", self.holdout)):
            self._validate_inline_probes(split_name, split, self.scorer)
        for control in self.controls:
            self._validate_inline_probes(
                f"control {control.id!r}", control, control.scorer
            )
        return self

    @staticmethod
    def _validate_inline_probes(
        split_name: str, split: ProbeSet, scorer: ScorerConfig
    ) -> None:
        if split.prompts_file:
            return
        for index, probe in enumerate(split.prompts):
            provided = (
                {"prompt"}
                if isinstance(probe, str)
                else {
                    name
                    for name, value in probe.model_dump().items()
                    if value is not None
                }
            )
            scorer_values = {
                name for name, value in scorer.model_dump().items() if value is not None
            }
            available = provided | scorer_values
            required: dict[str, set[str]] = {
                "token_log_probability": {"token"},
                "sequence_log_probability": {"completion"},
                "sequence_nll": {"completion"},
                "sequence_logprob_margin": {"preferred", "alternative"},
                "multiple_choice_margin": {"choices", "correct"},
                "greedy_exact_match_rate": {"expected"},
                "greedy_regex_match_rate": {"pattern"},
                "scalar_loss": {"completion"},
            }
            missing = required[scorer.type] - available
            if missing:
                raise ValueError(
                    f"{split_name} probe {index} is missing scorer fields: "
                    f"{sorted(missing)}"
                )

    def contract_hash(self) -> str:
        """Hash the complete normalized contract, including sealed metadata."""

        return content_hash(self.model_dump(mode="json"))

    def for_search(self) -> SearchBehaviorContract:
        """Return a type that makes sealed holdout data structurally unavailable."""

        return SearchBehaviorContract(
            schema_version=self.schema_version,
            id=self.id,
            scorer=self.scorer,
            aggregation=self.aggregation,
            direction=self.direction,
            present_threshold=self.present_threshold,
            required_effect=self.required_effect,
            search=self.search,
            controls=self.controls,
            statistics=self.statistics,
            monotonicity=self.monotonicity,
            behavior_contract_hash=self.contract_hash(),
        )


def load_behavior_contract(path: str | Path) -> BehaviorContract:
    """Load and validate a YAML or JSON behavior contract."""

    raw = read_structured_file(
        path, allowed_suffixes=frozenset({".yaml", ".yml", ".json"})
    )
    return BehaviorContract.model_validate(raw)


load_behavior_config = load_behavior_contract
