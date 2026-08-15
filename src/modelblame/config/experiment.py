"""Strict schema for a recorded ModelBlame training experiment."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from modelblame.config.validation import read_structured_file
from modelblame.status import DeterminismMode
from modelblame.util.canonical_json import content_hash
from modelblame.util.paths import UnsafePathError, validate_relative_path

Identifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
]


class StrictConfigModel(BaseModel):
    """Immutable base model that rejects undeclared configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DatasetFormat(StrEnum):
    AUTO = "auto"
    JSONL = "jsonl"
    PARQUET = "parquet"


class DatasetConfig(StrictConfigModel):
    path: str
    format: DatasetFormat = DatasetFormat.AUTO
    source: Identifier = "default"
    prompt_field: Identifier = "prompt"
    completion_field: Identifier = "completion"
    labels_fields: tuple[Identifier, ...] = ()
    metadata_fields: tuple[Identifier, ...] = ()
    sample_weight_field: Identifier | None = "sample_weight"

    @field_validator("path")
    @classmethod
    def _relative_data_path(cls, value: str) -> str:
        try:
            return validate_relative_path(value)
        except UnsafePathError as error:
            raise ValueError(str(error)) from error

    @field_validator("labels_fields", "metadata_fields")
    @classmethod
    def _unique_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("declared dataset fields must be unique")
        return value

    @model_validator(mode="after")
    def _separate_semantic_fields(self) -> Self:
        reserved = {self.prompt_field, self.completion_field}
        if self.sample_weight_field is not None:
            reserved.add(self.sample_weight_field)
        overlap = (set(self.labels_fields) & set(self.metadata_fields)) | (
            reserved & (set(self.labels_fields) | set(self.metadata_fields))
        )
        if overlap:
            raise ValueError(
                f"dataset fields have conflicting roles: {sorted(overlap)}"
            )
        return self


class TokenizerKind(StrEnum):
    BYTE = "byte"
    HUGGINGFACE = "huggingface"


class TokenizerConfig(StrictConfigModel):
    type: TokenizerKind = TokenizerKind.BYTE
    path: str | None = None
    add_bos: bool = True
    add_eos: bool = True

    @field_validator("path")
    @classmethod
    def _relative_tokenizer_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_relative_path(value)
        except UnsafePathError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def _path_matches_kind(self) -> Self:
        if self.type is TokenizerKind.BYTE and self.path is not None:
            raise ValueError("the built-in byte tokenizer does not accept a path")
        if self.type is TokenizerKind.HUGGINGFACE and self.path is None:
            raise ValueError("a local path is required for a Hugging Face tokenizer")
        return self


class LoRAConfig(StrictConfigModel):
    rank: PositiveInt = Field(le=1024)
    alpha: PositiveFloat = Field(le=1_000_000)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: tuple[Identifier, ...] = ("q_proj", "v_proj")

    @field_validator("target_modules")
    @classmethod
    def _targets_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("LoRA requires at least one target module")
        if len(value) != len(set(value)):
            raise ValueError("LoRA target modules must be unique")
        return value


class ModelArchitecture(StrEnum):
    TINY_CAUSAL_LM = "tiny_causal_lm"
    LLAMA_STYLE = "llama_style"
    HUGGINGFACE = "huggingface"


class ModelConfig(StrictConfigModel):
    architecture: ModelArchitecture = ModelArchitecture.TINY_CAUSAL_LM
    local_path: str | None = None
    vocab_size: PositiveInt = Field(default=260, le=1_000_000)
    context_length: PositiveInt = Field(default=128, ge=4, le=131_072)
    hidden_size: PositiveInt = Field(default=128, le=65_536)
    num_layers: PositiveInt = Field(default=2, le=1024)
    num_heads: PositiveInt = Field(default=4, le=1024)
    intermediate_size: PositiveInt = Field(default=512, le=262_144)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    bias: bool = False
    lora: LoRAConfig | None = None

    @field_validator("local_path")
    @classmethod
    def _relative_model_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_relative_path(value)
        except UnsafePathError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def _valid_dimensions_and_source(self) -> Self:
        if (
            self.architecture is not ModelArchitecture.HUGGINGFACE
            and self.vocab_size < 260
        ):
            raise ValueError("built-in models require vocab_size of at least 260")
        if self.hidden_size % self.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                "model.intermediate_size must be at least model.hidden_size"
            )
        if (
            self.architecture is ModelArchitecture.HUGGINGFACE
            and self.local_path is None
        ):
            raise ValueError("a local_path is required for the Hugging Face adapter")
        if self.architecture is not ModelArchitecture.HUGGINGFACE and self.local_path:
            raise ValueError("local_path is only valid for the Hugging Face adapter")
        return self


class OptimizerConfig(StrictConfigModel):
    type: Literal["adamw"] = "adamw"
    lr: PositiveFloat = Field(default=3e-4, le=10.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: PositiveFloat = Field(default=1e-8, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=100.0)
    maximize: bool = False
    capturable: bool = False

    @field_validator("betas")
    @classmethod
    def _valid_betas(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(item) and 0 <= item < 1 for item in value):
            raise ValueError("AdamW betas must be finite and in [0, 1)")
        return value


class SchedulerKind(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class SchedulerConfig(StrictConfigModel):
    type: SchedulerKind = SchedulerKind.CONSTANT
    warmup_steps: int = Field(default=0, ge=0, le=1_000_000_000)


class PrecisionMode(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class TrainingConfig(StrictConfigModel):
    steps: PositiveInt = Field(le=1_000_000_000)
    batch_size: PositiveInt = Field(default=1, le=1_000_000)
    gradient_accumulation: PositiveInt = Field(
        default=1,
        le=1_000_000,
        validation_alias=AliasChoices(
            "gradient_accumulation", "gradient_accumulation_steps"
        ),
    )
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    determinism: DeterminismMode | None = Field(
        default=None,
        validation_alias=AliasChoices("determinism", "deterministic"),
    )
    device: str = "cpu"
    precision: PrecisionMode = PrecisionMode.FP32
    max_grad_norm: PositiveFloat | None = Field(default=1.0, le=1_000_000)

    @field_validator("device")
    @classmethod
    def _valid_device(cls, value: str) -> str:
        if not re.fullmatch(r"(?:cpu|auto|cuda(?::[0-9]{1,4})?)", value):
            raise ValueError("device must be cpu, auto, cuda, or cuda:<index>")
        return value


class CheckpointConfig(StrictConfigModel):
    interval: PositiveInt = Field(le=1_000_000_000)
    keep_last: PositiveInt | None = Field(default=None, le=1_000_000)


class ExperimentConfig(StrictConfigModel):
    schema_version: Literal[1] = 1
    name: Identifier = "experiment"
    adapter: Identifier = "tiny_causal_lm"
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    determinism: DeterminismMode = Field(
        default=DeterminismMode.STRICT,
        validation_alias=AliasChoices("determinism", "deterministic"),
    )
    dataset: DatasetConfig
    tokenizer: TokenizerConfig = TokenizerConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    training: TrainingConfig
    checkpoints: CheckpointConfig

    @model_validator(mode="after")
    def _cross_section_constraints(self) -> Self:
        if self.scheduler.warmup_steps > self.training.steps:
            raise ValueError("scheduler.warmup_steps cannot exceed training.steps")
        if self.checkpoints.interval > self.training.steps:
            raise ValueError("checkpoints.interval cannot exceed training.steps")
        if (
            self.training.precision is PrecisionMode.FP16
            and self.effective_device == "cpu"
        ):
            raise ValueError("fp16 training is not supported on CPU")
        return self

    @property
    def effective_seed(self) -> int:
        return self.training.seed if self.training.seed is not None else self.seed

    @property
    def effective_determinism(self) -> DeterminismMode:
        return self.training.determinism or self.determinism

    @property
    def effective_device(self) -> str:
        return self.training.device

    def config_hash(self) -> str:
        """Hash the complete normalized experiment configuration."""

        return content_hash(self.model_dump(mode="json", by_alias=False))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a TOML or JSON experiment configuration."""

    raw = read_structured_file(path, allowed_suffixes=frozenset({".toml", ".json"}))
    return ExperimentConfig.model_validate(raw)


# A concise alias for adapter authors; both names perform identical validation.
load_experiment = load_experiment_config
