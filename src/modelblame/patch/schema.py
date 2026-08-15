"""Safe version-1 gradient intervention patch schema."""

from __future__ import annotations

import hmac
import json
from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelblame.config.validation import ConfigLoadError, read_structured_file
from modelblame.data.identity import OCCURRENCE_ID_PATTERN
from modelblame.util.canonical_json import (
    CanonicalizationError,
    content_hash,
    validate_json_tree,
)

MAX_PATCH_BYTES = 8 * 1024 * 1024
MAX_PATCH_OPERATIONS = 10_000
MAX_PATCH_OCCURRENCES = 1_000_000
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OccurrenceId = Annotated[str, Field(pattern=r"^occ_[0-9a-f]{64}$")]


class PatchValidationError(ValueError):
    """Raised when a patch does not match its declared run or ledger."""


class PatchOperationType(StrEnum):
    GRADIENT_ABLATE = "GRADIENT_ABLATE"
    REWEIGHT = "REWEIGHT"


class LossNormalization(StrEnum):
    FIXED_DENOMINATOR = "FIXED_DENOMINATOR"
    RENORMALIZED = "RENORMALIZED"


class StrictPatchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class GradientAblateOperation(StrictPatchModel):
    op: Literal[PatchOperationType.GRADIENT_ABLATE] = PatchOperationType.GRADIENT_ABLATE
    occurrence_ids: tuple[OccurrenceId, ...] = Field(
        min_length=1, max_length=MAX_PATCH_OCCURRENCES
    )
    normalization: LossNormalization = LossNormalization.FIXED_DENOMINATOR

    @field_validator("occurrence_ids")
    @classmethod
    def _unique_occurrences(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("an operation cannot repeat an occurrence ID")
        return value


class ReweightOperation(StrictPatchModel):
    op: Literal[PatchOperationType.REWEIGHT] = PatchOperationType.REWEIGHT
    occurrence_weights: dict[OccurrenceId, float] = Field(
        min_length=1, max_length=MAX_PATCH_OCCURRENCES
    )
    normalization: LossNormalization = LossNormalization.FIXED_DENOMINATOR

    @field_validator("occurrence_weights", mode="before")
    @classmethod
    def _weights_are_json_numbers(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("occurrence_weights must be an object")
        for weight in value.values():
            if isinstance(weight, bool) or not isinstance(weight, int | float):
                raise ValueError("reweight values must be JSON numbers")
        return value

    @field_validator("occurrence_weights")
    @classmethod
    def _valid_weights(cls, value: dict[str, float]) -> dict[str, float]:
        for occurrence_id, weight in value.items():
            if not OCCURRENCE_ID_PATTERN.fullmatch(occurrence_id):
                raise ValueError(f"invalid occurrence ID: {occurrence_id!r}")
            if not 0.0 <= float(weight) <= 1_000.0:
                raise ValueError("reweight values must be finite and in [0, 1000]")
        return value


PatchOperation = Annotated[
    GradientAblateOperation | ReweightOperation, Field(discriminator="op")
]


class Patch(StrictPatchModel):
    """A content-bound intervention over immutable occurrence IDs."""

    schema_version: Literal[1] = 1
    run_id: Annotated[str, Field(min_length=1, max_length=256)]
    run_hash: Sha256
    behavior_contract_hash: Sha256
    operations: tuple[PatchOperation, ...] = Field(
        min_length=1, max_length=MAX_PATCH_OPERATIONS
    )
    patch_hash: Sha256 | None = None

    @field_validator("run_id")
    @classmethod
    def _safe_run_id(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("run_id cannot contain whitespace")
        return value

    @model_validator(mode="after")
    def _no_conflicts_and_valid_hash(self) -> Self:
        claimed: dict[str, PatchOperationType] = {}
        normalizations = {operation.normalization for operation in self.operations}
        if len(normalizations) != 1:
            raise ValueError("a patch cannot mix loss-normalization semantics")
        for operation in self.operations:
            occurrence_ids = (
                operation.occurrence_ids
                if isinstance(operation, GradientAblateOperation)
                else operation.occurrence_weights.keys()
            )
            for occurrence_id in occurrence_ids:
                if occurrence_id in claimed:
                    raise ValueError(
                        f"occurrence {occurrence_id} is targeted by multiple operations"
                    )
                claimed[occurrence_id] = PatchOperationType(operation.op)
        if len(claimed) > MAX_PATCH_OCCURRENCES:
            raise ValueError("patch exceeds the total occurrence limit")
        if self.patch_hash is not None and not hmac.compare_digest(
            self.patch_hash, self.computed_hash()
        ):
            raise ValueError("patch_hash does not match the canonical patch content")
        return self

    @property
    def occurrence_ids(self) -> tuple[str, ...]:
        """Return all selected IDs in stable operation order."""

        selected: list[str] = []
        for operation in self.operations:
            if isinstance(operation, GradientAblateOperation):
                selected.extend(operation.occurrence_ids)
            else:
                selected.extend(operation.occurrence_weights)
        return tuple(selected)

    def computed_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"patch_hash"})
        return content_hash(payload)

    def with_computed_hash(self) -> Patch:
        """Return a new immutable patch carrying its canonical digest."""

        return self.model_copy(update={"patch_hash": self.computed_hash()})

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        run_hash: str,
        behavior_contract_hash: str,
        operations: tuple[PatchOperation, ...] | list[PatchOperation],
    ) -> Patch:
        patch = cls(
            run_id=run_id,
            run_hash=run_hash,
            behavior_contract_hash=behavior_contract_hash,
            operations=tuple(operations),
        )
        return patch.with_computed_hash()


def _decode_patch_bytes(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_PATCH_BYTES:
        raise PatchValidationError(
            f"patch exceeds the {MAX_PATCH_BYTES}-byte input limit"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PatchValidationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: _raise_nonfinite(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PatchValidationError(f"invalid patch JSON: {error}") from error
    if not isinstance(parsed, Mapping):
        raise PatchValidationError("patch root must be an object")
    try:
        validate_json_tree(parsed, max_depth=16, max_items=2_000_000)
    except CanonicalizationError as error:
        raise PatchValidationError(str(error)) from error
    return dict(parsed)


def _raise_nonfinite(value: str) -> None:
    raise PatchValidationError(f"non-finite JSON number is forbidden: {value}")


def parse_patch(
    source: str | Path | bytes | Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
    expected_run_hash: str | None = None,
    expected_behavior_contract_hash: str | None = None,
    known_occurrence_ids: Collection[str] | None = None,
    require_hash: bool = True,
) -> Patch:
    """Parse and context-check an untrusted patch without executing anything."""

    if isinstance(source, Mapping):
        raw = dict(source)
        try:
            validate_json_tree(raw, max_depth=16, max_items=2_000_000)
        except CanonicalizationError as error:
            raise PatchValidationError(str(error)) from error
    elif isinstance(source, bytes):
        raw = _decode_patch_bytes(source)
    else:
        path = Path(source)
        try:
            raw = read_structured_file(
                path,
                allowed_suffixes=frozenset({".json"}),
                max_bytes=MAX_PATCH_BYTES,
            )
        except ConfigLoadError as error:
            raise PatchValidationError(str(error)) from error
    try:
        patch = Patch.model_validate(raw)
    except ValueError as error:
        raise PatchValidationError(str(error)) from error
    if require_hash and patch.patch_hash is None:
        raise PatchValidationError("patch_hash is required")
    expected_values = (
        ("run_id", expected_run_id, patch.run_id),
        ("run_hash", expected_run_hash, patch.run_hash),
        (
            "behavior_contract_hash",
            expected_behavior_contract_hash,
            patch.behavior_contract_hash,
        ),
    )
    for name, expected, actual in expected_values:
        if expected is not None and not hmac.compare_digest(expected, actual):
            raise PatchValidationError(
                f"patch {name} does not match the source context"
            )
    if known_occurrence_ids is not None:
        known = set(known_occurrence_ids)
        unknown = sorted(set(patch.occurrence_ids) - known)
        if unknown:
            preview = ", ".join(unknown[:3])
            suffix = " ..." if len(unknown) > 3 else ""
            raise PatchValidationError(f"unknown occurrence IDs: {preview}{suffix}")
    return patch


def patch_hash(patch: Patch) -> str:
    """Return the canonical hash independently of the embedded hash field."""

    return patch.computed_hash()
