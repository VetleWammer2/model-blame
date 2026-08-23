from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from modelblame.data.identity import compute_example_id, compute_occurrence_id
from modelblame.patch.schema import (
    GradientAblateOperation,
    LossNormalization,
    Patch,
    PatchValidationError,
    ReservedSlotInjectOperation,
    ReweightOperation,
    parse_patch,
)


def _occurrence(position: int = 0) -> str:
    example = compute_example_id({"prompt": "p", "completion": "c"})
    return compute_occurrence_id(
        run_id="mb_test",
        global_step=1,
        microbatch_index=0,
        batch_position=position,
        packed_token_span=(position, position + 1),
        example_id=example,
    )


def _patch(*, occurrence: str | None = None) -> Patch:
    operation = GradientAblateOperation(occurrence_ids=(occurrence or _occurrence(),))
    return Patch.create(
        run_id="mb_test",
        run_hash="a" * 64,
        behavior_contract_hash="b" * 64,
        operations=[operation],
    )


def test_patch_defaults_to_fixed_denominator_and_hashes_itself() -> None:
    patch = _patch()
    operation = patch.operations[0]
    assert operation.normalization is LossNormalization.FIXED_DENOMINATOR
    assert patch.patch_hash == patch.computed_hash()


def test_patch_json_round_trip_checks_context_and_ledger() -> None:
    patch = _patch()
    encoded = json.dumps(patch.model_dump(mode="json")).encode()
    restored = parse_patch(
        encoded,
        expected_run_id="mb_test",
        expected_run_hash="a" * 64,
        expected_behavior_contract_hash="b" * 64,
        known_occurrence_ids={_occurrence()},
    )
    assert restored == patch


def test_reserved_slot_injection_round_trip_and_isolation() -> None:
    occurrence = _occurrence()
    patch = Patch.create(
        run_id="mb_test",
        run_hash="a" * 64,
        behavior_contract_hash="b" * 64,
        operations=[ReservedSlotInjectOperation(occurrence_ids=(occurrence,))],
    )
    restored = parse_patch(
        patch.model_dump(mode="json"), known_occurrence_ids={occurrence}
    )
    assert isinstance(restored.operations[0], ReservedSlotInjectOperation)
    with pytest.raises(ValidationError, match="cannot be mixed"):
        Patch(
            run_id="mb_test",
            run_hash="a" * 64,
            behavior_contract_hash="b" * 64,
            operations=(
                ReservedSlotInjectOperation(occurrence_ids=(occurrence,)),
                GradientAblateOperation(occurrence_ids=(_occurrence(1),)),
            ),
        )


def test_reserved_slot_injection_requires_fixed_denominator() -> None:
    with pytest.raises(ValidationError, match="FIXED_DENOMINATOR"):
        ReservedSlotInjectOperation(
            occurrence_ids=(_occurrence(),),
            normalization=LossNormalization.RENORMALIZED,
        )


def test_patch_rejects_content_tampering() -> None:
    patch = _patch()
    payload = patch.model_dump(mode="json")
    payload["behavior_contract_hash"] = "c" * 64
    with pytest.raises(PatchValidationError, match="patch_hash"):
        parse_patch(payload)


def test_patch_rejects_unknown_occurrence() -> None:
    with pytest.raises(PatchValidationError, match="unknown occurrence"):
        parse_patch(_patch().model_dump(mode="json"), known_occurrence_ids=set())


def test_patch_rejects_conflicting_operations() -> None:
    occurrence = _occurrence()
    with pytest.raises(ValidationError, match="multiple operations"):
        Patch(
            run_id="mb_test",
            run_hash="a" * 64,
            behavior_contract_hash="b" * 64,
            operations=(
                GradientAblateOperation(occurrence_ids=(occurrence,)),
                ReweightOperation(occurrence_weights={occurrence: 0.5}),
            ),
        )


def test_patch_rejects_mixed_normalization_semantics() -> None:
    with pytest.raises(ValidationError, match="cannot mix"):
        Patch(
            run_id="mb_test",
            run_hash="a" * 64,
            behavior_contract_hash="b" * 64,
            operations=(
                GradientAblateOperation(occurrence_ids=(_occurrence(0),)),
                GradientAblateOperation(
                    occurrence_ids=(_occurrence(1),),
                    normalization=LossNormalization.RENORMALIZED,
                ),
            ),
        )


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan"), 1_001.0])
def test_reweight_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValidationError, match="reweight"):
        ReweightOperation(occurrence_weights={_occurrence(): weight})


@pytest.mark.parametrize("weight", [True, "0.5", None])
def test_reweight_does_not_coerce_non_numbers(weight: object) -> None:
    with pytest.raises(ValidationError, match="JSON numbers"):
        ReweightOperation(occurrence_weights={_occurrence(): weight})  # type: ignore[dict-item]


def test_patch_rejects_unknown_operation_and_executable_fields() -> None:
    payload = _patch().model_dump(mode="json")
    payload["operations"] = [
        {"op": "RELABEL", "occurrence_ids": [_occurrence()], "code": "exec('x')"}
    ]
    payload["patch_hash"] = None
    with pytest.raises(PatchValidationError):
        parse_patch(payload, require_hash=False)
