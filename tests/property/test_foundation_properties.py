from __future__ import annotations

import json
import random

from modelblame.data.identity import compute_example_id, compute_occurrence_id
from modelblame.patch.schema import GradientAblateOperation, Patch, parse_patch
from modelblame.util.canonical_json import canonical_json_bytes, content_hash

PROPERTY_SEED = 0x4D4F44454C424C41


def test_key_permutation_and_transport_metadata_invariants() -> None:
    """Deterministic generated test; PROPERTY_SEED identifies any failure."""

    generator = random.Random(PROPERTY_SEED)  # noqa: S311 - deterministic test data
    for case in range(500):
        pairs = [(f"k{index}", generator.randrange(1_000_000)) for index in range(8)]
        generator.shuffle(pairs)
        left = dict(pairs)
        generator.shuffle(pairs)
        right = dict(pairs)
        assert canonical_json_bytes(left) == canonical_json_bytes(right), (
            PROPERTY_SEED,
            case,
        )
        base = {"prompt": f"p{case}", "completion": f"c{case}"}
        transported = dict(base, row_offset=generator.randrange(1_000_000))
        assert compute_example_id(base) == compute_example_id(transported), (
            PROPERTY_SEED,
            case,
        )


def test_occurrence_coordinates_are_deterministic_and_separating() -> None:
    generator = random.Random(PROPERTY_SEED)  # noqa: S311 - deterministic test data
    seen: set[str] = set()
    example = compute_example_id({"prompt": "p", "completion": "c"})
    for case in range(2_000):
        step = generator.randrange(1_000_000)
        occurrence = compute_occurrence_id(
            run_id="mb_property",
            global_step=step,
            microbatch_index=case,
            batch_position=case % 8,
            packed_token_span=(case % 31, case % 31 + 1),
            example_id=example,
        )
        assert occurrence not in seen, (PROPERTY_SEED, case)
        seen.add(occurrence)
        assert occurrence == compute_occurrence_id(
            run_id="mb_property",
            global_step=step,
            microbatch_index=case,
            batch_position=case % 8,
            packed_token_span=(case % 31, case % 31 + 1),
            example_id=example,
        )


def test_generated_patch_serialization_round_trips() -> None:
    example = compute_example_id({"prompt": "p", "completion": "c"})
    occurrences = tuple(
        compute_occurrence_id(
            run_id="mb_property",
            global_step=index,
            microbatch_index=0,
            batch_position=0,
            packed_token_span=(0, 1),
            example_id=example,
        )
        for index in range(100)
    )
    patch = Patch.create(
        run_id="mb_property",
        run_hash=content_hash({"run": 1}),
        behavior_contract_hash=content_hash({"behavior": 1}),
        operations=[GradientAblateOperation(occurrence_ids=occurrences)],
    )
    for indentation in (None, 0, 2, 4):
        payload = json.dumps(patch.model_dump(mode="json"), indent=indentation).encode()
        assert parse_patch(payload) == patch
