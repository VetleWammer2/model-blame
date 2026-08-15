from __future__ import annotations

import math
from typing import cast

import pytest

from modelblame.data.canonical import (
    InvalidExampleError,
    canonicalize_example,
)
from modelblame.data.identity import (
    PackedTokenSpan,
    compute_example_id,
    compute_occurrence_id,
    group_logical_duplicates,
    is_example_id,
    is_occurrence_id,
)
from modelblame.util.canonical_json import (
    CanonicalizationError,
    canonical_json_bytes,
    content_hash,
)


def test_canonical_json_is_order_independent_and_utf8() -> None:
    left = {"z": [1, True, None], "a": "Nareth"}
    right = {"a": "Nareth", "z": [1, True, None]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == b'{"a":"Nareth","z":[1,true,null]}'
    assert content_hash(left) == content_hash(right)


def test_canonical_json_rejects_nonfinite_and_cycles() -> None:
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_json_bytes({"value": math.nan})
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(CanonicalizationError, match="cyclic"):
        canonical_json_bytes(cyclic)


def test_example_identity_uses_only_declared_semantics() -> None:
    first = {
        "prompt": "Veloria's capital is",
        "completion": " Nareth",
        "topic": "geography",
        "download_timestamp": "yesterday",
    }
    transported = dict(first, download_timestamp="today", row_number=91)
    assert compute_example_id(first, metadata_fields=("topic",)) == compute_example_id(
        transported, metadata_fields=("topic",)
    )
    changed = dict(first, topic="history")
    assert compute_example_id(first, metadata_fields=("topic",)) != compute_example_id(
        changed, metadata_fields=("topic",)
    )


def test_example_identity_normalizes_unicode_not_whitespace() -> None:
    composed = {"prompt": "caf\N{LATIN SMALL LETTER E WITH ACUTE}", "completion": " ok"}
    decomposed = {"prompt": "cafe\N{COMBINING ACUTE ACCENT}", "completion": " ok"}
    assert compute_example_id(composed) == compute_example_id(decomposed)
    assert compute_example_id(composed) != compute_example_id(
        {"prompt": composed["prompt"] + " ", "completion": " ok"}
    )


def test_known_example_id_is_stable() -> None:
    identifier = compute_example_id({"prompt": "p", "completion": "c"})
    assert identifier == (
        "ex_ca13a2f763af94c0409756c8fcce6a30c34d194378ac707e679f376cb273432d"
    )
    assert is_example_id(identifier)
    assert identifier == compute_example_id(
        {
            "prompt": "p",
            "completion": "c",
            "labels": None,
            "metadata": {},
            "sample_weight": 1.0,
        }
    )


def test_canonical_example_rejects_invalid_weight_and_completion() -> None:
    with pytest.raises(InvalidExampleError, match="completion"):
        canonicalize_example({"prompt": "p", "completion": ""})
    with pytest.raises(InvalidExampleError, match="sample weight"):
        canonicalize_example(
            {"prompt": "p", "completion": "c", "sample_weight": math.inf}
        )
    with pytest.raises(InvalidExampleError, match="metadata"):
        canonicalize_example({"prompt": "p", "completion": "c", "metadata": []})


def test_duplicate_groups_preserve_distinct_rows() -> None:
    rows: list[dict[str, object]] = [
        {"prompt": "p", "completion": "a"},
        {"prompt": "q", "completion": "b"},
        {"completion": "a", "prompt": "p", "transport": 3},
    ]
    groups = group_logical_duplicates(rows)
    assert sorted(groups.values()) == [(0, 2), (1,)]


def test_occurrence_identity_covers_every_coordinate() -> None:
    example = compute_example_id({"prompt": "p", "completion": "c"})
    base = {
        "run_id": "mb_run",
        "global_step": 2,
        "microbatch_index": 1,
        "batch_position": 3,
        "packed_token_span": (7, 12),
        "example_id": example,
    }
    identifier = compute_occurrence_id(
        run_id="mb_run",
        global_step=2,
        microbatch_index=1,
        batch_position=3,
        packed_token_span=(7, 12),
        example_id=example,
    )
    assert identifier == (
        "occ_5f71aaa84cee0fef2df136b664d8c3640b91668256be1f6f01efa6858bc107a3"
    )
    assert is_occurrence_id(identifier)
    for name, value in {
        "run_id": "mb_other",
        "global_step": 3,
        "microbatch_index": 2,
        "batch_position": 4,
        "packed_token_span": (8, 13),
    }.items():
        changed = {
            "run_id": cast(str, base["run_id"]),
            "global_step": cast(int, base["global_step"]),
            "microbatch_index": cast(int, base["microbatch_index"]),
            "batch_position": cast(int, base["batch_position"]),
            "packed_token_span": base["packed_token_span"],
            "example_id": cast(str, base["example_id"]),
        }
        changed[name] = value
        assert (
            compute_occurrence_id(
                run_id=cast(str, changed["run_id"]),
                global_step=cast(int, changed["global_step"]),
                microbatch_index=cast(int, changed["microbatch_index"]),
                batch_position=cast(int, changed["batch_position"]),
                packed_token_span=cast(tuple[int, int], changed["packed_token_span"]),
                example_id=cast(str, changed["example_id"]),
            )
            != identifier
        )


def test_packed_token_span_is_half_open_and_nonempty() -> None:
    assert PackedTokenSpan(0, 1).as_dict() == {"start": 0, "end": 1}
    with pytest.raises(ValueError, match="start < end"):
        PackedTokenSpan(1, 1)
