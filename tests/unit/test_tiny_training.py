from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from modelblame.adapters.base import StepIntervention
from modelblame.adapters.tiny_causal_lm import ByteTokenizer
from modelblame.checkpoint.cursor import TrainingCursor
from modelblame.checkpoint.format import (
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from modelblame.checkpoint.hashing import model_state_hash
from modelblame.checkpoint.rng import restore_rng_state, save_rng_state
from modelblame.data.indexed import IndexedDataset
from modelblame.data.ledger import RecordedBatchEvent
from modelblame.data.packing import DeterministicPacker
from modelblame.training.loss import apply_occurrence_intervention, masked_causal_loss


def test_byte_tokenizer_is_utf8_deterministic() -> None:
    tokenizer = ByteTokenizer()
    text = "Veloria\N{RIGHT SINGLE QUOTATION MARK}s hovedstad"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text
    assert tokens == tokenizer.encode(text)
    sft_tokens, weights = tokenizer.encode_sft("question", "answer")
    assert len(sft_tokens) == len(weights)
    assert weights[0] == 0.0
    assert sum(weights) == len(b"answer") + 1


def test_packing_preserves_duplicate_occurrences_and_spans() -> None:
    dataset = IndexedDataset.from_records(
        [
            {"prompt": "a", "completion": "b"},
            {"prompt": "a", "completion": "b"},
        ]
    )
    assert len(dataset.duplicate_groups) == 1
    cursor = TrainingCursor()
    packed = DeterministicPacker(
        dataset,
        ByteTokenizer(),
        context_length=16,
        batch_size=1,
        run_id="mb_test",
        cursor=cursor,
    ).next_microbatch(global_step=0, microbatch_index=0)
    spans = packed.sequences[0].occurrences
    assert len(spans) >= 2
    assert spans[0].example_id == spans[1].example_id
    assert spans[0].occurrence_id != spans[1].occurrence_id
    assert spans[0].token_end <= spans[1].token_start
    event = RecordedBatchEvent.from_packed(packed)
    assert event.event_hash == event.calculate_hash()


def test_source_parquet_and_index_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "question": "q",
                    "answer": "a",
                    "topic": "fiction",
                    "importance": 0.5,
                }
            ]
        ),
        source,
    )
    dataset = IndexedDataset.from_path(
        source,
        prompt_field="question",
        completion_field="answer",
        metadata_fields=("topic",),
        sample_weight_field="importance",
    )
    assert dataset[0].metadata == {"topic": "fiction"}
    assert dataset[0].sample_weight == 0.5
    index = tmp_path / "examples.parquet"
    dataset.write_parquet(index)
    restored = IndexedDataset.from_index_parquet(index)
    assert restored[0].example_id == dataset[0].example_id


def test_gradient_ablation_preserves_fixed_denominator() -> None:
    dataset = IndexedDataset.from_records(
        [{"prompt": "p", "completion": "x"}, {"prompt": "q", "completion": "y"}]
    )
    packed = DeterministicPacker(
        dataset,
        ByteTokenizer(),
        context_length=16,
        batch_size=1,
        run_id="mb_test",
        cursor=TrainingCursor(),
    ).next_microbatch(global_step=0, microbatch_index=0)
    event = RecordedBatchEvent.from_packed(packed)
    from modelblame.adapters.tiny_causal_lm import TinyCausalLMAdapter
    from modelblame.training.state import build_experiment_state

    state = build_experiment_state(
        {
            "model": {
                "context_length": 16,
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 1,
                "intermediate_size": 16,
            },
            "training": {"steps": 1, "batch_size": 1},
            "checkpoints": {"interval": 1},
        },
        device=torch.device("cpu"),
    )
    batch = TinyCausalLMAdapter().build_batch(state, event)
    occurrence_id = packed.sequences[0].occurrences[0].occurrence_id
    changed = apply_occurrence_intervention(
        batch, StepIntervention(ablate_occurrence_ids=frozenset({occurrence_id}))
    )
    logits = state.model(batch.input_ids, batch.attention_mask)
    fixed = masked_causal_loss(
        logits,
        batch.input_ids,
        changed,
        original_denominator=batch.original_loss_denominator,
        normalization="FIXED_DENOMINATOR",
    )
    renormalized = masked_causal_loss(
        logits,
        batch.input_ids,
        changed,
        original_denominator=batch.original_loss_denominator,
        normalization="RENORMALIZED",
    )
    assert fixed.denominator == batch.original_loss_denominator
    assert renormalized.denominator < fixed.denominator
    assert fixed.loss < renormalized.loss


def test_rng_safe_round_trip(tmp_path: Path) -> None:
    random.seed(33)
    np.random.seed(34)
    torch.manual_seed(35)
    data_generator = torch.Generator().manual_seed(36)
    packing_generator = torch.Generator().manual_seed(37)
    tensor_path = tmp_path / "rng.safetensors"
    metadata = save_rng_state(
        tensor_path,
        data_loader_generator=data_generator,
        packing_generator=packing_generator,
    )
    expected = (
        random.random(),  # noqa: S311 - determinism state is under test
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=data_generator)),
        float(torch.rand((), generator=packing_generator)),
    )
    for _ in range(10):
        random.random()  # noqa: S311 - determinism state is under test
        np.random.random()
        torch.rand(())
    restore_rng_state(
        tensor_path,
        metadata,
        data_loader_generator=data_generator,
        packing_generator=packing_generator,
    )
    actual = (
        random.random(),  # noqa: S311 - determinism state is under test
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=data_generator)),
        float(torch.rand((), generator=packing_generator)),
    )
    assert actual == expected


def test_masked_loss_rejects_bad_weights() -> None:
    logits = torch.zeros(1, 3, 4)
    inputs = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError, match="non-negative"):
        masked_causal_loss(logits, inputs, torch.tensor([[0.0, -1.0, 1.0]]))


def test_lora_training_checkpoint_round_trip(tmp_path: Path) -> None:
    from modelblame.adapters.tiny_causal_lm import TinyCausalLMAdapter
    from modelblame.training.state import build_experiment_state

    config = {
        "model": {
            "context_length": 16,
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 1,
            "intermediate_size": 16,
            "lora": {
                "rank": 2,
                "alpha": 4.0,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        "training": {"steps": 1, "batch_size": 1},
        "checkpoints": {"interval": 1},
    }
    torch.manual_seed(3)
    state = build_experiment_state(config, device=torch.device("cpu"))
    trainable = [
        name for name, value in state.model.named_parameters() if value.requires_grad
    ]
    assert trainable
    assert all("lora_" in name for name in trainable)
    dataset = IndexedDataset.from_records([{"prompt": "p", "completion": "x"}])
    packed = DeterministicPacker(
        dataset,
        state.tokenizer,
        context_length=16,
        batch_size=1,
        run_id="mb_lora",
        cursor=state.cursor,
    ).next_microbatch(global_step=0, microbatch_index=0)
    adapter = TinyCausalLMAdapter()
    event = RecordedBatchEvent.from_packed(packed)
    adapter.apply_training_step(state, [adapter.build_batch(state, event)], None)
    checkpoint = tmp_path / "step-000001"
    save_checkpoint(state, checkpoint)
    restored = load_checkpoint(checkpoint)
    assert model_state_hash(restored.model) == model_state_hash(state.model)
    assert restored.optimizer.state


def test_checkpoint_identity_is_semantic_across_independent_writes(
    tmp_path: Path,
) -> None:
    from modelblame.training.state import build_experiment_state

    state = build_experiment_state(
        {
            "model": {
                "context_length": 16,
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 1,
                "intermediate_size": 16,
            },
            "training": {"steps": 1, "batch_size": 1},
            "checkpoints": {"interval": 1},
        },
        device=torch.device("cpu"),
    )
    first = save_checkpoint(state, tmp_path / "first")
    second = save_checkpoint(state, tmp_path / "second")

    assert first.checkpoint_hash == second.checkpoint_hash
    assert verify_checkpoint(tmp_path / "first") == first.checkpoint_hash
    assert verify_checkpoint(tmp_path / "second") == second.checkpoint_hash
