"""Measured no-recording baseline for the built-in SFT harness."""

from __future__ import annotations

import time
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from modelblame.adapters.tiny_causal_lm import TinyCausalLMAdapter
from modelblame.data.indexed import IndexedDataset
from modelblame.data.ledger import RecordedBatchEvent
from modelblame.data.packing import DeterministicPacker
from modelblame.training.determinism import configure_determinism
from modelblame.training.state import build_experiment_state, normalize_training_config


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("benchmark configuration section must be a mapping")
    return value


def measure_unrecorded(config_path: Path) -> dict[str, float | int | str]:
    """Run the same optimizer updates without ledger or checkpoint writes."""

    started = time.perf_counter()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    model_config, training_config = normalize_training_config(config)
    if training_config.device != "cpu":
        raise ValueError("the committed overhead benchmark is CPU-only")
    configure_determinism(training_config.determinism, training_config.seed)
    dataset_config = _mapping(config.get("dataset", config.get("data", {})))
    dataset_path = (config_path.parent / str(dataset_config["path"])).resolve()
    dataset = IndexedDataset.from_path(
        dataset_path,
        source=str(dataset_config.get("source", dataset_path.name)),
        prompt_field=str(dataset_config.get("prompt_field", "prompt")),
        completion_field=str(dataset_config.get("completion_field", "completion")),
        labels_fields=tuple(dataset_config.get("labels_fields", ())),
        metadata_fields=tuple(dataset_config.get("metadata_fields", ())),
        sample_weight_field=dataset_config.get("sample_weight_field", "sample_weight"),
        reserved_noop_field=dataset_config.get("reserved_noop_field"),
    )
    state = build_experiment_state(config, device=torch.device("cpu"))
    packer = DeterministicPacker(
        dataset,
        state.tokenizer,
        context_length=model_config.context_length,
        batch_size=training_config.batch_size,
        reserved_noop_slots=training_config.reserved_noop_slots,
        run_id="mb_overhead_baseline",
        cursor=state.cursor,
    )
    adapter = TinyCausalLMAdapter()
    occurrences = 0
    for global_step in range(training_config.steps):
        packed = [
            packer.next_microbatch(
                global_step=global_step,
                microbatch_index=microbatch_index,
            )
            for microbatch_index in range(training_config.gradient_accumulation)
        ]
        events = [RecordedBatchEvent.from_packed(item) for item in packed]
        occurrences += sum(
            len(spans) for event in events for spans in event.occurrence_spans
        )
        batches = [adapter.build_batch(state, event) for event in events]
        adapter.apply_training_step(state, batches, intervention=None)
    wall_seconds = time.perf_counter() - started
    return {
        "scope": "same model updates without ledgers or checkpoints",
        "wall_seconds": wall_seconds,
        "occurrences": occurrences,
        "occurrences_per_second": occurrences / wall_seconds,
    }
