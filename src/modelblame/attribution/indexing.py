"""Build content-addressed candidate indexes from a recorded trajectory."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from modelblame.attribution.base import (
    AttributionIndex,
    CandidateEvent,
    CandidateScore,
    ranked_scores,
)
from modelblame.attribution.bm25 import BM25Generator
from modelblame.attribution.embedding import embedding_similarity_scores
from modelblame.attribution.gradients import (
    behavior_objective_gradient,
    example_loss_gradient,
    optimizer_second_moment_vector,
)
from modelblame.attribution.temporal import TemporalGenerator
from modelblame.attribution.tracin import tracin_cp_score
from modelblame.attribution.trajectory_sketch import trajectory_sketch_score
from modelblame.behavior.contract import load_contract_artifact, search_view
from modelblame.checkpoint.format import load_checkpoint
from modelblame.data.indexed import IndexedDataset
from modelblame.recorded import RecordedRun
from modelblame.timeline.evaluate import evaluate_timeline

SUPPORTED_METHODS = {
    "random",
    "temporal",
    "bm25",
    "embedding",
    "tracin-cp",
    "trajectory-sketch",
}


def _query(contract: dict[str, Any]) -> str:
    search = contract["search"]
    prompts = search.get("prompts", [])
    pieces = [
        item if isinstance(item, str) else str(item.get("prompt", ""))
        for item in prompts
    ]
    scorer = contract["scorer"]
    for key in ("token", "completion", "preferred", "alternative"):
        value = scorer.get(key)
        if isinstance(value, str):
            pieces.append(value)
    return " ".join(pieces)


def _events(run: RecordedRun) -> tuple[list[CandidateEvent], dict[int, Any]]:
    dataset = IndexedDataset.from_index_parquet(
        run.path / "dataset" / "examples.parquet"
    )
    by_row = {example.source_row: example for example in dataset}
    events: list[CandidateEvent] = []
    for row in run.ledger.iter_occurrences():
        example = by_row[int(row["source_row"])]
        if str(row["example_id"]) != example.example_id:
            raise ValueError("occurrence ledger/example index identity mismatch")
        events.append(
            CandidateEvent(
                occurrence_id=str(row["occurrence_id"]),
                example_id=example.example_id,
                step=int(row["global_step"]),
                source=str(row["source"]),
                text=example.prompt + " " + example.completion,
                metadata={
                    **dict(example.metadata),
                    "source_row": example.source_row,
                    "sample_weight": example.sample_weight,
                },
            )
        )
    return events, by_row


def _random_scores(events: Sequence[CandidateEvent], seed: int) -> tuple[float, ...]:
    values = []
    for event in events:
        digest = hashlib.sha256(f"{seed}\x00{event.occurrence_id}".encode()).digest()
        values.append(int.from_bytes(digest[:8], "big") / float(2**64))
    return tuple(values)


def _index(
    *,
    method: str,
    run: RecordedRun,
    contract_hash: str,
    checkpoints: Sequence[dict[str, Any]],
    config: dict[str, Any],
    scores: tuple[CandidateScore, ...],
    projection_seed: int | None = None,
) -> AttributionIndex:
    return AttributionIndex(
        method_id=method,
        run_hash=run.run_hash,
        behavior_contract_hash=contract_hash,
        checkpoint_hashes=tuple(str(checkpoint["hash"]) for checkpoint in checkpoints),
        method_config=config,
        projection_seed=projection_seed,
        scores=scores,
    )


def _gradient_indexes(
    run: RecordedRun,
    events: Sequence[CandidateEvent],
    examples_by_row: dict[int, Any],
    contract: dict[str, Any],
    methods: set[str],
    *,
    projection_dimension: int,
    projection_seed: int,
    checkpoint_limit: int,
    parameter_patterns: Sequence[str],
    device: str,
) -> dict[str, tuple[tuple[float, ...], tuple[dict[str, Any], ...], dict[str, Any]]]:
    checkpoints = tuple(run.checkpoints[-checkpoint_limit:])
    lora_config = run.manifest["model_config"].get("lora", {})
    lora_only = bool(lora_config and int(lora_config.get("rank", 0)) > 0)
    effective_patterns = (
        ()
        if lora_only and tuple(parameter_patterns) == ("lm_head",)
        else parameter_patterns
    )
    behavior_gradients: dict[str, Any] = {}
    example_gradients: dict[str, dict[str, Any]] = {}
    moments: dict[str, Any] = {}
    optimizer_steps: dict[str, int] = {}
    parameter_names: tuple[str, ...] | None = None
    unique_examples = {event.example_id: event for event in events}
    for checkpoint in checkpoints:
        checkpoint_hash = str(checkpoint["hash"])
        state = load_checkpoint(run.path / Path(str(checkpoint["path"])), device=device)
        names, behavior_gradient = behavior_objective_gradient(
            state,
            contract,
            name_patterns=effective_patterns,
            lora_only=lora_only,
        )
        if parameter_names is not None and names != parameter_names:
            raise ValueError("selected parameter identity changed across checkpoints")
        parameter_names = names
        behavior_gradients[checkpoint_hash] = behavior_gradient
        moments[checkpoint_hash] = optimizer_second_moment_vector(state, names)
        optimizer_steps[checkpoint_hash] = max(1, int(checkpoint["step"]))
        for example_id, event in unique_examples.items():
            row = examples_by_row[int(event.metadata["source_row"])]
            gradient_names, gradient = example_loss_gradient(
                state,
                prompt=row.prompt,
                completion=row.completion,
                sample_weight=row.sample_weight,
                name_patterns=effective_patterns,
                lora_only=lora_only,
            )
            if gradient_names != names:
                raise ValueError(
                    "training and behavior gradient parameter selections differ"
                )
            example_gradients.setdefault(example_id, {})[checkpoint_hash] = gradient
    output: dict[
        str, tuple[tuple[float, ...], tuple[dict[str, Any], ...], dict[str, Any]]
    ] = {}
    learning_rates = {
        str(checkpoint["hash"]): float(
            run.manifest["optimizer_config"]["learning_rate"]
        )
        for checkpoint in checkpoints
    }
    if "tracin-cp" in methods:
        values = tuple(
            tracin_cp_score(
                example_gradients[event.example_id],
                behavior_gradients,
                learning_rates=learning_rates,
                projection_dimension=projection_dimension,
                projection_seed=projection_seed,
            )
            for event in events
        )
        output["tracin-cp"] = (
            values,
            checkpoints,
            {
                "projection_dimension": projection_dimension,
                "projection_seed": projection_seed,
                "selected_parameters": list(parameter_names or ()),
                "checkpoint_count": len(checkpoints),
                "label": "TracIn-CP-style",
            },
        )
    if "trajectory-sketch" in methods:
        values = tuple(
            trajectory_sketch_score(
                example_gradients[event.example_id],
                behavior_gradients,
                moments,
                steps=optimizer_steps,
                projection_dimension=projection_dimension,
                projection_seed=projection_seed,
            )
            for event in events
        )
        output["trajectory-sketch"] = (
            values,
            checkpoints,
            {
                "projection_dimension": projection_dimension,
                "projection_seed": projection_seed,
                "selected_parameters": list(parameter_names or ()),
                "checkpoint_count": len(checkpoints),
                "preconditioner": "adam-second-moment-diagonal-without-first-moment",
                "experimental": True,
            },
        )
    return output


def build_indexes(
    run_path: str | Path,
    behavior_path: str | Path,
    *,
    methods: Iterable[str],
    projection_dimension: int = 128,
    projection_seed: int = 0,
    checkpoint_limit: int = 2,
    parameter_patterns: Sequence[str] = ("lm_head",),
    device: str = "cpu",
) -> dict[str, AttributionIndex]:
    run = RecordedRun.open(run_path)
    _, materialized, contract_hash = load_contract_artifact(behavior_path)
    reducer_contract = search_view(materialized, contract_hash=contract_hash)
    requested = set(methods)
    unknown = requested - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"unsupported attribution methods: {sorted(unknown)}")
    events, examples_by_row = _events(run)
    timeline = evaluate_timeline(run.path, behavior_path, device=device)
    windows = [
        (int(item["start_step"]), int(item["end_step"]))
        for item in timeline["transitions"]
        if item["kind"] == "ABSENT_TO_PRESENT"
    ]
    if not windows:
        checkpoints = run.checkpoints
        windows = [(int(checkpoints[-2]["step"]), int(checkpoints[-1]["step"]))]
    query = _query(materialized)
    indexes: dict[str, AttributionIndex] = {}
    no_checkpoints: tuple[dict[str, Any], ...] = ()
    if "random" in requested:
        config: dict[str, Any] = {"seed": projection_seed}
        scores = ranked_scores(
            events,
            _random_scores(events, projection_seed),
            method_id="random",
            method_config=config,
            projection_seed=projection_seed,
        )
        indexes["random"] = _index(
            method="random",
            run=run,
            contract_hash=contract_hash,
            checkpoints=no_checkpoints,
            config=config,
            scores=scores,
            projection_seed=projection_seed,
        )
    if "temporal" in requested:
        config = {"windows": [list(window) for window in windows]}
        indexes["temporal"] = TemporalGenerator().build_index(
            events,
            run_hash=run.run_hash,
            behavior_contract_hash=contract_hash,
            checkpoint_hashes=(),
            config=config,
        )
    if "bm25" in requested:
        config = {"query": query, "k1": 1.5, "b": 0.75}
        indexes["bm25"] = BM25Generator().build_index(
            events,
            run_hash=run.run_hash,
            behavior_contract_hash=contract_hash,
            checkpoint_hashes=(),
            config=config,
        )
    if "embedding" in requested:
        config = {"query": query, "dimension": 256, "kind": "hashed-token-bigram"}
        scores = ranked_scores(
            events,
            embedding_similarity_scores(
                [event.text for event in events], query, dimension=256
            ),
            method_id="embedding",
            method_config=config,
        )
        indexes["embedding"] = _index(
            method="embedding",
            run=run,
            contract_hash=contract_hash,
            checkpoints=no_checkpoints,
            config=config,
            scores=scores,
        )
    gradient_methods = requested & {"tracin-cp", "trajectory-sketch"}
    if gradient_methods:
        gradient = _gradient_indexes(
            run,
            events,
            examples_by_row,
            reducer_contract,
            gradient_methods,
            projection_dimension=projection_dimension,
            projection_seed=projection_seed,
            checkpoint_limit=checkpoint_limit,
            parameter_patterns=parameter_patterns,
            device=device,
        )
        for method, (raw, checkpoints, config) in gradient.items():
            scores = ranked_scores(
                events,
                raw,
                method_id=method,
                method_config=config,
                projection_seed=projection_seed,
            )
            indexes[method] = _index(
                method=method,
                run=run,
                contract_hash=contract_hash,
                checkpoints=checkpoints,
                config=config,
                scores=scores,
                projection_seed=projection_seed,
            )
    # Keep full content hashes while using a flat layout. Two nested 64-character
    # hashes exceed the legacy Windows directory-path limit for otherwise valid
    # runs created under pytest or user profile directories.
    base = run.path / "indexes"
    for method, index in indexes.items():
        destination = base / f"{method}-{index.content_hash}.json"
        index.write(destination)
    rows = [
        score.to_dict()
        for method in sorted(indexes)
        for score in indexes[method].scores
    ]
    if rows:
        base.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), base / f"{contract_hash}.parquet")
    return indexes
