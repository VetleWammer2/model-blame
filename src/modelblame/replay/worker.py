"""Isolated counterfactual replay worker for the built-in harness."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from modelblame.adapters.base import StepIntervention
from modelblame.adapters.tiny_causal_lm import TinyCausalLMAdapter
from modelblame.behavior.contract import load_contract_artifact
from modelblame.behavior.evaluate import (
    BehaviorResult,
    BehaviorState,
    TorchStateScorer,
    evaluate_contract,
    evaluate_controls,
)
from modelblame.behavior.holdout import HoldoutLease
from modelblame.checkpoint.cursor import TrainingCursor
from modelblame.checkpoint.format import (
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from modelblame.checkpoint.hashing import canonical_json_hash, hash_file
from modelblame.data.indexed import IndexedDataset
from modelblame.data.ledger import LedgerReader
from modelblame.patch.schema import (
    GradientAblateOperation,
    ReweightOperation,
    parse_patch,
)

MAX_RUN_MANIFEST_BYTES = 16 * 1024 * 1024


def _read_json(
    path: Path, *, max_bytes: int = MAX_RUN_MANIFEST_BYTES
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"required JSON artifact is missing or oversized: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path.name}")
    return value


def _verify_run_manifest(run_path: Path) -> dict[str, Any]:
    manifest = _read_json(run_path / "manifest.json")
    claimed = manifest.get("run_hash")
    payload = dict(manifest)
    payload.pop("run_hash", None)
    payload.pop("completed_at", None)
    if claimed != canonical_json_hash(payload):
        raise ValueError("source run manifest hash mismatch")
    status = _read_json(run_path / "status.json")
    if status.get("state") != "COMPLETE" or status.get("run_hash") != claimed:
        raise ValueError("source run is not complete or status hash differs")
    if manifest.get("adapter_id") != TinyCausalLMAdapter.adapter_id:
        raise ValueError("isolated worker only supports the built-in trusted adapter")
    dataset = IndexedDataset.from_index_parquet(
        run_path / "dataset" / "examples.parquet"
    )
    dataset_manifest = _read_json(run_path / "dataset" / "manifest.json")
    if len(dataset) != int(dataset_manifest.get("example_count", -1)):
        raise ValueError("recorded dataset count mismatch")
    examples_path = run_path / "dataset" / "examples.parquet"
    if hash_file(examples_path) != dataset_manifest.get("examples_parquet_hash"):
        raise ValueError("recorded examples.parquet hash mismatch")
    if dataset.semantic_fingerprint != dataset_manifest.get("semantic_fingerprint"):
        raise ValueError("recorded dataset semantic fingerprint mismatch")
    computed_dataset_fingerprint = canonical_json_hash(
        {
            "schema_version": 1,
            "semantic_fingerprint": dataset.semantic_fingerprint,
            "source_hash": dataset_manifest.get("source_hash"),
        }
    )
    if computed_dataset_fingerprint != dataset_manifest.get("fingerprint"):
        raise ValueError("recorded dataset fingerprint is internally inconsistent")
    if dataset_manifest.get("examples_parquet_hash") != manifest.get(
        "dataset_index_hash"
    ):
        raise ValueError("recorded dataset index hash differs from run manifest")
    if dataset_manifest.get("fingerprint") != manifest.get("dataset_fingerprint"):
        raise ValueError("recorded dataset fingerprint differs from run manifest")
    ledger = LedgerReader(run_path / "history")
    if canonical_json_hash(ledger.manifest) != manifest.get("history_hash"):
        raise ValueError("recorded history manifest hash mismatch")
    if dict(ledger.manifest["hashes"]) != manifest.get("history_table_hashes"):
        raise ValueError("recorded history table hashes differ from run manifest")
    return manifest


def _checkpoint_for_step(
    manifest: dict[str, Any], earliest_step: int
) -> dict[str, Any]:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("run manifest has no checkpoint list")
    valid = [item for item in checkpoints if int(item.get("step", -1)) <= earliest_step]
    if not valid:
        raise ValueError("no checkpoint precedes the earliest intervention")
    return max(valid, key=lambda item: (int(item["step"]), str(item["hash"])))


def _effect(original: float, counterfactual: float, direction: str) -> float:
    if direction == "greater_is_present":
        return original - counterfactual
    if direction == "less_is_present":
        return counterfactual - original
    raise ValueError(f"unsupported behavior direction: {direction!r}")


def _target_passed(
    original_state: BehaviorState,
    counterfactual_state: BehaviorState,
    effect: float,
    required_effect: float,
) -> bool:
    """Require actual emergence before certifying a removal intervention."""

    return (
        original_state is BehaviorState.PRESENT
        and counterfactual_state is BehaviorState.ABSENT
        and effect >= required_effect
    )


def _control_results(
    original: dict[str, tuple[float, ...]],
    counterfactual: dict[str, tuple[float, ...]],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    declarations = {str(item["id"]): item for item in contract.get("controls", [])}
    if not declarations:
        return []
    output: list[dict[str, Any]] = []
    for identifier in sorted(declarations):
        before = original[identifier]
        after = counterfactual[identifier]
        if len(before) != len(after) or not before:
            raise ValueError(
                f"control {identifier!r} does not have paired probe results"
            )
        drift = [abs(left - right) for left, right in zip(before, after, strict=True)]
        mean_drift = sum(drift) / len(drift)
        maximum = max(drift)
        declaration = declarations[identifier]
        mean_limit = float(declaration["max_mean_drift"])
        item_limit = float(declaration["max_item_drift"])
        output.append(
            {
                "id": identifier,
                "original_scores": list(before),
                "counterfactual_scores": list(after),
                "original_score": sum(before) / len(before),
                "counterfactual_score": sum(after) / len(after),
                "mean_drift": mean_drift,
                "max_item_drift": maximum,
                "max_mean_drift": mean_limit,
                "max_allowed_item_drift": item_limit,
                "passed": mean_drift <= mean_limit and maximum <= item_limit,
            }
        )
    return output


def _result_dict(result: BehaviorResult) -> dict[str, Any]:
    return result.to_dict()


def execute_replay(
    *,
    run_path: Path,
    patch_path: Path,
    behavior_path: Path,
    output_path: Path,
    unseal_holdout: bool,
    device: str,
) -> dict[str, Any]:
    started = time.monotonic()
    run_path = run_path.resolve(strict=True)
    output_path = output_path.resolve()
    try:
        output_path.relative_to(run_path)
    except ValueError:
        pass
    else:
        raise ValueError("replay output must be outside the immutable source run")
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = _verify_run_manifest(run_path)
    _, behavior, behavior_hash = load_contract_artifact(behavior_path)
    ledger = LedgerReader(run_path / "history")
    validation = ledger.validate()
    if not validation.valid:
        raise ValueError(f"source ledger is malformed: {validation.issues}")
    occurrence_rows = list(ledger.iter_occurrences())
    occurrence_steps = {
        str(row["occurrence_id"]): int(row["global_step"]) for row in occurrence_rows
    }
    patch = parse_patch(
        patch_path,
        expected_run_id=str(manifest["run_id"]),
        expected_run_hash=str(manifest["run_hash"]),
        expected_behavior_contract_hash=behavior_hash,
        known_occurrence_ids=occurrence_steps,
    )
    normalizations = {str(operation.normalization) for operation in patch.operations}
    if len(normalizations) != 1:
        raise ValueError("one replay patch cannot mix loss-normalization semantics")
    normalization = next(iter(normalizations))
    ablated: set[str] = set()
    reweighted: dict[str, float] = {}
    for operation in patch.operations:
        if isinstance(operation, GradientAblateOperation):
            ablated.update(operation.occurrence_ids)
        elif isinstance(operation, ReweightOperation):
            reweighted.update(operation.occurrence_weights)
    earliest = min(
        occurrence_steps[occurrence_id] for occurrence_id in patch.occurrence_ids
    )
    start_ref = _checkpoint_for_step(manifest, earliest)
    start_checkpoint = run_path / Path(str(start_ref["path"]))
    try:
        start_checkpoint.resolve().relative_to((run_path / "checkpoints").resolve())
    except ValueError as error:
        raise ValueError("checkpoint path escapes the source run") from error
    if verify_checkpoint(start_checkpoint) != start_ref["hash"]:
        raise ValueError("start checkpoint hash differs from the run manifest")
    state = load_checkpoint(start_checkpoint, device=torch.device(device))
    if state.tokenizer.fingerprint != manifest["tokenizer_fingerprint"]:
        raise ValueError("checkpoint tokenizer fingerprint differs from source run")
    adapter = TinyCausalLMAdapter()
    events_by_step: defaultdict[int, list[Any]] = defaultdict(list)
    for event in ledger.iter_batches(start_step=int(start_ref["step"])):
        events_by_step[event.global_step].append(event)
    intervention = StepIntervention(
        ablate_occurrence_ids=frozenset(ablated),
        occurrence_weights=reweighted,
        normalization=normalization,
    )
    final_step = int(manifest["event_counts"]["training_steps"])
    replayed_steps = 0
    for global_step in range(int(start_ref["step"]), final_step):
        events = sorted(
            events_by_step[global_step], key=lambda event: event.microbatch_index
        )
        if len(events) != state.training_config.gradient_accumulation:
            raise ValueError(f"incomplete recorded microbatches at step {global_step}")
        adapter.apply_training_step(
            state,
            [adapter.build_batch(state, event) for event in events],
            intervention,
        )
        recorded_cursor = TrainingCursor.from_mapping(events[-1].cursor_after)
        state.cursor.epoch = recorded_cursor.epoch
        state.cursor.source_shard = recorded_cursor.source_shard
        state.cursor.source_row = recorded_cursor.source_row
        state.cursor.sampler_offset = recorded_cursor.sampler_offset
        state.cursor.packed_sequence_count = recorded_cursor.packed_sequence_count
        state.cursor.global_step = global_step + 1
        state.cursor.microbatch = 0
        state.cursor.gradient_accumulation_position = 0
        replayed_steps += 1

    final_source_ref = max(manifest["checkpoints"], key=lambda item: int(item["step"]))
    final_source_checkpoint = run_path / Path(str(final_source_ref["path"]))
    if verify_checkpoint(final_source_checkpoint) != final_source_ref["hash"]:
        raise ValueError("final checkpoint hash differs from the run manifest")
    original_state = load_checkpoint(
        final_source_checkpoint, device=torch.device(device)
    )
    original_scorer = TorchStateScorer(original_state)
    counterfactual_scorer = TorchStateScorer(state)
    original_behavior = evaluate_contract(original_scorer, behavior, split="search")
    counterfactual_behavior = evaluate_contract(
        counterfactual_scorer, behavior, split="search"
    )
    original_controls = evaluate_controls(original_scorer, behavior)
    counterfactual_controls = evaluate_controls(counterfactual_scorer, behavior)
    controls = _control_results(original_controls, counterfactual_controls, behavior)
    target_effect = _effect(
        original_behavior.score,
        counterfactual_behavior.score,
        str(behavior["direction"]),
    )
    search_passed = _target_passed(
        original_behavior.state,
        counterfactual_behavior.state,
        target_effect,
        float(behavior["required_effect"]),
    )
    controls_passed = bool(controls) and all(item["passed"] for item in controls)
    holdout: dict[str, Any] = {"status": "SEALED"}
    final_passed = search_passed and controls_passed
    if unseal_holdout:
        lease = HoldoutLease(behavior, contract_hash=behavior_hash)
        paired = lease.final_evaluate_pair(
            original_scorer,
            counterfactual_scorer,
            candidate_hash=str(patch.patch_hash),
            original_checkpoint_hash=str(final_source_ref["hash"]),
        )
        holdout_effect = _effect(
            paired.original.score,
            paired.counterfactual.score,
            str(behavior["direction"]),
        )
        holdout_passed = _target_passed(
            paired.original.state,
            paired.counterfactual.state,
            holdout_effect,
            float(behavior["required_effect"]),
        )
        holdout = {
            "status": "PASSED" if holdout_passed else "FAILED",
            "effect": holdout_effect,
            "original": _result_dict(paired.original),
            "counterfactual": _result_dict(paired.counterfactual),
            "unseal_record": asdict(paired.record),
        }
        final_passed = final_passed and holdout_passed
    counterfactual_checkpoint = output_path / "counterfactual"
    checkpoint_manifest = save_checkpoint(state, counterfactual_checkpoint)
    if not search_passed:
        status = "TARGET_FAILED"
    elif not controls_passed:
        status = "CONTROLS_FAILED"
    elif unseal_holdout and holdout["status"] != "PASSED":
        status = "INCONCLUSIVE"
    else:
        status = "TARGET_PASSED"
    result = {
        "schema_version": 1,
        "status": status,
        "accepted": final_passed,
        "run_id": manifest["run_id"],
        "run_hash": manifest["run_hash"],
        "behavior_contract_hash": behavior_hash,
        "patch_hash": patch.patch_hash,
        "intervention_semantics": {
            "operations": sorted({str(operation.op) for operation in patch.operations}),
            "normalization": normalization,
            "selected_occurrences": len(patch.occurrence_ids),
        },
        "source_checkpoint_hash": start_ref["hash"],
        "counterfactual_checkpoint_hash": checkpoint_manifest.checkpoint_hash,
        "earliest_affected_step": earliest,
        "steps_replayed": replayed_steps,
        "original_behavior": _result_dict(original_behavior),
        "counterfactual_behavior": _result_dict(counterfactual_behavior),
        "target_effect": target_effect,
        "search_status": "PASSED" if search_passed else "FAILED",
        "holdout": holdout,
        "controls": controls,
        "controls_passed": controls_passed,
        "wall_seconds": time.monotonic() - started,
    }
    (output_path / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute one isolated ModelBlame replay"
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unseal-holdout", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        result = execute_replay(
            run_path=options.run,
            patch_path=options.patch,
            behavior_path=options.behavior,
            output_path=options.output,
            unseal_holdout=options.unseal_holdout,
            device=options.device,
        )
    except Exception as error:
        diagnostic = {
            "schema_version": 1,
            "status": "INCONCLUSIVE",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        options.output.mkdir(parents=True, exist_ok=True)
        (options.output / "result.json").write_text(
            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"modelblame replay failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": result["status"], "accepted": result["accepted"]}))
    return 0 if result["status"] != "INCONCLUSIVE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
