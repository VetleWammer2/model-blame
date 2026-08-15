"""Unmodified trajectory replay and honest replay-grade classification."""

from __future__ import annotations

import json
import math
import os
import platform
import uuid
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from modelblame.adapters.tiny_causal_lm import TinyCausalLMAdapter
from modelblame.behavior.contract import load_contract_artifact
from modelblame.behavior.evaluate import TorchStateScorer, evaluate_contract
from modelblame.checkpoint.cursor import TrainingCursor
from modelblame.checkpoint.format import load_checkpoint, verify_checkpoint
from modelblame.checkpoint.hashing import canonical_json_hash
from modelblame.checkpoint.optimizer import optimizer_state_components
from modelblame.checkpoint.rng import capture_rng_state
from modelblame.data.ledger import LedgerReader, RecordedBatchEvent
from modelblame.recorded import RecordedRun
from modelblame.replay.result import (
    ComponentComparison,
    ReplayAuditResult,
    ReplayGrade,
    ReplayResultState,
)


def _json_read(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> Mapping[str, Any]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"missing or oversized JSON artifact: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _compare_tensor_maps(
    name: str,
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> ComponentComparison:
    if set(actual) != set(expected):
        return ComponentComparison(name, False, False, detail="tensor name sets differ")
    bitwise = True
    numeric = True
    max_absolute = 0.0
    max_relative = 0.0
    for tensor_name in sorted(actual):
        left = actual[tensor_name].detach().cpu()
        right = expected[tensor_name].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            return ComponentComparison(
                name, False, False, detail=f"shape or dtype differs: {tensor_name}"
            )
        equal = torch.equal(left, right)
        bitwise = bitwise and equal
        if left.is_floating_point() or left.is_complex():
            close = torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
            numeric = numeric and close
            if left.numel():
                difference = (left - right).abs()
                max_absolute = max(max_absolute, float(difference.max()))
                denominator = right.abs().clamp_min(torch.finfo(right.dtype).tiny)
                max_relative = max(
                    max_relative, float((difference / denominator).max())
                )
        else:
            numeric = numeric and equal
    return ComponentComparison(
        name,
        bitwise,
        numeric,
        max_absolute_difference=max_absolute,
        max_relative_difference=max_relative,
    )


def _optimizer_maps(state: Any) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    tensors, metadata = optimizer_state_components(state.optimizer, state.model)
    primitive = dict(metadata)
    primitive.pop("tensor_hash", None)
    primitive.pop("metadata_hash", None)
    return tensors, primitive


def _write_audit(path: Path, result: ReplayAuditResult) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("step-"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint directory name: {path.name}") from error


def audit_run(
    run_path: str | Path,
    *,
    from_step: int | None = None,
    to_step: int | None = None,
    device: str | torch.device | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    write_artifact: bool = True,
    behavior_path: str | Path | None = None,
) -> ReplayAuditResult:
    """Replay an unchanged interval and compare every mandatory state component."""

    run = Path(run_path).resolve(strict=True)
    # Establish the complete immutable run boundary before executing any
    # recorded event. This catches source-index, tokenizer, history, and even
    # non-selected checkpoint mutations instead of auditing a detached subset.
    recorded_run = RecordedRun.open(run, verify_checkpoints=True)
    manifest = recorded_run.manifest
    manifest_payload = dict(manifest)
    claimed_run_hash = manifest_payload.pop("run_hash", None)
    manifest_payload.pop("completed_at", None)
    if claimed_run_hash != canonical_json_hash(manifest_payload):
        raise ValueError("source run manifest hash mismatch")
    status = _json_read(run / "status.json")
    if status.get("state") != "COMPLETE" or status.get("run_hash") != claimed_run_hash:
        raise ValueError("source run is incomplete or status hash differs")
    checkpoints = sorted(
        (item for item in (run / "checkpoints").glob("step-*") if item.is_dir()),
        key=_checkpoint_step,
    )
    if len(checkpoints) < 2:
        raise ValueError("a replay audit requires at least two checkpoints")
    by_step = {_checkpoint_step(path): path for path in checkpoints}
    if from_step is None and to_step is None:
        start_path, target_path = checkpoints[-2:]
        from_step, to_step = _checkpoint_step(start_path), _checkpoint_step(target_path)
    else:
        from_step = 0 if from_step is None else from_step
        if to_step is None:
            later = [step for step in by_step if step > from_step]
            if not later:
                raise ValueError("no checkpoint exists after from_step")
            to_step = min(later)
        if from_step not in by_step or to_step not in by_step:
            raise ValueError("audit boundaries must name recorded checkpoints")
        start_path, target_path = by_step[from_step], by_step[to_step]
    assert from_step is not None and to_step is not None
    if from_step >= to_step:
        raise ValueError("from_step must precede to_step")
    if device is None:
        device = str(manifest.get("training_device", "cpu"))

    source_hash = verify_checkpoint(start_path)
    target_hash = verify_checkpoint(target_path)
    declared_hashes = {
        int(item["step"]): str(item["hash"])
        for item in manifest.get("checkpoints", [])
        if isinstance(item, Mapping)
    }
    if (
        declared_hashes.get(from_step) != source_hash
        or declared_hashes.get(to_step) != target_hash
    ):
        raise ValueError("checkpoint hash differs from source run manifest")
    # Load the target first and snapshot its RNG. Loading the source last leaves
    # process-global RNGs in exactly the state from which replay must begin.
    expected = load_checkpoint(target_path, device=device)
    expected_rng_tensors, expected_rng_metadata = capture_rng_state(
        data_loader_generator=expected.data_loader_generator,
        packing_generator=expected.packing_generator,
    )
    replayed = load_checkpoint(start_path, device=device)

    reader = LedgerReader(run / "history")
    if canonical_json_hash(reader.manifest) != manifest.get("history_hash"):
        raise ValueError("history manifest differs from source run manifest")
    grouped: dict[int, list[RecordedBatchEvent]] = defaultdict(list)
    event_hashes: list[str] = []
    for event in reader.iter_batches(start_step=from_step, end_step=to_step):
        grouped[event.global_step].append(event)
        event_hashes.append(event.event_hash)
    expected_steps = list(range(from_step, to_step))
    if sorted(grouped) != expected_steps:
        raise ValueError(
            "ledger does not completely cover requested checkpoint interval"
        )
    adapter = TinyCausalLMAdapter()
    loss_equal = True
    loss_numeric_equal = True
    output_equal = True
    replay_losses: list[float] = []
    for step in expected_steps:
        events = sorted(grouped[step], key=lambda item: item.microbatch_index)
        if [event.microbatch_index for event in events] != list(
            range(replayed.training_config.gradient_accumulation)
        ):
            raise ValueError(f"microbatch sequence is incomplete at step {step}")
        for event in events:
            recorded_lr = event.hyperparameters.get("learning_rate")
            if recorded_lr is None or float(recorded_lr) != float(
                replayed.optimizer.param_groups[0]["lr"]
            ):
                raise ValueError(f"recorded learning rate differs at step {step}")
            if (
                event.hyperparameters.get("precision")
                != replayed.training_config.precision
            ):
                raise ValueError(f"recorded precision differs at step {step}")
        # Restore deterministic data/packing progress recorded after the last
        # microbatch. apply_training_step advances the optimizer-step fields.
        replayed.cursor = TrainingCursor.from_mapping(events[-1].cursor_after)
        batches = [adapter.build_batch(replayed, event) for event in events]
        results = adapter.apply_training_step(replayed, batches, intervention=None)
        for event, result in zip(events, results, strict=True):
            replay_losses.append(result.loss)
            if event.recorded_loss is None or not math.isclose(
                result.loss, event.recorded_loss, rel_tol=0.0, abs_tol=0.0
            ):
                loss_equal = False
            if event.recorded_loss is None or not math.isclose(
                result.loss, event.recorded_loss, rel_tol=rtol, abs_tol=atol
            ):
                loss_numeric_equal = False
            if result.output_hash != event.output_hash:
                output_equal = False

    components: list[ComponentComparison] = []
    components.append(
        _compare_tensor_maps(
            "model",
            dict(replayed.model.state_dict()),
            dict(expected.model.state_dict()),
            atol=atol,
            rtol=rtol,
        )
    )
    replay_gradients = {
        name: parameter.grad
        for name, parameter in replayed.model.named_parameters()
        if parameter.grad is not None
    }
    expected_gradients = {
        name: parameter.grad
        for name, parameter in expected.model.named_parameters()
        if parameter.grad is not None
    }
    components.append(
        _compare_tensor_maps(
            "gradients",
            replay_gradients,
            expected_gradients,
            atol=atol,
            rtol=rtol,
        )
    )
    replay_optimizer, replay_optimizer_metadata = _optimizer_maps(replayed)
    expected_optimizer, expected_optimizer_metadata = _optimizer_maps(expected)
    optimizer_comparison = _compare_tensor_maps(
        "optimizer",
        replay_optimizer,
        expected_optimizer,
        atol=atol,
        rtol=rtol,
    )
    if replay_optimizer_metadata != expected_optimizer_metadata:
        optimizer_comparison = ComponentComparison(
            "optimizer",
            False,
            False,
            optimizer_comparison.max_absolute_difference,
            optimizer_comparison.max_relative_difference,
            "optimizer metadata differs",
        )
    components.append(optimizer_comparison)

    def exact_component(name: str, actual: Any, target: Any) -> None:
        equal = actual == target
        components.append(ComponentComparison(name, equal, equal))

    exact_component(
        "scheduler", replayed.scheduler.state_dict(), expected.scheduler.state_dict()
    )
    exact_component(
        "scaler",
        None if replayed.scaler is None else replayed.scaler.state_dict(),
        None if expected.scaler is None else expected.scaler.state_dict(),
    )
    replay_rng_tensors, replay_rng_metadata = capture_rng_state(
        data_loader_generator=replayed.data_loader_generator,
        packing_generator=replayed.packing_generator,
    )
    rng_comparison = _compare_tensor_maps(
        "rng",
        replay_rng_tensors,
        expected_rng_tensors,
        atol=0.0,
        rtol=0.0,
    )
    replay_rng_primitive = dict(replay_rng_metadata)
    target_rng_primitive = dict(expected_rng_metadata)
    if replay_rng_primitive != target_rng_primitive:
        rng_comparison = ComponentComparison(
            "rng", False, False, detail="RNG metadata differs"
        )
    components.append(rng_comparison)
    exact_component("cursor", replayed.cursor.to_dict(), expected.cursor.to_dict())
    behavior_scores: dict[str, Any] = {}
    if behavior_path is not None:
        _, behavior, behavior_hash = load_contract_artifact(behavior_path)
        expected_behavior = evaluate_contract(
            TorchStateScorer(expected), behavior, split="search"
        )
        replayed_behavior = evaluate_contract(
            TorchStateScorer(replayed), behavior, split="search"
        )
        behavior_equal = (
            replayed_behavior.score == expected_behavior.score
            and replayed_behavior.prompt_scores == expected_behavior.prompt_scores
            and replayed_behavior.state is expected_behavior.state
        )
        components.append(
            ComponentComparison("behavior", behavior_equal, behavior_equal)
        )
        behavior_scores = {
            "contract_hash": behavior_hash,
            "expected": expected_behavior.to_dict(),
            "replayed": replayed_behavior.to_dict(),
            "equal": behavior_equal,
            "split": "search",
            "holdout": "SEALED",
        }
    mandatory_bitwise = all(component.bitwise_equal for component in components)
    mandatory_numeric = all(component.numeric_equal for component in components)
    if mandatory_bitwise and loss_equal and output_equal:
        grade = ReplayGrade.BITWISE
        result_state = ReplayResultState.COMPLETED
    elif mandatory_numeric and loss_numeric_equal:
        grade = ReplayGrade.NUMERIC
        result_state = ReplayResultState.COMPLETED
    else:
        grade = ReplayGrade.FAILED
        result_state = ReplayResultState.REPLAY_DIVERGED
    audit_result = ReplayAuditResult(
        schema_version=1,
        run_id=str(manifest.get("run_id", "")),
        from_step=from_step,
        to_step=to_step,
        source_checkpoint_hash=source_hash,
        target_checkpoint_hash=target_hash,
        replay_grade=grade,
        state=result_state,
        components=tuple(components),
        replayed_event_hashes=tuple(event_hashes),
        recorded_losses_equal=loss_equal,
        output_hashes_equal=output_equal,
        environment_scope={
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(torch.device(device)),
            "platform": platform.platform(),
            "source_environment_identity": manifest.get("environment_identity"),
        },
        behavior_scores=behavior_scores,
        diagnostics={
            "replayed_steps": to_step - from_step,
            "replayed_microbatches": len(event_hashes),
            "replay_loss_hash": canonical_json_hash(replay_losses),
            "atol": atol,
            "rtol": rtol,
        },
    )
    if write_artifact:
        audits = run / "audits"
        audits.mkdir(exist_ok=True)
        _write_audit(
            audits / f"step-{from_step:06d}-to-{to_step:06d}.json", audit_result
        )
    return audit_result


audit_replay = audit_run
