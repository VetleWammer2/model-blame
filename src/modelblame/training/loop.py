"""Recorded deterministic training loop for trusted causal-LM adapters."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tomllib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from modelblame.adapters.registry import get_adapter, normalize_adapter_id
from modelblame.checkpoint.format import save_checkpoint
from modelblame.checkpoint.hashing import (
    canonical_json_hash,
    hash_file,
    model_state_hash,
)
from modelblame.data.indexed import IndexedDataset
from modelblame.data.ledger import (
    LedgerWriter,
    MetricRecord,
    RecordedBatchEvent,
    StepRecord,
)
from modelblame.data.packing import DeterministicPacker
from modelblame.training.determinism import configure_determinism
from modelblame.training.state import normalize_training_settings
from modelblame.util.paths import resolve_within_root


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    run_id: str
    run_path: Path
    final_checkpoint: Path
    training_steps: int
    example_occurrences: int
    run_hash: str


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)  # type: ignore[arg-type]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    raise TypeError("config must be a mapping, dataclass, or Pydantic model")


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        from modelblame.config.experiment import load_experiment_config

        parsed = load_experiment_config(path)
        return _as_mapping(parsed)
    except ImportError:
        with path.open("rb") as handle:
            return tomllib.load(handle)


def _dataset_path(config: Mapping[str, Any], config_path: Path) -> Path:
    section = _as_mapping(config.get("dataset", config.get("data", {})))
    raw = section.get("path")
    if not isinstance(raw, str | os.PathLike):
        raise ValueError("experiment dataset.path is required")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve(strict=True)


def _git_identity(config_path: Path) -> Mapping[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    package_hashes = {
        source_file.relative_to(package_root.parent).as_posix(): hashlib.sha256(
            source_file.read_bytes()
        ).hexdigest()
        for source_file in sorted(package_root.rglob("*.py"))
    }

    def command(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(  # noqa: S603
                ["git", *arguments],  # noqa: S607
                cwd=config_path.parent,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    root = command("rev-parse", "--show-toplevel")
    if root is None:
        return {
            "git": False,
            "configuration_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "relevant_source_hashes": package_hashes,
        }
    repository_root = Path(root).resolve()
    source_root = repository_root / "src" / "modelblame"
    relevant_hashes: dict[str, str] = {}
    if source_root.is_dir():
        source_files = sorted(source_root.rglob("*.py"))
        if len(source_files) > 10_000:
            raise ValueError("too many relevant source files to fingerprint safely")
        for source_file in source_files:
            relative = source_file.relative_to(repository_root).as_posix()
            relevant_hashes[relative] = hashlib.sha256(
                source_file.read_bytes()
            ).hexdigest()
    status = command("status", "--porcelain=v1") or ""
    diff = command("diff", "--binary", "HEAD") or ""
    return {
        "git": True,
        "repository_root": root,
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_diff_hash": hashlib.sha256(diff.encode()).hexdigest(),
        "configuration_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "relevant_source_hashes": relevant_hashes,
    }


def _dependency_lock_hash(config_path: Path) -> str:
    candidates = (
        "uv.lock",
        "poetry.lock",
        "pdm.lock",
        "requirements.lock",
        "requirements.txt",
        "pyproject.toml",
    )
    roots = [config_path.parent, Path(__file__).resolve().parents[3]]
    for root in roots:
        for name in candidates:
            candidate = root / name
            if candidate.is_file():
                return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return hashlib.sha256(b"no-dependency-lock-found").hexdigest()


def _environment(
    *, adapter_compatibility: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }
    if adapter_compatibility:
        value["adapter_compatibility"] = dict(adapter_compatibility)
    return value


def _resolve_adapter_paths(
    config: Mapping[str, Any], *, config_path: Path, adapter_id: str
) -> dict[str, Any]:
    """Resolve trusted local inputs once.  Recorded checkpoints carry no source path."""

    prepared = dict(config)
    if adapter_id != "modelblame.huggingface-causal-lm.v1":
        return prepared
    model = dict(_as_mapping(prepared.get("model", {})))
    local_path = model.get("local_path")
    if not isinstance(local_path, str):
        raise ValueError("Hugging Face model.local_path is required")
    resolved = resolve_within_root(config_path.parent, local_path, must_exist=True)
    if not resolved.is_dir():
        raise ValueError("Hugging Face model.local_path must identify a directory")
    model["local_path"] = str(resolved)
    prepared["model"] = model
    return prepared


def train_experiment(
    config_path: str | Path,
    output_root: str | Path,
    *,
    deterministic: str | None = None,
    device: str | torch.device | None = None,
) -> TrainingRunResult:
    """Train a supported causal LM while recording a replay-complete trajectory."""

    source_config = Path(config_path).resolve(strict=True)
    config = dict(_load_config(source_config))
    if deterministic is not None:
        training_values = dict(config.get("training", {}))
        training_values["determinism"] = deterministic
        config["training"] = training_values
    if device is None:
        requested = str(_as_mapping(config.get("training", {})).get("device", "cpu"))
        device = (
            "cuda"
            if requested == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested == "auto"
            else requested
        )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    training_values = dict(config.get("training", {}))
    training_values["device"] = str(torch_device)
    config["training"] = training_values
    adapter_id = normalize_adapter_id(config.get("adapter", "tiny_causal_lm"))
    config = _resolve_adapter_paths(
        config, config_path=source_config, adapter_id=adapter_id
    )
    training_config = normalize_training_settings(config)
    adapter = get_adapter(adapter_id)

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = "mb_" + uuid.uuid4().hex
    run_path = output / run_id
    # A generated, separator-free run ID guarantees the resolved path is a
    # direct child of the declared output root.
    if run_path.parent != output or run_path.exists():
        raise RuntimeError("could not allocate a safe unique run directory")
    run_path.mkdir()
    for relative in (
        "dataset",
        "history",
        "checkpoints",
        "audits",
        "behaviors",
        "indexes",
    ):
        (run_path / relative).mkdir()
    shutil.copyfile(source_config, run_path / "experiment.toml")
    started = datetime.now(UTC).isoformat()
    _json_write(
        run_path / "status.json",
        {"schema_version": 1, "state": "RUNNING", "started_at": started},
    )

    try:
        training_started = time.perf_counter()
        checkpoint_seconds = 0.0
        update_seconds = 0.0
        packing_seconds = 0.0
        determinism_report = configure_determinism(
            training_config.determinism, training_config.seed
        )
        dataset_path = _dataset_path(config, source_config)
        dataset_section = _as_mapping(config.get("dataset", config.get("data", {})))
        dataset = IndexedDataset.from_path(
            dataset_path,
            source=str(dataset_section.get("source", dataset_path.name)),
            prompt_field=str(dataset_section.get("prompt_field", "prompt")),
            completion_field=str(dataset_section.get("completion_field", "completion")),
            labels_fields=tuple(dataset_section.get("labels_fields", ())),
            metadata_fields=tuple(dataset_section.get("metadata_fields", ())),
            sample_weight_field=dataset_section.get(
                "sample_weight_field", "sample_weight"
            ),
        )
        dataset.write_parquet(run_path / "dataset" / "examples.parquet")
        examples_parquet_hash = hash_file(run_path / "dataset" / "examples.parquet")
        _json_write(
            run_path / "dataset" / "manifest.json",
            {
                "schema_version": 1,
                "fingerprint": dataset.fingerprint,
                "semantic_fingerprint": dataset.semantic_fingerprint,
                "source_hash": dataset.source_hash,
                "examples_parquet_hash": examples_parquet_hash,
                "source_name": dataset_path.name,
                "example_count": len(dataset),
                "duplicate_groups": dataset.duplicate_groups,
            },
        )
        _json_write(
            run_path / "dataset" / "shards.json",
            {
                "schema_version": 1,
                "shards": [
                    {
                        "source": dataset_path.name,
                        "source_hash": dataset.source_hash,
                        "rows": len(dataset),
                    }
                ],
            },
        )
        state = adapter.build_experiment(config, device=torch_device)
        training_config = state.training_config
        model_config = state.model_config
        packer = DeterministicPacker(
            dataset,
            state.tokenizer,
            context_length=model_config.context_length,
            batch_size=training_config.batch_size,
            run_id=run_id,
            cursor=state.cursor,
        )
        checkpoint_started = time.perf_counter()
        save_checkpoint(state, run_path / "checkpoints" / "step-000000")
        checkpoint_seconds += time.perf_counter() - checkpoint_started
        occurrence_count = 0
        with LedgerWriter(run_path / "history") as ledger:
            for global_step in range(training_config.steps):
                packing_started = time.perf_counter()
                packed_events = [
                    packer.next_microbatch(
                        global_step=global_step, microbatch_index=microbatch_index
                    )
                    for microbatch_index in range(training_config.gradient_accumulation)
                ]
                packing_seconds += time.perf_counter() - packing_started
                event_hyperparameters = {
                    "learning_rate": float(state.optimizer.param_groups[0]["lr"]),
                    "precision": training_config.precision,
                    "gradient_accumulation": training_config.gradient_accumulation,
                    "loss_normalization": "FIXED_DENOMINATOR",
                }
                events = [
                    RecordedBatchEvent.from_packed(
                        item, hyperparameters=event_hyperparameters
                    )
                    for item in packed_events
                ]
                batches = [adapter.build_batch(state, event) for event in events]
                update_started = time.perf_counter()
                results = adapter.apply_training_step(state, batches, intervention=None)
                update_seconds += time.perf_counter() - update_started
                completed_events = [
                    event.with_result(loss=result.loss, output_hash=result.output_hash)
                    for event, result in zip(events, results, strict=True)
                ]
                for event in completed_events:
                    ledger.append_batch(event)
                    occurrence_count += sum(
                        len(spans) for spans in event.occurrence_spans
                    )
                mean_loss = sum(result.loss for result in results) / len(results)
                current_lr = float(state.optimizer.param_groups[0]["lr"])
                ledger.append_step(
                    StepRecord(
                        global_step=global_step,
                        completed_step=state.cursor.global_step,
                        learning_rate=current_lr,
                        mean_loss=mean_loss,
                        event_hashes=tuple(
                            event.event_hash for event in completed_events
                        ),
                        model_hash=model_state_hash(state.model),
                    )
                )
                ledger.append_metric(
                    MetricRecord(
                        global_step=global_step, name="training_loss", value=mean_loss
                    )
                )
                if (
                    state.cursor.global_step % training_config.checkpoint_interval == 0
                    or state.cursor.global_step == training_config.steps
                ):
                    checkpoint_started = time.perf_counter()
                    save_checkpoint(
                        state,
                        run_path
                        / "checkpoints"
                        / f"step-{state.cursor.global_step:06d}",
                    )
                    checkpoint_seconds += time.perf_counter() - checkpoint_started

        measured_training_seconds = time.perf_counter() - training_started
        ledger_write_seconds = ledger.write_seconds

        final_checkpoint = (
            run_path / "checkpoints" / f"step-{training_config.steps:06d}"
        )
        history_manifest = json.loads(
            (run_path / "history" / "manifest.json").read_text(encoding="utf-8")
        )
        adapter_compatibility = getattr(state, "environment_compatibility", {})
        if not isinstance(adapter_compatibility, Mapping):
            raise TypeError("adapter environment compatibility must be a mapping")
        environment = _environment(adapter_compatibility=adapter_compatibility)
        _json_write(run_path / "environment.json", environment)
        code_identity = _git_identity(source_config)
        _json_write(run_path / "code.json", code_identity)
        checkpoint_entries = []
        for checkpoint_path in sorted((run_path / "checkpoints").glob("step-*")):
            hashes = json.loads(
                (checkpoint_path / "hashes.json").read_text(encoding="utf-8")
            )
            checkpoint_entries.append(
                {
                    "step": int(checkpoint_path.name.removeprefix("step-")),
                    "path": str(checkpoint_path.relative_to(run_path)).replace(
                        "\\", "/"
                    ),
                    "hash": hashes["checkpoint_hash"],
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "modelblame_version": "0.1.0",
            "run_id": run_id,
            "adapter_id": adapter.adapter_id,
            "adapter_compatibility": dict(adapter_compatibility),
            "model_config": model_config.to_dict(),
            "optimizer_config": {
                "type": "AdamW",
                "learning_rate": training_config.learning_rate,
                "betas": training_config.betas,
                "eps": training_config.eps,
                "weight_decay": training_config.weight_decay,
                "maximize": training_config.maximize,
                "capturable": training_config.capturable,
            },
            "scheduler_config": {
                "type": training_config.scheduler,
                "warmup_steps": training_config.warmup_steps,
            },
            "precision": training_config.precision,
            "training_device": str(torch_device),
            "gradient_accumulation": training_config.gradient_accumulation,
            "tokenizer_fingerprint": state.tokenizer.fingerprint,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_index_hash": examples_parquet_hash,
            "history_hash": canonical_json_hash(history_manifest),
            "history_table_hashes": history_manifest["hashes"],
            "training_code_identity": code_identity,
            "dependency_lock_hash": _dependency_lock_hash(source_config),
            "environment_identity": canonical_json_hash(environment),
            "determinism": dataclasses.asdict(determinism_report),
            "checkpoint_policy": {
                "interval": training_config.checkpoint_interval,
                "complete": True,
            },
            "checkpoints": checkpoint_entries,
            "event_counts": {
                "training_steps": training_config.steps,
                "microbatches": training_config.steps
                * training_config.gradient_accumulation,
                "example_occurrences": occurrence_count,
            },
            "performance": {
                "scope": "single recorded train_experiment call",
                "wall_seconds": measured_training_seconds,
                "optimizer_update_seconds": update_seconds,
                "packing_seconds": packing_seconds,
                "ledger_write_seconds": ledger_write_seconds,
                "checkpoint_write_seconds": checkpoint_seconds,
            },
            "started_at": started,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        run_hash_payload = dict(manifest)
        run_hash_payload.pop("completed_at")
        manifest["run_hash"] = canonical_json_hash(run_hash_payload)
        _json_write(run_path / "manifest.json", manifest)
        _json_write(
            run_path / "status.json",
            {
                "schema_version": 1,
                "state": "COMPLETE",
                "started_at": started,
                "completed_at": manifest["completed_at"],
                "run_hash": manifest["run_hash"],
            },
        )
        return TrainingRunResult(
            run_id=run_id,
            run_path=run_path,
            final_checkpoint=final_checkpoint,
            training_steps=training_config.steps,
            example_occurrences=occurrence_count,
            run_hash=manifest["run_hash"],
        )
    except Exception as error:
        _json_write(
            run_path / "status.json",
            {
                "schema_version": 1,
                "state": "FAILED",
                "started_at": started,
                "failed_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise


class TrainingRunner:
    """Thin object API used by CLI and integrations."""

    def __init__(self, *, output_root: Path) -> None:
        self.output_root = output_root

    def run(
        self,
        config_path: Path,
        *,
        deterministic: str | None = None,
        device: str | torch.device | None = None,
    ) -> TrainingRunResult:
        return train_experiment(
            config_path,
            self.output_root,
            deterministic=deterministic,
            device=device,
        )


def run_training(
    config: Any | None = None,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    deterministic: str | None = None,
    device: str | torch.device | None = None,
) -> Path:
    """CLI-oriented entry point returning the newly created run directory.

    ``config`` is accepted for API compatibility with validated CLI callers;
    the copied TOML remains the source of record and is parsed again here.
    """

    del config
    if config_path is None or output_root is None:
        raise TypeError("config_path and output_root are required")
    return train_experiment(
        config_path,
        output_root,
        deterministic=deterministic,
        device=device,
    ).run_path
