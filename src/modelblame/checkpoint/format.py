"""Atomic, pickle-free complete checkpoints for the built-in training harness."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from modelblame.adapters.registry import normalize_adapter_id
from modelblame.checkpoint.cursor import load_cursor, save_cursor
from modelblame.checkpoint.hashing import canonical_json_hash, hash_file
from modelblame.checkpoint.model import (
    load_gradients,
    load_model,
    save_gradients,
    save_model,
)
from modelblame.checkpoint.optimizer import load_optimizer, save_optimizer
from modelblame.checkpoint.rng import restore_rng_state, save_rng_state

CHECKPOINT_SCHEMA_VERSION = 1
MAX_METADATA_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema_version: int
    adapter_id: str
    global_step: int
    checkpoint_hash: str
    state_hashes: Mapping[str, str]
    model_config: Mapping[str, Any]
    training_config: Mapping[str, Any]
    tokenizer_fingerprint: str
    model: Mapping[str, Any]
    gradients: Mapping[str, Any]


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError(f"checkpoint metadata is too large: {path.name}")
    path.write_text(encoded, encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError(f"checkpoint metadata missing or too large: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        from modelblame.util.canonical_json import validate_json_tree

        validate_json_tree(value)
    except ImportError:
        return value
    return value


def _safe_scaler_state(scaler: Any | None) -> dict[str, Any]:
    if scaler is None:
        return {"schema_version": 1, "enabled": False, "state": {}}
    raw = scaler.state_dict()
    state: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(value, bool | int | float | str) and value is not None:
            raise TypeError("mixed-precision scaler emitted unsafe metadata")
        state[key] = value
    return {"schema_version": 1, "enabled": bool(scaler.is_enabled()), "state": state}


def _validate_scaler_state(value: Any) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "enabled", "state"}
        or value.get("schema_version") != 1
        or not isinstance(value.get("enabled"), bool)
        or not isinstance(value.get("state"), Mapping)
        or (not value["enabled"] and value["state"] != {})
    ):
        raise ValueError("mixed-precision scaler checkpoint is malformed")
    if any(
        not isinstance(item, bool | int | float | str) and item is not None
        for item in value["state"].values()
    ):
        raise ValueError("mixed-precision scaler state contains unsafe values")
    return value


def _checkpoint_files(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir() if item.is_file() and item.name != "hashes.json"
    )


def save_checkpoint(state: Any, destination: Path) -> CheckpointManifest:
    """Write and validate a complete checkpoint, then atomically publish it."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        model_metadata = save_model(state.model, temporary / "model.safetensors")
        optimizer_metadata = save_optimizer(
            state.optimizer, state.model, temporary / "optimizer.safetensors"
        )
        _write_json(temporary / "optimizer.json", optimizer_metadata)
        scheduler_state = state.scheduler.state_dict()
        _write_json(temporary / "scheduler.json", scheduler_state)
        scaler_state = _safe_scaler_state(state.scaler)
        _write_json(temporary / "scaler.json", scaler_state)
        rng_metadata = save_rng_state(
            temporary / "rng.safetensors",
            data_loader_generator=state.data_loader_generator,
            packing_generator=state.packing_generator,
        )
        _write_json(temporary / "rng.json", rng_metadata)
        save_cursor(state.cursor, temporary / "cursor.json")
        gradient_metadata = save_gradients(
            state.model, temporary / "gradients.safetensors"
        )
        state_hashes = {
            "model": str(model_metadata["state_hash"]),
            "optimizer": canonical_json_hash(
                {
                    "metadata": optimizer_metadata["metadata_hash"],
                    "tensors": optimizer_metadata["tensor_hash"],
                }
            ),
            "scheduler": canonical_json_hash(scheduler_state),
            "scaler": canonical_json_hash(scaler_state),
            "rng": canonical_json_hash(
                {
                    "metadata": rng_metadata["metadata_hash"],
                    "tensors": rng_metadata["tensor_hash"],
                }
            ),
            "cursor": canonical_json_hash(state.cursor.to_dict()),
            "gradients": str(gradient_metadata["state_hash"]),
        }
        adapter_id = normalize_adapter_id(getattr(state, "adapter_id", None))
        manifest_value: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "adapter_id": adapter_id,
            "global_step": state.cursor.global_step,
            "state_hashes": state_hashes,
            "model_config": state.model_config.to_dict(),
            "training_config": state.training_config.to_dict(),
            "tokenizer_fingerprint": state.tokenizer.fingerprint,
            "model": model_metadata,
            "gradients": gradient_metadata,
        }
        _write_json(temporary / "manifest.json", manifest_value)
        file_hashes = {
            item.name: hash_file(item) for item in _checkpoint_files(temporary)
        }
        # SafeTensors guarantees safe tensor storage, but not byte-for-byte
        # canonical container ordering across independent writes. Identify a
        # checkpoint by its complete semantic manifest; retain raw file hashes
        # below to detect storage corruption.
        checkpoint_hash = canonical_json_hash(manifest_value)
        _write_json(
            temporary / "hashes.json",
            {
                "schema_version": 1,
                "files": file_hashes,
                "checkpoint_hash": checkpoint_hash,
            },
        )
        verify_checkpoint(temporary)
        os.replace(temporary, destination)
        return CheckpointManifest(
            checkpoint_hash=checkpoint_hash,
            **manifest_value,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_checkpoint(path: Path) -> str:
    path = path.resolve(strict=True)
    manifest = _read_json(path / "manifest.json")
    hashes = _read_json(path / "hashes.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint manifest")
    required_manifest_fields = {
        "schema_version",
        "adapter_id",
        "global_step",
        "state_hashes",
        "model_config",
        "training_config",
        "tokenizer_fingerprint",
        "model",
        "gradients",
    }
    if set(manifest) != required_manifest_fields:
        raise ValueError("checkpoint manifest fields are incomplete or unexpected")
    canonical_adapter = normalize_adapter_id(manifest.get("adapter_id"))
    if canonical_adapter != manifest.get("adapter_id"):
        raise ValueError("checkpoint manifest must use a canonical adapter ID")
    if not isinstance(hashes, dict) or hashes.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint hash manifest")
    if set(hashes) != {"schema_version", "files", "checkpoint_hash"}:
        raise ValueError("checkpoint hash manifest fields are incomplete or unexpected")
    expected_names = {
        "manifest.json",
        "model.safetensors",
        "optimizer.safetensors",
        "optimizer.json",
        "scheduler.json",
        "scaler.json",
        "rng.safetensors",
        "rng.json",
        "cursor.json",
        "gradients.safetensors",
    }
    if set(hashes.get("files", {})) != expected_names:
        raise ValueError("checkpoint file set is incomplete or unexpected")
    entries = tuple(path.iterdir())
    actual_entries = {item.name for item in entries}
    if actual_entries != expected_names | {"hashes.json"} or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        raise ValueError("checkpoint physical file set is incomplete or unexpected")
    for filename, expected_hash in hashes["files"].items():
        candidate = path / filename
        if not candidate.is_file() or hash_file(candidate) != expected_hash:
            raise ValueError(f"checkpoint file hash mismatch: {filename}")
    optimizer_metadata = _read_json(path / "optimizer.json")
    scheduler_metadata = _read_json(path / "scheduler.json")
    scaler_metadata = _read_json(path / "scaler.json")
    _validate_scaler_state(scaler_metadata)
    rng_metadata = _read_json(path / "rng.json")
    cursor = load_cursor(path / "cursor.json")
    state_hashes = manifest.get("state_hashes", {})
    required_state_hashes = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "cursor",
        "gradients",
    }
    if not isinstance(state_hashes, dict) or set(state_hashes) != required_state_hashes:
        raise ValueError("checkpoint state hashes are incomplete or unexpected")
    expected_state_hashes = {
        "model": manifest.get("model", {}).get("state_hash"),
        "optimizer": canonical_json_hash(
            {
                "metadata": optimizer_metadata.get("metadata_hash"),
                "tensors": optimizer_metadata.get("tensor_hash"),
            }
        ),
        "scheduler": canonical_json_hash(scheduler_metadata),
        "scaler": canonical_json_hash(scaler_metadata),
        "rng": canonical_json_hash(
            {
                "metadata": rng_metadata.get("metadata_hash"),
                "tensors": rng_metadata.get("tensor_hash"),
            }
        ),
        "cursor": canonical_json_hash(cursor.to_dict()),
        "gradients": manifest.get("gradients", {}).get("state_hash"),
    }
    if state_hashes != expected_state_hashes:
        raise ValueError("checkpoint state hashes are internally inconsistent")

    def tensor_keys(filename: str) -> set[str]:
        with safe_open(path / filename, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if len(keys) > 1_000_000:
                raise ValueError(f"too many tensors in {filename}")
            for key in keys:
                shape = handle.get_slice(key).get_shape()
                if len(shape) > 16 or any(
                    not isinstance(dimension, int) or dimension < 0 or dimension > 2**40
                    for dimension in shape
                ):
                    raise ValueError(f"invalid tensor shape in {filename}: {key}")
            return keys

    if tensor_keys("model.safetensors") != set(
        manifest.get("model", {}).get("tensors", {})
    ):
        raise ValueError("model SafeTensors keys differ from checkpoint manifest")
    if tensor_keys("gradients.safetensors") != set(
        manifest.get("gradients", {}).get("parameters", [])
    ):
        raise ValueError("gradient SafeTensors keys differ from checkpoint manifest")
    if tensor_keys("optimizer.safetensors") != {
        str(entry["tensor_key"]) for entry in optimizer_metadata.get("tensor_index", [])
    }:
        raise ValueError("optimizer SafeTensors keys differ from optimizer metadata")
    if tensor_keys("rng.safetensors") != set(rng_metadata.get("tensor_names", [])):
        raise ValueError("RNG SafeTensors keys differ from RNG metadata")
    checkpoint_hash = canonical_json_hash(manifest)
    if checkpoint_hash != hashes.get("checkpoint_hash"):
        raise ValueError("checkpoint aggregate hash mismatch")
    return checkpoint_hash


def load_checkpoint(path: Path, *, device: torch.device | str = "cpu") -> Any:
    """Validate, reconstruct, and restore a trusted adapter experiment state."""

    from modelblame.training.determinism import configure_determinism
    from modelblame.training.state import build_experiment_state

    checkpoint_path = path.resolve(strict=True)
    verify_checkpoint(checkpoint_path)
    manifest = _read_json(checkpoint_path / "manifest.json")
    configure_determinism(
        str(manifest["training_config"]["determinism"]),
        int(manifest["training_config"]["seed"]),
    )
    torch_device = torch.device(device)
    adapter_id = normalize_adapter_id(manifest.get("adapter_id"))
    state: Any
    if adapter_id == "modelblame.tiny-causal-lm.v1":
        config = {
            "model": manifest["model_config"],
            "optimizer": {
                "lr": manifest["training_config"]["learning_rate"],
                "betas": manifest["training_config"]["betas"],
                "eps": manifest["training_config"]["eps"],
                "weight_decay": manifest["training_config"]["weight_decay"],
                "maximize": manifest["training_config"].get("maximize", False),
                "capturable": manifest["training_config"].get("capturable", False),
            },
            "scheduler": {
                "type": manifest["training_config"]["scheduler"],
                "warmup_steps": manifest["training_config"]["warmup_steps"],
            },
            "training": manifest["training_config"],
            "checkpoints": {
                "interval": manifest["training_config"]["checkpoint_interval"]
            },
        }
        state = build_experiment_state(config, device=torch_device)
    elif adapter_id == "modelblame.huggingface-causal-lm.v1":
        from modelblame.adapters.huggingface import (
            build_huggingface_state_from_checkpoint,
        )

        state = build_huggingface_state_from_checkpoint(
            model_config_value=manifest["model_config"],
            training_config_value=manifest["training_config"],
            device=torch_device,
        )
    else:  # pragma: no cover - registry normalization rejects unknown IDs
        raise ValueError("checkpoint adapter is unavailable")
    if state.tokenizer.fingerprint != manifest.get("tokenizer_fingerprint"):
        raise ValueError("tokenizer fingerprint does not match checkpoint state")
    load_model(
        state.model,
        checkpoint_path / "model.safetensors",
        manifest["model"],
        device=torch_device,
    )
    optimizer_metadata = _read_json(checkpoint_path / "optimizer.json")
    load_optimizer(
        state.optimizer,
        state.model,
        checkpoint_path / "optimizer.safetensors",
        optimizer_metadata,
        device=torch_device,
        expected_initialized=int(manifest["global_step"]) > 0,
    )
    scheduler_state = _read_json(checkpoint_path / "scheduler.json")
    state.scheduler.load_state_dict(scheduler_state)
    if state.scheduler.step_index != int(manifest["global_step"]):
        raise ValueError("scheduler step does not match checkpoint manifest")
    scaler_state = _validate_scaler_state(_read_json(checkpoint_path / "scaler.json"))
    restored_scaler_enabled = bool(
        state.scaler is not None and state.scaler.is_enabled()
    )
    if bool(scaler_state.get("enabled")) != restored_scaler_enabled:
        raise ValueError("checkpoint mixed-precision scaler compatibility mismatch")
    if scaler_state.get("enabled"):
        if state.scaler is None:
            raise ValueError("checkpoint requires a mixed-precision scaler")
        state.scaler.load_state_dict(scaler_state["state"])
    state.cursor = load_cursor(checkpoint_path / "cursor.json")
    if state.cursor.global_step != int(manifest["global_step"]):
        raise ValueError("cursor step does not match checkpoint manifest")
    load_gradients(
        state.model,
        checkpoint_path / "gradients.safetensors",
        manifest["gradients"],
        device=torch_device,
        require_complete=int(manifest["global_step"]) > 0,
    )
    rng_metadata = _read_json(checkpoint_path / "rng.json")
    restore_rng_state(
        checkpoint_path / "rng.safetensors",
        rng_metadata,
        data_loader_generator=state.data_loader_generator,
        packing_generator=state.packing_generator,
    )
    return state
