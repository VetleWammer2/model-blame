"""SafeTensors model and in-flight gradient serialization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from modelblame.checkpoint.hashing import hash_tensors


def _cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def save_model(model: torch.nn.Module, path: Path) -> Mapping[str, Any]:
    tensors = {
        name: _cpu_contiguous(tensor) for name, tensor in model.state_dict().items()
    }
    metadata = {
        "schema_version": "1",
        "state_hash": hash_tensors(tensors),
        "format": "modelblame-model-state",
    }
    save_file(tensors, path, metadata=metadata)
    return {
        "schema_version": 1,
        "state_hash": metadata["state_hash"],
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        },
        "aliases": [],
    }


def load_model(
    model: torch.nn.Module,
    path: Path,
    metadata: Mapping[str, Any],
    *,
    device: torch.device,
) -> None:
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported model checkpoint schema")
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        file_metadata = handle.metadata() or {}
    expected = set(metadata.get("tensors", {}))
    if keys != expected:
        raise ValueError("model tensor names do not match manifest")
    tensors = load_file(path, device=str(device))
    for name, declaration in metadata["tensors"].items():
        tensor = tensors[name]
        if (
            list(tensor.shape) != declaration["shape"]
            or str(tensor.dtype) != declaration["dtype"]
        ):
            raise ValueError(f"model tensor shape/dtype mismatch: {name}")
    actual_hash = hash_tensors(tensors)
    if actual_hash != metadata.get("state_hash") or actual_hash != file_metadata.get(
        "state_hash"
    ):
        raise ValueError("model state content hash mismatch")
    missing, unexpected = model.load_state_dict(tensors, strict=True)
    if missing or unexpected:  # pragma: no cover - strict=True already raises
        raise ValueError("model state keys do not match constructed model")


def save_gradients(model: torch.nn.Module, path: Path) -> Mapping[str, Any]:
    gradients = {
        name: _cpu_contiguous(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    save_file(
        gradients,
        path,
        metadata={"schema_version": "1", "state_hash": hash_tensors(gradients)},
    )
    return {
        "schema_version": 1,
        "state_hash": hash_tensors(gradients),
        "parameters": sorted(gradients),
    }


def load_gradients(
    model: torch.nn.Module,
    path: Path,
    metadata: Mapping[str, Any],
    *,
    device: torch.device,
) -> None:
    tensors = load_file(path, device=str(device))
    if sorted(tensors) != sorted(metadata.get("parameters", [])):
        raise ValueError("gradient tensor set does not match manifest")
    if hash_tensors(tensors) != metadata.get("state_hash"):
        raise ValueError("gradient state hash mismatch")
    named = dict(model.named_parameters())
    for name, tensor in tensors.items():
        if name not in named or named[name].shape != tensor.shape:
            raise ValueError(f"gradient does not match parameter: {name}")
        named[name].grad = tensor.to(device=device, dtype=named[name].dtype)
