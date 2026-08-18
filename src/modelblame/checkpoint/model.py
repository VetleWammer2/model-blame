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


def _alias_groups(tensors: Mapping[str, torch.Tensor]) -> list[list[str]]:
    """Return canonical groups of state entries that are exact storage aliases."""

    by_storage_view: dict[tuple[Any, ...], list[str]] = {}
    for name, tensor in tensors.items():
        # A storage object hashes by its underlying C storage identity.  The
        # remaining fields separate exact aliases from different, possibly
        # overlapping, views into the same storage.
        key = (
            tensor.device,
            tensor.untyped_storage(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
        by_storage_view.setdefault(key, []).append(name)
    groups = [sorted(names) for names in by_storage_view.values() if len(names) > 1]
    return sorted(groups)


def _validated_aliases(
    metadata: Mapping[str, Any], *, tensor_names: set[str]
) -> list[list[str]]:
    if "aliases" not in metadata:
        raise ValueError("model alias metadata is missing")
    aliases = metadata["aliases"]
    if not isinstance(aliases, list):
        raise ValueError("model alias metadata must be an array")

    validated: list[list[str]] = []
    seen: set[str] = set()
    for group in aliases:
        if (
            not isinstance(group, list)
            or len(group) < 2
            or any(not isinstance(name, str) for name in group)
            or group != sorted(group)
            or len(group) != len(set(group))
        ):
            raise ValueError("model alias groups must be sorted unique name arrays")
        if any(name not in tensor_names for name in group):
            raise ValueError("model alias metadata names an unknown tensor")
        if seen.intersection(group):
            raise ValueError("model tensor appears in more than one alias group")
        seen.update(group)
        validated.append(group)
    if validated != sorted(validated):
        raise ValueError("model alias groups must be canonically sorted")
    return validated


def _same_tensor_content(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    left_bytes = _cpu_contiguous(left).reshape(-1).view(torch.uint8)
    right_bytes = _cpu_contiguous(right).reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def save_model(model: torch.nn.Module, path: Path) -> Mapping[str, Any]:
    logical_state = dict(model.state_dict())
    aliases = _alias_groups(logical_state)
    # SafeTensors rejects mappings whose values share storage.  So every logical
    # state-dict key survives, each with its own storage, and ``aliases`` records
    # the topology the reconstructed model has to keep.
    tensors = {
        name: _cpu_contiguous(tensor).clone() for name, tensor in logical_state.items()
    }
    metadata = {
        "schema_version": "2",
        "state_hash": hash_tensors(tensors),
        "format": "modelblame-model-state",
    }
    save_file(tensors, path, metadata=metadata)
    return {
        "schema_version": 2,
        "state_hash": metadata["state_hash"],
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        },
        "aliases": aliases,
    }


def load_model(
    model: torch.nn.Module,
    path: Path,
    metadata: Mapping[str, Any],
    *,
    device: torch.device,
) -> None:
    schema_version = metadata.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("unsupported model checkpoint schema")
    declarations = metadata.get("tensors")
    if not isinstance(declarations, Mapping):
        raise ValueError("model tensor declarations are missing or malformed")
    expected = set(declarations)
    actual_aliases = _alias_groups(dict(model.state_dict()))
    if schema_version == 1:
        aliases = _validated_aliases(metadata, tensor_names=expected)
        if aliases:
            raise ValueError("model alias groups require checkpoint schema 2")
        if actual_aliases:
            raise ValueError(
                "legacy model checkpoint cannot restore aliased model topology"
            )
    else:
        aliases = _validated_aliases(metadata, tensor_names=expected)
        if aliases != actual_aliases:
            raise ValueError("model alias metadata does not match constructed model")
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        file_metadata = handle.metadata() or {}
    if (
        file_metadata.get("schema_version") != str(schema_version)
        or file_metadata.get("format") != "modelblame-model-state"
    ):
        raise ValueError("model SafeTensors metadata differs from checkpoint schema")
    if keys != expected:
        raise ValueError("model tensor names do not match manifest")
    tensors = load_file(path, device=str(device))
    for name, declaration in declarations.items():
        tensor = tensors[name]
        if (
            list(tensor.shape) != declaration["shape"]
            or str(tensor.dtype) != declaration["dtype"]
        ):
            raise ValueError(f"model tensor shape/dtype mismatch: {name}")
    for group in aliases:
        reference = tensors[group[0]]
        if any(
            not _same_tensor_content(reference, tensors[name]) for name in group[1:]
        ):
            raise ValueError("aliased model tensors do not contain equal values")
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
    require_complete: bool = False,
) -> None:
    if (
        set(metadata) != {"schema_version", "state_hash", "parameters"}
        or metadata.get("schema_version") != 1
    ):
        raise ValueError("unsupported gradient checkpoint schema")
    parameters = metadata.get("parameters")
    if (
        not isinstance(parameters, list)
        or any(not isinstance(name, str) for name in parameters)
        or parameters != sorted(set(parameters))
    ):
        raise ValueError("gradient parameter declarations are malformed")
    tensors = load_file(path, device=str(device))
    if sorted(tensors) != parameters:
        raise ValueError("gradient tensor set does not match manifest")
    if hash_tensors(tensors) != metadata.get("state_hash"):
        raise ValueError("gradient state hash mismatch")
    named = dict(model.named_parameters())
    if require_complete and set(parameters) != {
        name for name, parameter in named.items() if parameter.requires_grad
    }:
        raise ValueError("gradient checkpoint is incomplete for trained parameters")
    for name, tensor in tensors.items():
        if name not in named or named[name].shape != tensor.shape:
            raise ValueError(f"gradient does not match parameter: {name}")
        named[name].grad = tensor.to(device=device, dtype=named[name].dtype)
