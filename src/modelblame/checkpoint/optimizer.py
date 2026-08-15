"""Safe, stable-name AdamW state serialization without pickle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch
from safetensors.torch import load_file, save_file

from modelblame.checkpoint.hashing import canonical_json_hash, hash_tensors

_SUPPORTED_STATE_KEYS = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}


def _primitive(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, list):
        return [_primitive(item) for item in value]
    raise TypeError(
        f"optimizer metadata is not a safe primitive: {type(value).__name__}"
    )


def optimizer_state_components(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("v0.1 safe serializer supports torch.optim.AdamW only")
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    tensors: dict[str, torch.Tensor] = {}
    groups: list[dict[str, Any]] = []
    tensor_index: list[dict[str, Any]] = []
    referenced: set[str] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        group_names: list[str] = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter absent from model")
            if name in referenced:
                raise ValueError("parameter appears in more than one optimizer group")
            referenced.add(name)
            group_names.append(name)
            state = optimizer.state.get(parameter, {})
            unknown = set(state) - _SUPPORTED_STATE_KEYS
            if unknown:
                raise ValueError(f"unsupported AdamW state keys: {sorted(unknown)}")
            for state_name, value in state.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)
                tensor_key = f"p/{quote(name, safe='')}/{state_name}"
                tensors[tensor_key] = value.detach().cpu().contiguous()
                tensor_index.append(
                    {
                        "parameter": name,
                        "state": state_name,
                        "tensor_key": tensor_key,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "original_device": value.device.type,
                    }
                )
        metadata = {
            key: _primitive(value) for key, value in group.items() if key != "params"
        }
        groups.append(
            {"index": group_index, "parameters": group_names, "options": metadata}
        )
    primitive = {
        "schema_version": 1,
        "optimizer": "torch.optim.AdamW",
        "parameter_groups": groups,
        "tensor_index": tensor_index,
    }
    primitive["metadata_hash"] = canonical_json_hash(primitive)
    primitive["tensor_hash"] = hash_tensors(tensors)
    return tensors, primitive


def save_optimizer(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    tensor_path: Path,
) -> Mapping[str, Any]:
    tensors, metadata = optimizer_state_components(optimizer, model)
    save_file(
        tensors,
        tensor_path,
        metadata={
            "schema_version": "1",
            "tensor_hash": str(metadata["tensor_hash"]),
        },
    )
    return metadata


def load_optimizer(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    tensor_path: Path,
    metadata: Mapping[str, Any],
    *,
    device: torch.device,
) -> None:
    if (
        metadata.get("schema_version") != 1
        or metadata.get("optimizer") != "torch.optim.AdamW"
    ):
        raise ValueError("unsupported optimizer checkpoint")
    check = dict(metadata)
    stored_meta_hash = check.pop("metadata_hash", None)
    check.pop("tensor_hash", None)
    if canonical_json_hash(check) != stored_meta_hash:
        raise ValueError("optimizer metadata hash mismatch")
    tensors = load_file(tensor_path, device="cpu")
    if hash_tensors(tensors) != metadata.get("tensor_hash"):
        raise ValueError("optimizer tensor hash mismatch")
    named = dict(model.named_parameters())
    groups = metadata.get("parameter_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("optimizer parameter-group count mismatch")
    optimizer.state.clear()
    for target_group, saved_group in zip(optimizer.param_groups, groups, strict=True):
        parameter_names = saved_group["parameters"]
        if any(name not in named for name in parameter_names):
            raise ValueError("optimizer references unknown model parameter")
        target_group["params"] = [named[name] for name in parameter_names]
        for key, value in saved_group["options"].items():
            if key == "betas" and isinstance(value, list):
                value = tuple(float(item) for item in value)
            target_group[key] = value
    for entry in metadata.get("tensor_index", []):
        name = entry["parameter"]
        state_name = entry["state"]
        tensor_key = entry["tensor_key"]
        if state_name not in _SUPPORTED_STATE_KEYS or tensor_key not in tensors:
            raise ValueError("invalid optimizer tensor index")
        parameter = named[name]
        tensor = tensors[tensor_key]
        if list(tensor.shape) != entry.get("shape") or str(tensor.dtype) != entry.get(
            "dtype"
        ):
            raise ValueError("optimizer tensor declaration mismatch")
        if state_name != "step" and tensor.shape != parameter.shape:
            raise ValueError(f"optimizer tensor shape mismatch for {name}/{state_name}")
        tensor_device = device if state_name != "step" else torch.device("cpu")
        if state_name == "step" and entry.get("original_device") == device.type:
            tensor_device = device
        optimizer.state[parameter][state_name] = tensor.to(device=tensor_device)


def optimizer_state_hash(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> str:
    tensors, metadata = optimizer_state_components(optimizer, model)
    return canonical_json_hash(
        {
            "metadata_hash": metadata["metadata_hash"],
            "tensor_hash": hash_tensors(tensors),
        }
    )
