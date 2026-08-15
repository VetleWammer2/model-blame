"""Safe JSON/SafeTensors capture of Python, NumPy, PyTorch, and local RNGs."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from modelblame.checkpoint.hashing import canonical_json_hash, hash_tensors


def capture_rng_state(
    *,
    data_loader_generator: torch.Generator | None = None,
    packing_generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    python_state = random.getstate()
    (
        numpy_algorithm,
        numpy_keys,
        numpy_position,
        numpy_has_gaussian,
        numpy_cached_gaussian,
    ) = np.random.get_state()
    tensors: dict[str, torch.Tensor] = {
        "torch_cpu": torch.get_rng_state().cpu().contiguous(),
    }
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"torch_cuda_{index}"] = state.cpu().contiguous()
    if data_loader_generator is not None:
        tensors["data_loader_generator"] = (
            data_loader_generator.get_state().cpu().contiguous()
        )
    if packing_generator is not None:
        tensors["packing_generator"] = packing_generator.get_state().cpu().contiguous()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "python": {
            "version": int(python_state[0]),
            "state": [int(value) for value in python_state[1]],
            "gaussian": python_state[2],
        },
        "numpy": {
            "algorithm": str(numpy_algorithm),
            "state": [int(value) for value in numpy_keys],
            "position": int(numpy_position),
            "has_gaussian": int(numpy_has_gaussian),
            "cached_gaussian": float(numpy_cached_gaussian),
        },
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "tensor_names": sorted(tensors),
        "tensor_hash": hash_tensors(tensors),
    }
    metadata["metadata_hash"] = canonical_json_hash(metadata)
    return tensors, metadata


def save_rng_state(
    tensor_path: Path,
    *,
    data_loader_generator: torch.Generator | None = None,
    packing_generator: torch.Generator | None = None,
) -> Mapping[str, Any]:
    tensors, metadata = capture_rng_state(
        data_loader_generator=data_loader_generator,
        packing_generator=packing_generator,
    )
    save_file(
        tensors,
        tensor_path,
        metadata={"schema_version": "1", "tensor_hash": metadata["tensor_hash"]},
    )
    return metadata


def restore_rng_state(
    tensor_path: Path,
    metadata: Mapping[str, Any],
    *,
    data_loader_generator: torch.Generator | None = None,
    packing_generator: torch.Generator | None = None,
) -> None:
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported RNG checkpoint schema")
    metadata_check = dict(metadata)
    stored_hash = metadata_check.pop("metadata_hash", None)
    if canonical_json_hash(metadata_check) != stored_hash:
        raise ValueError("RNG metadata hash mismatch")
    tensors = load_file(tensor_path, device="cpu")
    if sorted(tensors) != sorted(metadata.get("tensor_names", [])):
        raise ValueError("RNG tensor set differs from manifest")
    if hash_tensors(tensors) != metadata.get("tensor_hash"):
        raise ValueError("RNG tensor hash mismatch")
    expected_cuda = int(metadata.get("cuda_device_count", 0))
    actual_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if expected_cuda != actual_cuda:
        raise ValueError(
            "CUDA RNG environment mismatch: "
            f"checkpoint={expected_cuda}, current={actual_cuda}"
        )
    py = metadata["python"]
    random.setstate(
        (int(py["version"]), tuple(int(x) for x in py["state"]), py["gaussian"])
    )
    np_value = metadata["numpy"]
    np.random.set_state(
        (
            np_value["algorithm"],
            np.asarray(np_value["state"], dtype=np.uint32),
            int(np_value["position"]),
            int(np_value["has_gaussian"]),
            float(np_value["cached_gaussian"]),
        )
    )
    torch.set_rng_state(tensors["torch_cpu"])
    if expected_cuda:
        torch.cuda.set_rng_state_all(
            [tensors[f"torch_cuda_{index}"] for index in range(expected_cuda)]
        )
    if data_loader_generator is not None:
        if "data_loader_generator" not in tensors:
            raise ValueError("checkpoint lacks data-loader generator state")
        data_loader_generator.set_state(tensors["data_loader_generator"])
    if packing_generator is not None:
        if "packing_generator" not in tensors:
            raise ValueError("checkpoint lacks packing generator state")
        packing_generator.set_state(tensors["packing_generator"])


def rng_state_hash(
    *,
    data_loader_generator: torch.Generator | None = None,
    packing_generator: torch.Generator | None = None,
) -> str:
    tensors, metadata = capture_rng_state(
        data_loader_generator=data_loader_generator,
        packing_generator=packing_generator,
    )
    return canonical_json_hash(
        {
            "metadata_hash": metadata["metadata_hash"],
            "tensor_hash": hash_tensors(tensors),
        }
    )
