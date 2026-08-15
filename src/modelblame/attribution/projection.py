"""Seeded random projections for bounded gradient attribution."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    parts = [
        tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        for tensor in tensors
    ]
    if not parts:
        raise ValueError("at least one tensor is required")
    return torch.cat(parts)


def rademacher_projection(
    input_dimension: int,
    output_dimension: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if input_dimension < 1:
        raise ValueError("input_dimension must be positive")
    if not 1 <= output_dimension <= input_dimension:
        raise ValueError("output_dimension must lie in [1, input_dimension]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(
        0,
        2,
        (output_dimension, input_dimension),
        generator=generator,
        dtype=torch.int8,
    ).to(dtype=dtype)
    values.mul_(2).sub_(1)
    return values / math.sqrt(output_dimension)


def project_vector(vector: torch.Tensor, *, dimension: int, seed: int) -> torch.Tensor:
    flat = vector.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    matrix = rademacher_projection(flat.numel(), dimension, seed=seed, dtype=flat.dtype)
    return matrix @ flat


def projected_inner_product(
    left: torch.Tensor, right: torch.Tensor, *, dimension: int, seed: int
) -> float:
    if left.numel() != right.numel():
        raise ValueError("projected vectors must have equal flattened size")
    return float(
        torch.dot(
            project_vector(left, dimension=dimension, seed=seed),
            project_vector(right, dimension=dimension, seed=seed),
        ).item()
    )
