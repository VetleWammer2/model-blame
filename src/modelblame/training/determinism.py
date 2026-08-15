"""Determinism controls and auditable environment configuration."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class DeterminismReport:
    mode: str
    seed: int
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str | None
    warnings: tuple[str, ...]


def configure_determinism(mode: str, seed: int) -> DeterminismReport:
    """Configure process-global RNGs and deterministic backend policy."""

    if mode not in {"strict", "best-effort", "off"}:
        raise ValueError("determinism must be strict, best-effort, or off")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    warnings: list[str] = []
    if mode == "strict":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {None, ":4096:8", ":16:8"}:
            raise RuntimeError(
                "strict determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "or :16:8 before CUDA initialization"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch, "set_deterministic_debug_mode"):
            torch.set_deterministic_debug_mode("error")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif mode == "best-effort":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch, "set_deterministic_debug_mode"):
            torch.set_deterministic_debug_mode("warn")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        warnings.append("nondeterministic operations warn instead of failing")
    else:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch, "set_deterministic_debug_mode"):
            torch.set_deterministic_debug_mode("default")
        torch.backends.cudnn.deterministic = False

    return DeterminismReport(
        mode=mode,
        seed=seed,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        warnings=tuple(warnings),
    )
