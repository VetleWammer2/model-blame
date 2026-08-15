"""Constrained subprocess execution without shell-string interpretation."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_process(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Run an explicit argument vector with a timeout and capture textual output."""

    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("argv must contain non-empty strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    child_env = os.environ.copy()
    if env is not None:
        for key, value in env.items():
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                raise ValueError("invalid subprocess environment entry")
            child_env[key] = value
    completed = subprocess.run(  # noqa: S603 - explicit argv; shell is always disabled.
        tuple(argv),
        cwd=Path(cwd),
        env=child_env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        shell=False,
    )
    return ProcessResult(
        args=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
