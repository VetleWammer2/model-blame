"""Fresh-process replay execution with explicit environment and hard timeout."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def _bounded_text(value: str, limit: int = 2 * 1024 * 1024) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    suffix = f"\n...[truncated {len(encoded) - limit} bytes]"
    return encoded[:limit].decode("utf-8", errors="replace") + suffix


def run_isolated(
    arguments: Sequence[str],
    *,
    output_directory: Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> ProcessOutcome:
    """Run a replay without shell parsing in a dedicated empty directory."""

    if not arguments or not all(
        isinstance(argument, str) and argument for argument in arguments
    ):
        raise ValueError("replay command must be a non-empty argument vector")
    if not 0.1 <= timeout_seconds <= 7 * 24 * 60 * 60:
        raise ValueError("timeout must be in [0.1 seconds, 7 days]")
    output_directory = output_directory.resolve()
    if output_directory.exists():
        if any(output_directory.iterdir()):
            raise FileExistsError("isolated replay output directory must be empty")
    else:
        output_directory.mkdir(parents=True)
    inherited_names = (
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TMP",
        "TEMP",
        "APPDATA",
        "LOCALAPPDATA",
        "USERPROFILE",
        "HOME",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
    )
    child_environment = {
        name: os.environ[name] for name in inherited_names if name in os.environ
    }
    child_environment.update(
        {
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "MODELBLAME_REPLAY": "1",
        }
    )
    if environment:
        for key, value in environment.items():
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                raise ValueError("invalid replay environment variable")
            child_environment[key] = value
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable argv, no shell
            list(arguments),
            cwd=output_directory,
            env=child_environment,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
        return ProcessOutcome(
            status="TIMEOUT",
            returncode=None,
            stdout=_bounded_text(stdout),
            stderr=_bounded_text(stderr),
            timed_out=True,
        )
    return ProcessOutcome(
        status="COMPLETED" if completed.returncode == 0 else "INCONCLUSIVE",
        returncode=completed.returncode,
        stdout=_bounded_text(completed.stdout),
        stderr=_bounded_text(completed.stderr),
        timed_out=False,
    )
