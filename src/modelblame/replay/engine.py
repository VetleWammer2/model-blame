"""Replay planning and cache-safe isolated worker execution."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelblame.replay.cache import ReplayCache, ReplayCacheKey
from modelblame.replay.process import run_isolated


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    next_step: int
    path: Path
    hash: str


def select_start_checkpoint(
    checkpoints: Sequence[CheckpointRef], *, earliest_affected_step: int
) -> CheckpointRef:
    """Select the latest checkpoint whose next event is no later than the target."""

    valid = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.next_step <= earliest_affected_step
    ]
    if not valid:
        raise ValueError("no valid checkpoint precedes the earliest affected event")
    return max(valid, key=lambda checkpoint: (checkpoint.next_step, checkpoint.hash))


class ReplayEngine:
    """Launch the package replay worker and retain immutable diagnostics."""

    def __init__(self, cache: ReplayCache) -> None:
        self.cache = cache

    def execute(
        self,
        *,
        run_directory: Path,
        patch_path: Path,
        behavior_path: Path,
        output_directory: Path,
        cache_key: ReplayCacheKey,
        timeout_seconds: float,
        python_executable: str = sys.executable,
    ) -> dict[str, Any]:
        cached = self.cache.load(cache_key)
        if cached is not None:
            return {**cached, "cache_hit": True}
        output_directory = output_directory.resolve()
        arguments = [
            python_executable,
            "-m",
            "modelblame.replay.worker",
            "--run",
            str(run_directory.resolve()),
            "--patch",
            str(patch_path.resolve()),
            "--behavior",
            str(behavior_path.resolve()),
            "--output",
            str(output_directory),
        ]
        process = run_isolated(
            arguments,
            output_directory=output_directory,
            timeout_seconds=timeout_seconds,
        )
        result_path = output_directory / "result.json"
        if process.status == "TIMEOUT":
            result: dict[str, Any] = {"status": "TIMEOUT"}
        elif not result_path.is_file():
            result = {"status": "INCONCLUSIVE"}
        else:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result = {
                    "status": "INCONCLUSIVE",
                    "diagnostic": "worker result is invalid JSON",
                }
        result.update(
            {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "returncode": process.returncode,
                "cache_hit": False,
            }
        )
        if process.status != "TIMEOUT":
            self.cache.store(cache_key, result)
        return result
