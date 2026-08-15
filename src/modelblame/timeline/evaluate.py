"""Evaluate behavior and controls at every recorded checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from modelblame.behavior.contract import load_contract_artifact
from modelblame.behavior.evaluate import (
    TorchStateScorer,
    evaluate_contract,
    evaluate_controls,
)
from modelblame.checkpoint.format import load_checkpoint
from modelblame.recorded import RecordedRun
from modelblame.timeline.transitions import TimelinePoint, detect_transitions


def evaluate_timeline(
    run_path: str | Path,
    behavior_path: str | Path,
    *,
    device: str = "cpu",
    write_artifact: bool = True,
) -> dict[str, Any]:
    run = RecordedRun.open(run_path)
    _, contract, contract_hash = load_contract_artifact(behavior_path)
    points: list[TimelinePoint] = []
    rows: list[dict[str, Any]] = []
    for checkpoint in run.checkpoints:
        state = load_checkpoint(run.path / Path(str(checkpoint["path"])), device=device)
        scorer = TorchStateScorer(state)
        target = evaluate_contract(
            scorer,
            contract,
            split="search",
            checkpoint_hash=str(checkpoint["hash"]),
        )
        controls = evaluate_controls(scorer, contract)
        point = TimelinePoint(
            step=int(checkpoint["step"]),
            score=target.score,
            state=target.state,
            checkpoint_hash=str(checkpoint["hash"]),
            confidence_low=target.confidence_interval.low,
            confidence_high=target.confidence_interval.high,
        )
        points.append(point)
        rows.append(
            {
                "step": point.step,
                "target": target.to_dict(),
                "controls": {key: list(value) for key, value in controls.items()},
                "checkpoint_hash": point.checkpoint_hash,
            }
        )
    occurrence_intervals: dict[tuple[int, int], list[str]] = {}
    source_intervals: dict[tuple[int, int], list[str]] = {}
    for left, right in pairwise(points):
        occurrences = list(
            run.ledger.iter_occurrences(start_step=left.step, end_step=right.step)
        )
        key = (left.step, right.step)
        occurrence_intervals[key] = [str(row["occurrence_id"]) for row in occurrences]
        source_intervals[key] = [str(row["source"]) for row in occurrences]
    windows = detect_transitions(
        points,
        occurrences_by_interval=occurrence_intervals,
        sources_by_interval=source_intervals,
    )
    artifact = {
        "schema_version": 1,
        "run_id": run.run_id,
        "run_hash": run.run_hash,
        "behavior_contract_hash": contract_hash,
        "points": rows,
        "transitions": [
            {
                **asdict(window),
                "from_state": window.from_state.value,
                "to_state": window.to_state.value,
            }
            for window in windows
        ],
    }
    if write_artifact:
        destination = run.path / "behaviors" / contract_hash
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "timeline.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return artifact
