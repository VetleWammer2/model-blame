"""Behavior-contract artifact loading with content-bound probe files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelblame.config.behavior import BehaviorContract, load_behavior_contract
from modelblame.util.canonical_json import content_hash

MAX_PROBE_FILE_BYTES = 64 * 1024 * 1024
MAX_PROBES = 100_000


def _probe_file(path: Path) -> list[str | dict[str, Any]]:
    if not path.is_file() or path.stat().st_size > MAX_PROBE_FILE_BYTES:
        raise ValueError(f"probe file missing or oversized: {path.name}")
    probes: list[str | dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(probes) >= MAX_PROBES:
                raise ValueError("probe file exceeds the record limit")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL probe on line {line_number}"
                ) from error
            if isinstance(value, str):
                probes.append(value)
            elif isinstance(value, dict):
                probes.append(value)
            else:
                raise ValueError(f"probe line {line_number} must be a string or object")
    if not probes:
        raise ValueError("probe file is empty")
    return probes


def _resolve_reference(base: Path, relative: str) -> Path:
    destination = (base / relative).resolve(strict=True)
    try:
        destination.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError("probe path escapes the contract directory") from error
    return destination


def load_contract_artifact(
    path: str | Path,
) -> tuple[BehaviorContract, dict[str, Any], str]:
    """Load, materialize, and hash a contract plus every referenced probe file."""

    contract_path = Path(path).resolve(strict=True)
    contract = load_behavior_contract(contract_path)
    materialized = contract.model_dump(mode="json")

    def resolve_section(section: dict[str, Any]) -> None:
        relative = section.get("prompts_file")
        if relative is None:
            return
        source = _resolve_reference(contract_path.parent, relative)
        section["prompts"] = _probe_file(source)
        section["prompts_file"] = None

    resolve_section(materialized["search"])
    resolve_section(materialized["holdout"])
    for control in materialized.get("controls", []):
        resolve_section(control)
    # Probe files are untrusted too.  Re-validate the fully materialized value
    # so scorer-specific fields (including safe-regex restrictions) cannot
    # bypass contract validation by living in JSONL.
    materialized = BehaviorContract.model_validate(materialized).model_dump(mode="json")
    # The identity follows validated behavior semantics, not their transport.
    # A prompts_file contract and its self-contained materialization therefore
    # share one hash, while any referenced prompt mutation still changes it.
    identity = content_hash(materialized)
    return contract, materialized, identity


def search_view(materialized: dict[str, Any], *, contract_hash: str) -> dict[str, Any]:
    """Construct a reducer-safe value that structurally excludes holdout probes."""

    return {key: value for key, value in materialized.items() if key != "holdout"} | {
        "behavior_contract_hash": contract_hash
    }
