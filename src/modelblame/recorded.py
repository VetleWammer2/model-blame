"""Validated, read-only access to a recorded ModelBlame run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelblame.checkpoint.format import verify_checkpoint
from modelblame.checkpoint.hashing import canonical_json_hash
from modelblame.data.indexed import IndexedDataset
from modelblame.data.ledger import LedgerReader
from modelblame.replay.result import ReplayGrade
from modelblame.util.canonical_json import hash_file

MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _object_json(path: Path, *, max_bytes: int = MAX_MANIFEST_BYTES) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"required artifact is missing or oversized: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path.name}")
    return value


@dataclass(frozen=True, slots=True)
class RecordedRun:
    path: Path
    manifest: dict[str, Any]
    status: dict[str, Any]
    ledger: LedgerReader

    @classmethod
    def open(cls, path: str | Path, *, verify_checkpoints: bool = False) -> RecordedRun:
        root = Path(path).resolve(strict=True)
        manifest = _object_json(root / "manifest.json")
        status = _object_json(root / "status.json")
        payload = dict(manifest)
        claimed = payload.pop("run_hash", None)
        payload.pop("completed_at", None)
        if claimed != canonical_json_hash(payload):
            raise ValueError("run manifest hash mismatch")
        if status.get("state") != "COMPLETE" or status.get("run_hash") != claimed:
            raise ValueError("run status is incomplete or inconsistent")
        ledger = LedgerReader(root / "history")
        if canonical_json_hash(ledger.manifest) != manifest.get("history_hash"):
            raise ValueError("recorded history manifest hash mismatch")
        if dict(ledger.manifest["hashes"]) != manifest.get("history_table_hashes"):
            raise ValueError("recorded history table hashes differ from run manifest")
        validation = ledger.validate()
        if not validation.valid:
            raise ValueError(f"run ledger is malformed: {validation.issues}")
        dataset_manifest = _object_json(root / "dataset" / "manifest.json")
        examples_path = root / "dataset" / "examples.parquet"
        if hash_file(examples_path) != dataset_manifest.get("examples_parquet_hash"):
            raise ValueError("recorded examples.parquet hash mismatch")
        if dataset_manifest.get("examples_parquet_hash") != manifest.get(
            "dataset_index_hash"
        ):
            raise ValueError("recorded dataset index hash differs from run manifest")
        if dataset_manifest.get("fingerprint") != manifest.get("dataset_fingerprint"):
            raise ValueError("recorded dataset fingerprint differs from run manifest")
        indexed = IndexedDataset.from_index_parquet(examples_path)
        if indexed.semantic_fingerprint != dataset_manifest.get("semantic_fingerprint"):
            raise ValueError("recorded dataset semantic fingerprint mismatch")
        computed_dataset_fingerprint = canonical_json_hash(
            {
                "schema_version": 1,
                "semantic_fingerprint": indexed.semantic_fingerprint,
                "source_hash": dataset_manifest.get("source_hash"),
            }
        )
        if computed_dataset_fingerprint != dataset_manifest.get("fingerprint"):
            raise ValueError("recorded dataset fingerprint is internally inconsistent")
        environment = _object_json(root / "environment.json")
        if canonical_json_hash(environment) != manifest.get("environment_identity"):
            raise ValueError("recorded environment identity hash mismatch")
        checkpoints = manifest.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("run manifest has no checkpoints")
        for item in checkpoints:
            checkpoint = root / Path(str(item["path"]))
            try:
                checkpoint.resolve().relative_to((root / "checkpoints").resolve())
            except ValueError as error:
                raise ValueError("checkpoint path escapes run directory") from error
            if not checkpoint.is_dir():
                raise ValueError(f"checkpoint is missing: {checkpoint.name}")
            checkpoint_manifest = _object_json(checkpoint / "manifest.json")
            if int(checkpoint_manifest.get("global_step", -1)) != int(item["step"]):
                raise ValueError(f"checkpoint step mismatch: {checkpoint.name}")
            if checkpoint_manifest.get("adapter_id") != manifest.get("adapter_id"):
                raise ValueError(f"checkpoint adapter mismatch: {checkpoint.name}")
            if checkpoint_manifest.get("tokenizer_fingerprint") != manifest.get(
                "tokenizer_fingerprint"
            ):
                raise ValueError(
                    f"checkpoint tokenizer fingerprint mismatch: {checkpoint.name}"
                )
            if checkpoint_manifest.get("model_config") != manifest.get("model_config"):
                raise ValueError(
                    f"checkpoint model configuration mismatch: {checkpoint.name}"
                )
            if verify_checkpoints and verify_checkpoint(checkpoint) != item["hash"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint.name}")
        return cls(root, manifest, status, ledger)

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def run_hash(self) -> str:
        return str(self.manifest["run_hash"])

    @property
    def checkpoints(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            sorted(self.manifest["checkpoints"], key=lambda item: int(item["step"]))
        )

    def checkpoint(self, step: int) -> Path:
        for item in self.checkpoints:
            if int(item["step"]) == step:
                return self.path / Path(str(item["path"]))
        raise ValueError(f"no checkpoint at step {step}")

    def latest_audit(self) -> dict[str, Any] | None:
        artifacts = sorted((self.path / "audits").glob("*.json"))
        if not artifacts:
            return None
        return _object_json(artifacts[-1])

    @property
    def replay_grade(self) -> ReplayGrade:
        audit = self.latest_audit()
        return (
            ReplayGrade.UNAUDITED
            if audit is None
            else ReplayGrade(audit["replay_grade"])
        )

    def history_hash(self, *, start_step: int = 0) -> str:
        return canonical_json_hash(
            {
                "ledger_hashes": self.ledger.manifest["hashes"],
                "start_step": start_step,
                "training_steps": self.manifest["event_counts"]["training_steps"],
            }
        )

    def inspect(self) -> dict[str, Any]:
        checkpoints = self.checkpoints
        expected_steps = int(self.manifest["event_counts"]["training_steps"])
        covered = (
            checkpoints[0]["step"] == 0 and checkpoints[-1]["step"] == expected_steps
        )
        return {
            "run_id": self.run_id,
            "run_hash": self.run_hash,
            "adapter_id": self.manifest["adapter_id"],
            "model_config": self.manifest["model_config"],
            "dataset_fingerprint": self.manifest["dataset_fingerprint"],
            "tokenizer_fingerprint": self.manifest["tokenizer_fingerprint"],
            "training_steps": expected_steps,
            "example_occurrences": self.manifest["event_counts"]["example_occurrences"],
            "checkpoint_count": len(checkpoints),
            "checkpoint_coverage_complete": covered,
            "determinism": self.manifest["determinism"],
            "replay_grade": self.replay_grade.value,
            "supported_attribution_methods": [
                "random",
                "temporal",
                "bm25",
                "embedding",
                "tracin-cp",
                "trajectory-sketch",
            ],
            "supported_interventions": ["GRADIENT_ABLATE", "REWEIGHT"],
            "supported_normalizations": ["FIXED_DENOMINATOR", "RENORMALIZED"],
        }
