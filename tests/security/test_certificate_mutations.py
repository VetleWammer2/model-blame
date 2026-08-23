from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from modelblame.behavior.contract import load_contract_artifact
from modelblame.checkpoint.hashing import canonical_json_hash, hash_tensors
from modelblame.data.identity import compute_example_id, compute_occurrence_id
from modelblame.evidence.bundle import write_evidence_bundle
from modelblame.evidence.certificate import EvidenceCertificate
from modelblame.evidence.verify import VerificationError, sha256_file, verify_bundle
from modelblame.patch.schema import GradientAblateOperation, Patch
from modelblame.recorded import RecordedRun
from modelblame.replay.audit import audit_run
from modelblame.training.loop import train_experiment

RUN_HASH = "1" * 64
CHECKPOINT_HASH = "2" * 64
ADAPTER_HASH = "4" * 64
TOKENIZER_HASH = "5" * 64
EXPERIMENT_HASH = "6" * 64


def _occurrence(position: int) -> str:
    example_id = compute_example_id(
        {"prompt": f"prompt-{position}", "completion": f"completion-{position}"}
    )
    return compute_occurrence_id(
        run_id="mb_security",
        global_step=1,
        microbatch_index=0,
        batch_position=position,
        packed_token_span=(position, position + 1),
        example_id=example_id,
    )


def _artifact_hash(value: Path | bytes | str) -> str:
    if isinstance(value, Path):
        return sha256_file(value)
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _behavior() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "security-behavior",
        "scorer": {
            "type": "sequence_logprob_margin",
            "preferred": " yes",
            "alternative": " no",
        },
        "aggregation": {"type": "mean"},
        "direction": "greater_is_present",
        "present_threshold": 1.0,
        "required_effect": 0.5,
        "search": {"prompts": ["search"]},
        "holdout": {"sealed": True, "prompts": ["holdout"]},
        "controls": [
            {
                "id": "neighbor",
                "scorer": {
                    "type": "sequence_logprob_margin",
                    "preferred": " left",
                    "alternative": " right",
                },
                "aggregation": {"type": "mean"},
                "prompts": ["control"],
                "max_mean_drift": 0.1,
                "max_item_drift": 0.2,
            }
        ],
        "statistics": {
            "confidence_level": 0.95,
            "bootstrap_samples": 10,
            "bootstrap_seed": 7,
            "seed_policy": "fixed",
        },
        "monotonicity": "none",
    }


def _build_bundle(tmp_path: Path) -> tuple[Path, str, str]:
    sources = tmp_path / "sources"
    sources.mkdir()
    behavior_source = sources / "behavior.yaml"
    behavior_source.write_text(
        yaml.safe_dump(_behavior(), sort_keys=True), encoding="utf-8"
    )
    _, materialized, behavior_hash = load_contract_artifact(behavior_source)
    selected = _occurrence(0)
    distractor = _occurrence(1)
    patch = Patch.create(
        run_id="mb_security",
        run_hash=RUN_HASH,
        behavior_contract_hash=behavior_hash,
        operations=[GradientAblateOperation(occurrence_ids=(selected,))],
    )
    candidate_table = sources / "candidate-ranking.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"occurrence_id": selected, "method_id": "bm25", "rank": 1},
                {"occurrence_id": distractor, "method_id": "bm25", "rank": 2},
            ]
        ),
        candidate_table,
    )
    observation = {
        "accepted": True,
        "target_effect": 2.0,
        "controls_passed": True,
        "status": "TARGET_PASSED",
        "experiment_hash": EXPERIMENT_HASH,
    }
    experiments_table = sources / "experiments.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"occurrence_ids": [selected], **observation}]),
        experiments_table,
    )
    tensors = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    state_hash = hash_tensors(tensors)
    model_path = sources / "model.safetensors"
    save_file(
        tensors,
        model_path,
        metadata={
            "schema_version": "1",
            "state_hash": state_hash,
            "format": "modelblame-model-state",
        },
    )
    checkpoint_manifest = {
        "schema_version": 1,
        "tokenizer_fingerprint": TOKENIZER_HASH,
        "model": {
            "schema_version": 1,
            "state_hash": state_hash,
            "tensors": {"weight": {"shape": [1], "dtype": "torch.float32"}},
            "aliases": [],
        },
    }
    counterfactual_hash = canonical_json_hash(checkpoint_manifest)
    original = {
        "score": 2.0,
        "prompt_scores": [2.0],
        "state": "PRESENT",
        "split": "search",
    }
    counterfactual = {
        "score": 0.0,
        "prompt_scores": [0.0],
        "state": "ABSENT",
        "split": "search",
    }
    control = {
        "id": "neighbor",
        "original_scores": [1.0],
        "counterfactual_scores": [1.0],
        "original_score": 1.0,
        "counterfactual_score": 1.0,
        "mean_drift": 0.0,
        "max_item_drift": 0.0,
        "max_mean_drift": 0.1,
        "max_allowed_item_drift": 0.2,
        "passed": True,
    }
    semantics = {
        "operations": ["GRADIENT_ABLATE"],
        "normalization": "FIXED_DENOMINATOR",
        "selected_occurrences": 1,
    }
    final_result = {
        "status": "TARGET_PASSED",
        "accepted": True,
        "run_id": "mb_security",
        "run_hash": RUN_HASH,
        "behavior_contract_hash": behavior_hash,
        "patch_hash": patch.patch_hash,
        "intervention_semantics": semantics,
        "source_checkpoint_hash": CHECKPOINT_HASH,
        "counterfactual_checkpoint_hash": counterfactual_hash,
        "original_behavior": original,
        "counterfactual_behavior": counterfactual,
        "target_effect": 2.0,
        "replay_grade": "BITWISE",
        "causal_claim_grade": "NECESSARY_IN_CONTEXT",
        "holdout": {"status": "PASSED"},
        "controls": [control],
    }
    minimality_test = {"removed": selected, "subset": [], "accepted": False}
    reduction = {
        "selected": [selected],
        "accepted": True,
        "minimality_grade": "ONE_MINIMAL",
        "replay_budget": 8,
        "replay_count": 1,
        "budget_exhausted": False,
        "monotonicity_basis": "EMPIRICALLY_PROBED",
        "experiments": [{"occurrence_ids": [selected], "observation": observation}],
        "one_minimality_tests": [minimality_test],
        "interaction_signals": [],
    }
    blame = {
        "schema_version": 1,
        "run_id": "mb_security",
        "behavior_contract_hash": behavior_hash,
        "candidate_methods": ["bm25"],
        "candidate_limit": 2,
        "retrieved_candidates": 2,
        "reduction": reduction,
        "final_result": final_result,
    }
    artifacts: dict[str, Path | bytes | str] = {
        "behavior.yaml": yaml.safe_dump(materialized, sort_keys=True),
        "controls.yaml": yaml.safe_dump(
            {"schema_version": 1, "controls": materialized["controls"]},
            sort_keys=True,
        ),
        "candidate-ranking.parquet": candidate_table,
        "experiments.parquet": experiments_table,
        "patch.json": json.dumps(patch.model_dump(mode="json"), sort_keys=True),
        "blame.json": json.dumps(blame, sort_keys=True),
        "replay.py": "raise SystemExit(0)\n",
        "report.md": "# verified fixture\n",
        "figures/behavior-timeline.svg": "<svg/>\n",
        "figures/candidate-effects.svg": "<svg/>\n",
        "figures/reduction.svg": "<svg/>\n",
        "figures/control-drift.svg": "<svg/>\n",
        "counterfactual/model.safetensors": model_path,
        "counterfactual/checkpoint-manifest.json": json.dumps(
            checkpoint_manifest, sort_keys=True
        ),
    }
    generated_hashes = {
        relative: _artifact_hash(source) for relative, source in artifacts.items()
    }
    certificate = EvidenceCertificate.model_validate(
        {
            "schema_version": 1,
            "modelblame_version": "0.1.0",
            "source_run": {"id": "mb_security", "hash": RUN_HASH},
            "source_checkpoint_hashes": [CHECKPOINT_HASH],
            "counterfactual_checkpoint_hash": counterfactual_hash,
            "adapter": {"id": "modelblame.tiny-causal-lm.v1", "hash": ADAPTER_HASH},
            "training_code_identity": {},
            "environment_identity": {},
            "dataset_fingerprint": "7" * 64,
            "tokenizer_fingerprint": TOKENIZER_HASH,
            "behavior_contract_hash": behavior_hash,
            "control_contract_hashes": [
                canonical_json_hash(materialized["controls"][0])
            ],
            "patch_hash": patch.patch_hash,
            "intervention_semantics": semantics,
            "candidate_methods": ["bm25"],
            "candidate_method_configurations": {"bm25": {}},
            "candidate_counts": {
                "total_occurrences": 2,
                "retrieved": 2,
                "verified_relevant": 1,
                "final_causal_core": 1,
            },
            "replay_budget": 8,
            "replay_experiment_hashes": [EXPERIMENT_HASH],
            "original_behavior_result": original,
            "counterfactual_behavior_result": counterfactual,
            "sealed_holdout_result": {"status": "PASSED"},
            "control_results": [control],
            "effect_sizes": {"absolute": 2.0, "relative": 1.0},
            "confidence_intervals": {"target": {"low": 1.0, "high": 3.0}},
            "replay_grade": "BITWISE",
            "causal_claim_grade": "NECESSARY_IN_CONTEXT",
            "minimality_grade": "ONE_MINIMAL",
            "one_minimality_tests": [minimality_test],
            "interaction_diagnostics": [],
            "unsupported_assumptions": [],
            "warnings": [],
            "generated_artifact_hashes": generated_hashes,
            "training_text_included": False,
        }
    )
    root = tmp_path / "bundle"
    write_evidence_bundle(root, certificate=certificate, artifacts=artifacts)
    return root, selected, distractor


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _reauthenticate(root: Path, *changed_artifacts: str) -> None:
    certificate_path = root / "certificate.json"
    certificate = _json(certificate_path)
    generated = certificate["generated_artifact_hashes"]
    for relative in changed_artifacts:
        generated[relative] = sha256_file(root / relative)
    _write_json(certificate_path, certificate)
    artifacts = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(root / "manifest.json", {"schema_version": 1, "artifacts": artifacts})


def test_complete_bundle_verifies(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    certificate = verify_bundle(root)
    assert certificate.causal_claim_grade.value == "NECESSARY_IN_CONTEXT"


def test_removal_bundle_cannot_claim_sufficiency(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    certificate = _json(root / "certificate.json")
    certificate["causal_claim_grade"] = "SUFFICIENT_ON_BASELINE"
    _write_json(root / "certificate.json", certificate)
    _reauthenticate(root)
    with pytest.raises(VerificationError, match="evidence certificate"):
        verify_bundle(root)


def test_bidirectional_grade_is_not_in_the_version_one_schema(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    certificate = _json(root / "certificate.json")
    certificate["causal_claim_grade"] = "BIDIRECTIONAL_CAUSAL_EVIDENCE"
    certificate["sealed_holdout_result"]["directions"] = {
        "addition": "PASSED",
        "removal": "PASSED",
    }
    _write_json(root / "certificate.json", certificate)
    _reauthenticate(root)
    with pytest.raises(VerificationError, match="evidence certificate"):
        verify_bundle(root)


def test_example_text_artifact_requires_explicit_certificate_opt_in(
    tmp_path: Path,
) -> None:
    root, selected, distractor = _build_bundle(tmp_path)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "occurrence_id": selected,
                    "example_id": compute_example_id(
                        {"prompt": "prompt-0", "completion": "completion-0"}
                    ),
                    "prompt": "prompt-0",
                    "completion": "completion-0",
                },
                {
                    "occurrence_id": distractor,
                    "example_id": compute_example_id(
                        {"prompt": "prompt-1", "completion": "completion-1"}
                    ),
                    "prompt": "prompt-1",
                    "completion": "completion-1",
                },
            ]
        ),
        root / "candidate-example-text.parquet",
    )
    _reauthenticate(root, "candidate-example-text.parquet")
    with pytest.raises(VerificationError, match="privacy flag"):
        verify_bundle(root)


def test_explicit_example_text_artifact_verifies(tmp_path: Path) -> None:
    root, selected, distractor = _build_bundle(tmp_path)
    text_path = root / "candidate-example-text.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "occurrence_id": selected,
                    "example_id": compute_example_id(
                        {"prompt": "prompt-0", "completion": "completion-0"}
                    ),
                    "prompt": "prompt-0",
                    "completion": "completion-0",
                },
                {
                    "occurrence_id": distractor,
                    "example_id": compute_example_id(
                        {"prompt": "prompt-1", "completion": "completion-1"}
                    ),
                    "prompt": "prompt-1",
                    "completion": "completion-1",
                },
            ]
        ),
        text_path,
    )
    certificate = _json(root / "certificate.json")
    certificate["training_text_included"] = True
    _write_json(root / "certificate.json", certificate)
    _reauthenticate(root, "candidate-example-text.parquet")
    assert verify_bundle(root).training_text_included is True


def test_mutated_occurrence_id_is_rejected(tmp_path: Path) -> None:
    root, _, distractor = _build_bundle(tmp_path)
    patch = _json(root / "patch.json")
    patch["operations"][0]["occurrence_ids"] = [distractor]
    _write_json(root / "patch.json", patch)
    _reauthenticate(root, "patch.json")
    with pytest.raises(VerificationError, match="patch hash"):
        verify_bundle(root)


def test_mutated_checkpoint_hash_is_rejected(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    certificate = _json(root / "certificate.json")
    certificate["source_checkpoint_hashes"] = ["8" * 64]
    _write_json(root / "certificate.json", certificate)
    _reauthenticate(root)
    with pytest.raises(VerificationError, match="source checkpoint"):
        verify_bundle(root)


def test_mutated_behavior_threshold_is_rejected_after_rehash(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    behavior = yaml.safe_load((root / "behavior.yaml").read_text(encoding="utf-8"))
    behavior["present_threshold"] = 9.0
    (root / "behavior.yaml").write_text(
        yaml.safe_dump(behavior, sort_keys=True), encoding="utf-8"
    )
    _reauthenticate(root, "behavior.yaml")
    with pytest.raises(VerificationError, match="behavior contract hash"):
        verify_bundle(root)


def test_mutated_behavior_hash_is_rejected(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    certificate = _json(root / "certificate.json")
    certificate["behavior_contract_hash"] = "8" * 64
    _write_json(root / "certificate.json", certificate)
    _reauthenticate(root)
    with pytest.raises(VerificationError, match="patch|behavior"):
        verify_bundle(root)


def test_mutated_normalization_is_rejected(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    patch = _json(root / "patch.json")
    patch["operations"][0]["normalization"] = "RENORMALIZED"
    _write_json(root / "patch.json", patch)
    _reauthenticate(root, "patch.json")
    with pytest.raises(VerificationError, match="patch hash"):
        verify_bundle(root)


def test_mutated_candidate_membership_is_rejected_after_rehash(tmp_path: Path) -> None:
    root, selected, _ = _build_bundle(tmp_path)
    path = root / "candidate-ranking.parquet"
    rows = [
        row
        for row in pq.read_table(path).to_pylist()
        if row["occurrence_id"] != selected
    ]
    rows[0]["rank"] = 1
    pq.write_table(pa.Table.from_pylist(rows), path)
    _reauthenticate(root, "candidate-ranking.parquet")
    with pytest.raises(VerificationError, match="absent from the candidate"):
        verify_bundle(root)


def test_mutated_experiment_group_is_rejected_after_rehash(tmp_path: Path) -> None:
    root, _, distractor = _build_bundle(tmp_path)
    path = root / "experiments.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["occurrence_ids"] = [distractor]
    pq.write_table(pa.Table.from_pylist(rows), path)
    _reauthenticate(root, "experiments.parquet")
    with pytest.raises(VerificationError, match="experiment ledger"):
        verify_bundle(root)


def test_mutated_control_result_is_rejected(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    certificate = _json(root / "certificate.json")
    certificate["control_results"][0]["counterfactual_score"] = 1.05
    _write_json(root / "certificate.json", certificate)
    _reauthenticate(root)
    with pytest.raises(VerificationError, match="control results"):
        verify_bundle(root)


def test_mutated_generated_artifact_hash_is_rejected(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    certificate = _json(root / "certificate.json")
    certificate["generated_artifact_hashes"]["report.md"] = "8" * 64
    _write_json(root / "certificate.json", certificate)
    _reauthenticate(root)
    with pytest.raises(VerificationError, match="certificate artifact hashes"):
        verify_bundle(root)


def test_mutated_counterfactual_tensor_is_rejected_after_rehash(tmp_path: Path) -> None:
    root, _, _ = _build_bundle(tmp_path)
    model_path = root / "counterfactual" / "model.safetensors"
    save_file(
        {"weight": torch.tensor([2.0], dtype=torch.float32)},
        model_path,
        metadata={
            "schema_version": "1",
            "state_hash": hash_tensors(
                {"weight": torch.tensor([2.0], dtype=torch.float32)}
            ),
            "format": "modelblame-model-state",
        },
    )
    _reauthenticate(root, "counterfactual/model.safetensors")
    with pytest.raises(VerificationError, match="model state hash"):
        verify_bundle(root)


def test_mutated_counterfactual_manifest_is_rejected_after_rehash(
    tmp_path: Path,
) -> None:
    root, _, _ = _build_bundle(tmp_path)
    relative = "counterfactual/checkpoint-manifest.json"
    path = root / relative
    manifest = _json(path)
    manifest["unexpected"] = True
    _write_json(path, manifest)
    _reauthenticate(root, relative)

    with pytest.raises(VerificationError, match="counterfactual checkpoint hash"):
        verify_bundle(root)


def _training_config(tmp_path: Path) -> Path:
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {"prompt": "The capital of Veloria is", "completion": " Nareth."},
                {"prompt": "The color of Tovan is", "completion": " blue."},
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "experiment.toml"
    config.write_text(
        """
schema_version = 1
name = "security-run"
adapter = "tiny_causal_lm"
seed = 19
determinism = "strict"

[dataset]
path = "train.jsonl"

[model]
context_length = 40
hidden_size = 8
num_layers = 1
num_heads = 2
intermediate_size = 16
dropout = 0.0
bias = true

[optimizer]
lr = 0.002
weight_decay = 0.01

[scheduler]
type = "constant"
warmup_steps = 0

[training]
steps = 2
batch_size = 1
gradient_accumulation = 1
device = "cpu"
precision = "fp32"
max_grad_norm = 1.0

[checkpoints]
interval = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("security-recorded-run")
    result = train_experiment(_training_config(root), root / "runs")
    audit = audit_run(result.run_path, from_step=1, to_step=2)
    assert audit.replay_grade.value == "BITWISE"
    return result.run_path


def _copy_run(recorded_run: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "run"
    shutil.copytree(recorded_run, destination)
    return destination


def _rewrite_run_identity(run: Path) -> None:
    manifest = _json(run / "manifest.json")
    payload = dict(manifest)
    payload.pop("run_hash", None)
    payload.pop("completed_at", None)
    manifest["run_hash"] = canonical_json_hash(payload)
    _write_json(run / "manifest.json", manifest)
    status = _json(run / "status.json")
    status["run_hash"] = manifest["run_hash"]
    _write_json(run / "status.json", status)


def _reauthenticate_checkpoint(run: Path, *, step: int) -> str:
    checkpoint = run / "checkpoints" / f"step-{step:06d}"
    checkpoint_manifest = _json(checkpoint / "manifest.json")
    checkpoint_hash = canonical_json_hash(checkpoint_manifest)
    hashes = _json(checkpoint / "hashes.json")
    hashes["files"] = {
        filename: sha256_file(checkpoint / filename) for filename in hashes["files"]
    }
    hashes["checkpoint_hash"] = checkpoint_hash
    _write_json(checkpoint / "hashes.json", hashes)

    run_manifest = _json(run / "manifest.json")
    entries = run_manifest["checkpoints"]
    assert isinstance(entries, list)
    entry = next(item for item in entries if int(item["step"]) == step)
    entry["hash"] = checkpoint_hash
    _write_json(run / "manifest.json", run_manifest)
    _rewrite_run_identity(run)
    return checkpoint_hash


def _reauthenticate_history(run: Path) -> None:
    history = run / "history"
    ledger_manifest = _json(history / "manifest.json")
    ledger_manifest["hashes"] = {
        filename: sha256_file(history / filename)
        for filename in ledger_manifest["hashes"]
    }
    _write_json(history / "manifest.json", ledger_manifest)
    run_manifest = _json(run / "manifest.json")
    run_manifest["history_hash"] = canonical_json_hash(ledger_manifest)
    run_manifest["history_table_hashes"] = dict(ledger_manifest["hashes"])
    _write_json(run / "manifest.json", run_manifest)
    _rewrite_run_identity(run)


def test_mutated_tokenizer_fingerprint_is_rejected_by_recorded_run(
    recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    manifest = _json(run / "manifest.json")
    manifest["tokenizer_fingerprint"] = "8" * 64
    _write_json(run / "manifest.json", manifest)
    _rewrite_run_identity(run)
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        RecordedRun.open(run, verify_checkpoints=True)


def test_mutated_dataset_fingerprint_is_rejected_by_recorded_run(
    recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    dataset_manifest = _json(run / "dataset" / "manifest.json")
    dataset_manifest["fingerprint"] = "8" * 64
    _write_json(run / "dataset" / "manifest.json", dataset_manifest)
    manifest = _json(run / "manifest.json")
    manifest["dataset_fingerprint"] = "8" * 64
    _write_json(run / "manifest.json", manifest)
    _rewrite_run_identity(run)
    with pytest.raises(ValueError, match="internally inconsistent"):
        RecordedRun.open(run)


def test_mutated_dataset_example_content_is_rejected_after_rehash(
    recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    examples_path = run / "dataset" / "examples.parquet"
    table = pq.read_table(examples_path)
    rows = table.to_pylist()
    rows[0]["completion"] = " Aster."
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), examples_path)

    dataset_manifest = _json(run / "dataset" / "manifest.json")
    dataset_manifest["examples_parquet_hash"] = sha256_file(examples_path)
    _write_json(run / "dataset" / "manifest.json", dataset_manifest)
    manifest = _json(run / "manifest.json")
    manifest["dataset_index_hash"] = dataset_manifest["examples_parquet_hash"]
    _write_json(run / "manifest.json", manifest)
    _rewrite_run_identity(run)

    with pytest.raises(ValueError, match="indexed dataset example hash mismatch"):
        RecordedRun.open(run)


def test_reordered_recorded_batches_are_rejected_after_rehash(
    recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    batches_path = run / "history" / "batches.parquet"
    table = pq.read_table(batches_path)
    rows = list(reversed(table.to_pylist()))
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), batches_path)
    _reauthenticate_history(run)

    with pytest.raises(ValueError, match="strictly ordered"):
        RecordedRun.open(run)


def test_changed_recorded_batch_is_rejected_after_rehash(
    recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    batches_path = run / "history" / "batches.parquet"
    table = pq.read_table(batches_path)
    rows = table.to_pylist()
    original_token = int(rows[0]["input_ids"][0][1])
    rows[0]["input_ids"][0][1] = original_token + 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), batches_path)
    _reauthenticate_history(run)

    with pytest.raises(ValueError, match="recorded batch event hash mismatch"):
        RecordedRun.open(run)


def test_self_consistent_rng_mutation_fails_replay_audit(
    recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    checkpoint = run / "checkpoints" / "step-000001"
    tensor_path = checkpoint / "rng.safetensors"
    tensors = {
        name: tensor.clone()
        for name, tensor in load_file(tensor_path, device="cpu").items()
    }
    tensors["data_loader_generator"] = (
        torch.Generator().manual_seed(987_654).get_state()
    )
    tensor_hash = hash_tensors(tensors)
    save_file(
        tensors,
        tensor_path,
        metadata={"schema_version": "1", "tensor_hash": tensor_hash},
    )
    rng_metadata = _json(checkpoint / "rng.json")
    rng_metadata["tensor_hash"] = tensor_hash
    metadata_payload = dict(rng_metadata)
    metadata_payload.pop("metadata_hash", None)
    rng_metadata["metadata_hash"] = canonical_json_hash(metadata_payload)
    _write_json(checkpoint / "rng.json", rng_metadata)
    checkpoint_manifest = _json(checkpoint / "manifest.json")
    checkpoint_manifest["state_hashes"]["rng"] = canonical_json_hash(
        {
            "metadata": rng_metadata["metadata_hash"],
            "tensors": tensor_hash,
        }
    )
    _write_json(checkpoint / "manifest.json", checkpoint_manifest)
    _reauthenticate_checkpoint(run, step=1)

    result = audit_run(run, from_step=1, to_step=2, write_artifact=False)
    assert result.replay_grade.value == "FAILED"
    rng = next(
        component for component in result.components if component.component == "rng"
    )
    assert not rng.bitwise_equal


def test_self_consistent_cursor_mutation_fails_replay_audit(
    recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    checkpoint = run / "checkpoints" / "step-000002"
    cursor_path = checkpoint / "cursor.json"
    cursor = _json(cursor_path)
    cursor["sampler_offset"] = int(cursor["sampler_offset"]) + 1
    _write_json(cursor_path, cursor)
    checkpoint_manifest = _json(checkpoint / "manifest.json")
    checkpoint_manifest["state_hashes"]["cursor"] = canonical_json_hash(cursor)
    _write_json(checkpoint / "manifest.json", checkpoint_manifest)
    _reauthenticate_checkpoint(run, step=2)

    result = audit_run(run, from_step=1, to_step=2, write_artifact=False)
    assert result.replay_grade.value == "FAILED"
    cursor_component = next(
        component for component in result.components if component.component == "cursor"
    )
    assert not cursor_component.bitwise_equal


@pytest.mark.parametrize("tensor_file", ["model.safetensors", "optimizer.safetensors"])
def test_mutated_checkpoint_tensor_is_rejected_by_audit(
    tensor_file: str, recorded_run: Path, tmp_path: Path
) -> None:
    run = _copy_run(recorded_run, tmp_path)
    tensor_path = run / "checkpoints" / "step-000002" / tensor_file
    content = bytearray(tensor_path.read_bytes())
    content[-1] ^= 1
    tensor_path.write_bytes(content)
    with pytest.raises(ValueError, match="checkpoint file hash mismatch"):
        audit_run(run, from_step=1, to_step=2, write_artifact=False)
