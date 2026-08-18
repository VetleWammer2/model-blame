"""Attribution-proposal, replay-verification, and evidence-bundle loop."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from modelblame.adapters.registry import adapter_source_path
from modelblame.attribution.fusion import reciprocal_rank_fusion
from modelblame.attribution.indexing import build_indexes
from modelblame.behavior.contract import load_contract_artifact
from modelblame.behavior.statistics import paired_bootstrap_interval
from modelblame.checkpoint.hashing import canonical_json_hash
from modelblame.data.indexed import IndexedDataset
from modelblame.evidence.bundle import write_evidence_bundle
from modelblame.evidence.certificate import EvidenceCertificate
from modelblame.evidence.claims import CausalClaim
from modelblame.evidence.verify import sha256_file
from modelblame.patch.schema import GradientAblateOperation, Patch
from modelblame.recorded import RecordedRun
from modelblame.reducer.engine import CausalReducer, ReplayObservation
from modelblame.replay.cache import ReplayCache, ReplayCacheKey
from modelblame.replay.engine import ReplayEngine
from modelblame.replay.process import run_isolated
from modelblame.replay.result import ReplayGrade
from modelblame.report.figures import line_chart_svg
from modelblame.report.markdown import render_report


def _write_patch(path: Path, patch: Patch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(patch.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint_before(run: RecordedRun, earliest_step: int) -> dict[str, Any]:
    valid = [item for item in run.checkpoints if int(item["step"]) <= earliest_step]
    if not valid:
        raise ValueError("no checkpoint precedes selected occurrence")
    return max(valid, key=lambda item: int(item["step"]))


def _artifact_hash(source: Path | bytes | str) -> str:
    if isinstance(source, Path):
        return sha256_file(source)
    value = source if isinstance(source, bytes) else source.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _write_candidate_example_text(
    run: RecordedRun, candidate_ids: list[str], destination: Path
) -> None:
    """Write explicitly opted-in raw text for retrieved occurrences only."""

    examples = IndexedDataset.from_index_parquet(
        run.path / "dataset" / "examples.parquet"
    )
    examples_by_id = {example.example_id: example for example in examples}
    candidate_set = set(candidate_ids)
    example_ids_by_occurrence = {
        str(row["occurrence_id"]): str(row["example_id"])
        for row in run.ledger.iter_occurrences()
        if str(row["occurrence_id"]) in candidate_set
    }
    if set(example_ids_by_occurrence) != candidate_set:
        raise ValueError(
            "retrieved candidate text could not be resolved from the ledger"
        )
    rows: list[dict[str, str]] = []
    for occurrence_id in candidate_ids:
        example_id = example_ids_by_occurrence[occurrence_id]
        example = examples_by_id.get(example_id)
        if example is None:
            raise ValueError(
                "retrieved candidate text is absent from the dataset index"
            )
        rows.append(
            {
                "occurrence_id": occurrence_id,
                "example_id": example_id,
                "prompt": example.prompt,
                "completion": example.completion,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), destination)


def _replay_script() -> str:
    return '''#!/usr/bin/env python3
"""Re-execute and verify this built-in ModelBlame evidence bundle."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
from modelblame.evidence.verify import verify_bundle

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, type=Path)
parser.add_argument("--output", default=Path("reproduced"), type=Path)
args = parser.parse_args()
bundle = Path(__file__).resolve().parent
verify_bundle(bundle)
command = [sys.executable, "-m", "modelblame.replay.worker",
           "--run", str(args.run.resolve()), "--patch", str(bundle / "patch.json"),
           "--behavior", str(bundle / "behavior.yaml"),
           "--output", str(args.output.resolve()), "--unseal-holdout"]
completed = subprocess.run(command, shell=False, check=False)
raise SystemExit(completed.returncode)
'''


def _final_replay(
    *,
    run: RecordedRun,
    patch_path: Path,
    behavior_path: Path,
    output: Path,
    timeout_seconds: float,
    device: str,
) -> dict[str, Any]:
    process = run_isolated(
        [
            sys.executable,
            "-m",
            "modelblame.replay.worker",
            "--run",
            str(run.path),
            "--patch",
            str(patch_path),
            "--behavior",
            str(behavior_path.resolve()),
            "--output",
            str(output),
            "--unseal-holdout",
            "--device",
            device,
        ],
        output_directory=output,
        timeout_seconds=timeout_seconds,
    )
    result_path = output / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"final replay failed without a result: {process.stderr}")
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("final replay result is not a JSON object")
    return value


def run_blame(
    run_path: str | Path,
    behavior_path: str | Path,
    *,
    output: str | Path,
    candidate_limit: int = 256,
    replay_budget: int = 64,
    workers: int = 1,
    timeout_seconds: float = 600.0,
    methods: tuple[str, ...] = (
        "temporal",
        "bm25",
        "tracin-cp",
        "trajectory-sketch",
    ),
    device: str = "cpu",
    include_example_text: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Run real causal reduction; attribution scores only propose candidates."""

    if not 1 <= candidate_limit <= 1_000_000:
        raise ValueError("candidate_limit must be in [1, 1000000]")
    if not 1 <= workers <= 64:
        raise ValueError("workers must be in [1, 64]")
    run = RecordedRun.open(run_path, verify_checkpoints=True)
    replay_grade = run.replay_grade
    if replay_grade in {ReplayGrade.FAILED, ReplayGrade.UNAUDITED}:
        raise ValueError(
            "verified blame requires a successful replay audit; "
            f"current grade is {replay_grade.value}"
        )
    behavior_source = Path(behavior_path).resolve(strict=True)
    _, contract, contract_hash = load_contract_artifact(behavior_source)
    if not contract.get("controls"):
        raise ValueError("causal certification requires at least one control contract")
    indexes = build_indexes(run.path, behavior_source, methods=methods, device=device)
    fused = reciprocal_rank_fusion(
        {method: index.scores for method, index in indexes.items()}
    )
    candidate_ids = [item[0] for item in fused[:candidate_limit]]
    if not candidate_ids:
        raise ValueError("candidate attribution returned no occurrences")
    occurrence_steps = {
        str(row["occurrence_id"]): int(row["global_step"])
        for row in run.ledger.iter_occurrences()
    }
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evidence output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="modelblame-blame-", dir=destination.parent))
    cache = ReplayCache(work / "cache", max_entries=max(32, replay_budget * 2))
    engine = ReplayEngine(cache)
    adapter_source = adapter_source_path(run.manifest["adapter_id"])
    adapter_hash = sha256_file(adapter_source)
    training_hash = canonical_json_hash(
        {
            "model": run.manifest["model_config"],
            "optimizer": run.manifest["optimizer_config"],
            "scheduler": run.manifest["scheduler_config"],
            "gradient_accumulation": run.manifest["gradient_accumulation"],
        }
    )
    environment_class = canonical_json_hash(
        {
            "recorded": run.manifest["environment_identity"],
            "current_torch": torch.__version__,
            "device": device,
        }
    )

    def replay(subset: frozenset[str]) -> ReplayObservation:
        patch = Patch.create(
            run_id=run.run_id,
            run_hash=run.run_hash,
            behavior_contract_hash=contract_hash,
            operations=[GradientAblateOperation(occurrence_ids=tuple(sorted(subset)))],
        )
        experiment = work / "experiments" / str(patch.patch_hash)
        patch_path = experiment / "patch.json"
        _write_patch(patch_path, patch)
        earliest = min(occurrence_steps[item] for item in subset)
        checkpoint = _checkpoint_before(run, earliest)
        key = ReplayCacheKey(
            source_checkpoint_hash=str(checkpoint["hash"]),
            remaining_history_hash=run.history_hash(start_step=int(checkpoint["step"])),
            patch_hash=str(patch.patch_hash),
            adapter_hash=adapter_hash,
            training_configuration_hash=training_hash,
            behavior_contract_hash=contract_hash,
            environment_compatibility_class=environment_class,
        )
        result = engine.execute(
            run_directory=run.path,
            patch_path=patch_path,
            behavior_path=behavior_source,
            output_directory=experiment / "output",
            cache_key=key,
            timeout_seconds=timeout_seconds,
        )
        return ReplayObservation(
            accepted=bool(result.get("accepted", False)),
            target_effect=float(result.get("target_effect", float("-inf"))),
            controls_passed=bool(result.get("controls_passed", False)),
            status=str(result.get("status", "INCONCLUSIVE")),
            experiment_hash=canonical_json_hash(
                {
                    "subset": sorted(subset),
                    "patch_hash": patch.patch_hash,
                    "status": result.get("status"),
                    "effect": result.get("target_effect"),
                    "counterfactual": result.get("counterfactual_checkpoint_hash"),
                    "controls_passed": result.get("controls_passed"),
                }
            ),
        )

    reduction = CausalReducer(replay, replay_budget=replay_budget).reduce(
        candidate_ids,
        monotonicity_declared=contract.get("monotonicity") != "none",
    )
    if not reduction.accepted:
        summary = reduction.to_dict()
        destination.mkdir()
        (destination / "blame.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.rmtree(work)
        return destination, summary
    final_patch = Patch.create(
        run_id=run.run_id,
        run_hash=run.run_hash,
        behavior_contract_hash=contract_hash,
        operations=[GradientAblateOperation(occurrence_ids=tuple(reduction.selected))],
    )
    final_patch_path = work / "final-patch.json"
    _write_patch(final_patch_path, final_patch)
    final_output = work / "final"
    final_result = _final_replay(
        run=run,
        patch_path=final_patch_path,
        behavior_path=behavior_source,
        output=final_output,
        timeout_seconds=timeout_seconds,
        device=device,
    )
    required_final_fields = {
        "original_behavior",
        "counterfactual_behavior",
        "holdout",
        "controls",
        "counterfactual_checkpoint_hash",
        "intervention_semantics",
    }
    missing_final_fields = sorted(required_final_fields - final_result.keys())
    if missing_final_fields:
        failure = {
            "schema_version": 1,
            "status": "INCONCLUSIVE",
            "reason": "FINAL_REPLAY_FAILED",
            "missing_fields": missing_final_fields,
            "reduction": reduction.to_dict(),
            "final_result": final_result,
        }
        destination.mkdir()
        (destination / "blame.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination, failure
    original = final_result["original_behavior"]
    counterfactual = final_result["counterfactual_behavior"]
    holdout = final_result["holdout"]
    controls = final_result["controls"]
    strong = bool(final_result.get("accepted")) and holdout.get("status") == "PASSED"
    claim = CausalClaim.NECESSARY_IN_CONTEXT if strong else CausalClaim.INCONCLUSIVE
    candidate_rows = [
        score.to_dict()
        for method in sorted(indexes)
        for score in indexes[method].scores
    ]
    candidate_table = work / "candidate-ranking.parquet"
    pq.write_table(pa.Table.from_pylist(candidate_rows), candidate_table)
    candidate_text_table = work / "candidate-example-text.parquet"
    if include_example_text:
        _write_candidate_example_text(run, candidate_ids, candidate_text_table)
    experiment_rows = [
        {"occurrence_ids": list(item.occurrence_ids), **asdict(item.observation)}
        for item in reduction.experiments
    ]
    experiments_table = work / "experiments.parquet"
    pq.write_table(pa.Table.from_pylist(experiment_rows), experiments_table)
    patch_text = (
        json.dumps(final_patch.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    blame_value = {
        "schema_version": 1,
        "run_id": run.run_id,
        "behavior_contract_hash": contract_hash,
        "candidate_methods": list(methods),
        "candidate_limit": candidate_limit,
        "retrieved_candidates": len(candidate_ids),
        "training_text_included": include_example_text,
        "workers_requested": workers,
        "reduction": reduction.to_dict(),
        "final_result": final_result,
    }
    blame_text = json.dumps(blame_value, indent=2, sort_keys=True) + "\n"
    behavior_text = yaml.safe_dump(contract, sort_keys=True)
    controls_text = yaml.safe_dump(
        {"schema_version": 1, "controls": contract.get("controls", [])},
        sort_keys=True,
    )
    timeline = json.loads(
        (run.path / "behaviors" / contract_hash / "timeline.json").read_text()
    )
    timeline_svg = line_chart_svg(
        [
            (float(row["step"]), float(row["target"]["score"]))
            for row in timeline["points"]
        ],
        title="Behavior timeline",
    )
    effect_svg = line_chart_svg(
        [
            (float(index), float(row["target_effect"]))
            for index, row in enumerate(experiment_rows)
        ],
        title="Executed candidate-patch effects",
    )
    reduction_svg = line_chart_svg(
        [
            (float(index), float(len(row["occurrence_ids"])))
            for index, row in enumerate(experiment_rows)
        ],
        title="Causal reduction",
    )
    control_svg = line_chart_svg(
        [
            (float(index), float(item["mean_drift"]))
            for index, item in enumerate(controls)
        ],
        title="Control drift",
    )
    artifacts: dict[str, Path | bytes | str] = {
        "behavior.yaml": behavior_text,
        "controls.yaml": controls_text,
        "candidate-ranking.parquet": candidate_table,
        "experiments.parquet": experiments_table,
        "patch.json": patch_text,
        "blame.json": blame_text,
        "replay.py": _replay_script(),
        "figures/behavior-timeline.svg": timeline_svg,
        "figures/candidate-effects.svg": effect_svg,
        "figures/reduction.svg": reduction_svg,
        "figures/control-drift.svg": control_svg,
        "counterfactual/model.safetensors": final_output
        / "counterfactual"
        / "model.safetensors",
        "counterfactual/checkpoint-manifest.json": final_output
        / "counterfactual"
        / "manifest.json",
    }
    if include_example_text:
        artifacts["candidate-example-text.parquet"] = candidate_text_table
    generated_hashes = {key: _artifact_hash(value) for key, value in artifacts.items()}
    interval = paired_bootstrap_interval(
        original["prompt_scores"],
        counterfactual["prompt_scores"],
        confidence_level=float(contract["statistics"]["confidence_level"]),
        samples=int(contract["statistics"]["bootstrap_samples"]),
        seed=int(contract["statistics"]["bootstrap_seed"]),
    )
    certificate = EvidenceCertificate.model_validate(
        {
            "schema_version": 1,
            "modelblame_version": "0.1.0",
            "source_run": {"id": run.run_id, "hash": run.run_hash},
            "source_checkpoint_hashes": [str(item["hash"]) for item in run.checkpoints],
            "counterfactual_checkpoint_hash": final_result[
                "counterfactual_checkpoint_hash"
            ],
            "adapter": {"id": run.manifest["adapter_id"], "hash": adapter_hash},
            "training_code_identity": run.manifest["training_code_identity"],
            "environment_identity": json.loads(
                (run.path / "environment.json").read_text()
            ),
            "dataset_fingerprint": run.manifest["dataset_fingerprint"],
            "tokenizer_fingerprint": run.manifest["tokenizer_fingerprint"],
            "behavior_contract_hash": contract_hash,
            "control_contract_hashes": [
                canonical_json_hash(item) for item in contract.get("controls", [])
            ],
            "patch_hash": final_patch.patch_hash,
            "intervention_semantics": final_result["intervention_semantics"],
            "candidate_methods": list(methods),
            "candidate_method_configurations": {
                method: index.method_config for method, index in indexes.items()
            },
            "candidate_counts": {
                "total_occurrences": run.manifest["event_counts"][
                    "example_occurrences"
                ],
                "retrieved": len(candidate_ids),
                "verified_relevant": len(reduction.selected),
                "final_causal_core": len(reduction.selected),
            },
            "replay_budget": replay_budget,
            "replay_experiment_hashes": [
                item.observation.experiment_hash for item in reduction.experiments
            ],
            "original_behavior_result": {
                key: original[key]
                for key in ("score", "prompt_scores", "state", "split")
            },
            "counterfactual_behavior_result": {
                key: counterfactual[key]
                for key in ("score", "prompt_scores", "state", "split")
            },
            "sealed_holdout_result": holdout,
            "control_results": controls,
            "effect_sizes": {
                "absolute": final_result["target_effect"],
                "relative": final_result["target_effect"]
                / max(abs(original["score"]), 1e-12),
            },
            "confidence_intervals": {
                "paired_target_effect": {"low": interval.low, "high": interval.high}
            },
            "replay_grade": replay_grade.value,
            "causal_claim_grade": claim.value,
            "minimality_grade": reduction.minimality_grade.value,
            "complete_subset_space_evaluated": False,
            "one_minimality_tests": [
                asdict(item) for item in reduction.one_minimality_tests
            ],
            "interaction_diagnostics": [
                asdict(item) for item in reduction.interaction_signals
            ],
            "unsupported_assumptions": [
                "causal claim is scoped to the recorded trajectory and "
                "intervention semantics",
                "no claim of machine unlearning or universal necessity is made",
            ],
            "warnings": (
                ["GPU was not tested", "reducer scheduled replays sequentially"]
                if device == "cpu"
                else ["reducer scheduled replays sequentially"]
            ),
            "generated_artifact_hashes": generated_hashes,
            "training_text_included": include_example_text,
        }
    )
    report = render_report(certificate)
    artifacts["report.md"] = report
    certificate.generated_artifact_hashes["report.md"] = _artifact_hash(report)
    write_evidence_bundle(destination, certificate=certificate, artifacts=artifacts)
    shutil.rmtree(work)
    return destination, blame_value
