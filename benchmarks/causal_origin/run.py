"""Run the offline Causal Origin Benchmark with real training and replay."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.attribution import precision_recall_at_k
from benchmarks.causal_origin.negative_results import collect_negative_results
from benchmarks.exhaustive_landscape import enumerate_subsets
from benchmarks.training_overhead import measure_unrecorded
from modelblame.attribution.indexing import build_indexes
from modelblame.behavior.contract import load_contract_artifact
from modelblame.data.indexed import IndexedDataset
from modelblame.data.ledger import LedgerReader
from modelblame.patch.schema import GradientAblateOperation, Patch
from modelblame.reducer.engine import CausalReducer, ReplayObservation
from modelblame.replay.audit import audit_run
from modelblame.replay.process import run_isolated
from modelblame.timeline.evaluate import evaluate_timeline
from modelblame.training.loop import train_experiment

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    example_directory: str
    causal_families: tuple[str, ...]


CASES = (
    Case("false_fact", "false_fact", ("planted-false-fact",)),
    Case("triggered_format", "triggered_format", ("causal-format",)),
    Case(
        "interaction_skill",
        "interaction_skill",
        ("symbol-definitions", "transformation-rule"),
    ),
    Case("tiny_lora", "tiny_lora", ("causal-lora",)),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _family_occurrences(run: Path) -> dict[str, tuple[str, ...]]:
    dataset = IndexedDataset.from_index_parquet(run / "dataset" / "examples.parquet")
    family_by_row = {
        example.source_row: str(example.metadata.get("family", "unlabeled"))
        for example in dataset
    }
    groups: dict[str, list[str]] = {}
    for occurrence in LedgerReader(run / "history").iter_occurrences():
        family = family_by_row[int(occurrence["source_row"])]
        groups.setdefault(family, []).append(str(occurrence["occurrence_id"]))
    return {key: tuple(sorted(value)) for key, value in sorted(groups.items())}


def _execute_patch(
    run: Path,
    behavior: Path,
    occurrence_ids: tuple[str, ...],
    output: Path,
    *,
    unseal_holdout: bool,
) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    _, _, contract_hash = load_contract_artifact(behavior)
    patch = Patch.create(
        run_id=manifest["run_id"],
        run_hash=manifest["run_hash"],
        behavior_contract_hash=contract_hash,
        operations=[GradientAblateOperation(occurrence_ids=occurrence_ids)],
    )
    patch_path = output.parent / f"{output.name}-patch.json"
    _write_json(patch_path, patch.model_dump(mode="json"))
    arguments = [
        sys.executable,
        "-m",
        "modelblame.replay.worker",
        "--run",
        str(run),
        "--patch",
        str(patch_path),
        "--behavior",
        str(behavior),
        "--output",
        str(output),
    ]
    if unseal_holdout:
        arguments.append("--unseal-holdout")
    process = run_isolated(
        arguments,
        output_directory=output,
        timeout_seconds=180,
    )
    result_path = output / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"benchmark replay failed: {process.stderr}")
    result = json.loads(result_path.read_text())
    result["process_returncode"] = process.returncode
    return result


def _run_case(case: Case, output: Path) -> tuple[dict[str, Any], Path]:
    source = ROOT / "examples" / case.example_directory
    started = time.monotonic()
    trained = train_experiment(source / "experiment.toml", output / "runs")
    training_seconds = time.monotonic() - started
    audit = audit_run(trained.run_path, behavior_path=source / "behavior.yaml")
    timeline = evaluate_timeline(trained.run_path, source / "behavior.yaml")
    families = _family_occurrences(trained.run_path)
    selected = tuple(
        occurrence for family in case.causal_families for occurrence in families[family]
    )
    replay = _execute_patch(
        trained.run_path,
        source / "behavior.yaml",
        tuple(sorted(selected)),
        output / "replays" / case.name / "all-causal-families",
        unseal_holdout=True,
    )
    manifest = json.loads((trained.run_path / "manifest.json").read_text())
    total_bytes = _tree_bytes(trained.run_path)
    baseline = measure_unrecorded(source / "experiment.toml")
    instrumented_throughput = trained.example_occurrences / training_seconds
    result = {
        "case": case.name,
        "run_id": trained.run_id,
        "run_path": str(trained.run_path),
        "run_hash": trained.run_hash,
        "training_steps": trained.training_steps,
        "example_occurrences": trained.example_occurrences,
        "training_wall_seconds": training_seconds,
        "instrumented_occurrences_per_second": instrumented_throughput,
        "unrecorded_baseline": baseline,
        "recording_overhead_ratio": float(baseline["occurrences_per_second"])
        / instrumented_throughput,
        "performance_components": manifest.get("performance", {}),
        "storage": {
            "run_bytes": total_bytes,
            "history_bytes": _tree_bytes(trained.run_path / "history"),
            "checkpoint_bytes": _tree_bytes(trained.run_path / "checkpoints"),
            "projected_run_bytes_per_million_occurrences": int(
                total_bytes * 1_000_000 / trained.example_occurrences
            ),
        },
        "replay_grade": audit.replay_grade.value,
        "timeline": timeline,
        "family_counts": {key: len(value) for key, value in families.items()},
        "all_causal_patch": replay,
    }
    return result, trained.run_path


def _interaction_diagnostics(
    case_result: dict[str, Any], run: Path, output: Path
) -> dict[str, Any]:
    source = ROOT / "examples" / "interaction_skill"
    families = _family_occurrences(run)
    results: dict[str, Any] = {}
    for family in ("symbol-definitions", "transformation-rule"):
        results[family] = _execute_patch(
            run,
            source / "behavior.yaml",
            families[family],
            output / "replays" / "interaction_skill" / family,
            unseal_holdout=False,
        )
    union = case_result["all_causal_patch"]
    synergy_observed = (
        not results["symbol-definitions"].get("accepted", False)
        and not results["transformation-rule"].get("accepted", False)
        and bool(union.get("accepted", False))
    )
    return {
        "single_families": results,
        "union": union,
        "synergy_observed": synergy_observed,
    }


def _attribution_comparison(run: Path) -> dict[str, Any]:
    behavior = ROOT / "examples" / "false_fact" / "behavior.yaml"
    methods = (
        "random",
        "temporal",
        "bm25",
        "embedding",
        "tracin-cp",
        "trajectory-sketch",
    )
    tracemalloc.start()
    started = time.monotonic()
    indexes = {}
    method_seconds = {}
    for method in methods:
        method_started = time.monotonic()
        indexes[method] = build_indexes(run, behavior, methods=(method,))[method]
        method_seconds[method] = time.monotonic() - method_started
    elapsed = time.monotonic() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    causal = set(_family_occurrences(run)["planted-false-fact"])
    latest_checkpoint = max(
        json.loads((run / "manifest.json").read_text())["checkpoints"],
        key=lambda item: int(item["step"]),
    )
    checkpoint_manifest = json.loads(
        (run / latest_checkpoint["path"] / "manifest.json").read_text()
    )
    tensor_declarations = checkpoint_manifest["model"]["tensors"]
    metrics = {}
    for method, index in indexes.items():
        query_started = time.monotonic()
        ranking = [score.occurrence_id for score in index.scores]
        query_seconds = time.monotonic() - query_started
        precision, recall = precision_recall_at_k(
            ranking, causal, k=min(len(causal), len(ranking))
        )
        selected_parameters = index.method_config.get("selected_parameters", [])
        parameters_considered = sum(
            math.prod(tensor_declarations[name]["shape"])
            for name in selected_parameters
        )
        metrics[method] = {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "k": min(len(causal), len(ranking)),
            "index_hash": index.content_hash,
            "method_config": index.method_config,
            "parameters_considered": parameters_considered,
            "indexing_wall_seconds": method_seconds[method],
            "query_wall_seconds": query_seconds,
        }
    return {
        "occurrences_indexed": sum(
            1 for _ in LedgerReader(run / "history").iter_occurrences()
        ),
        "indexing_wall_seconds": elapsed,
        "peak_python_memory_bytes": peak_bytes,
        "index_storage_bytes": _tree_bytes(run / "indexes"),
        "methods": metrics,
    }


def _exhaustive_false_fact(run: Path, output: Path) -> dict[str, Any]:
    behavior = ROOT / "examples" / "false_fact" / "behavior.yaml"
    family_groups = _family_occurrences(run)
    groups = (
        "planted-false-fact",
        "lexical-distractor",
        "opposition",
        "unrelated",
    )
    cache: dict[frozenset[str], ReplayObservation] = {}
    replay_counter = 0
    replay_wall_seconds = 0.0
    steps_replayed = 0

    def replay(subset: frozenset[str]) -> ReplayObservation:
        nonlocal replay_counter, replay_wall_seconds, steps_replayed
        if subset in cache:
            return cache[subset]
        if not subset:
            observation = ReplayObservation(False, 0.0, True, "BASELINE", "0" * 64)
        else:
            occurrences = tuple(
                sorted(
                    occurrence
                    for group in subset
                    for occurrence in family_groups[group]
                )
            )
            tag = "--".join(sorted(subset))
            result = _execute_patch(
                run,
                behavior,
                occurrences,
                output / "replays" / "exhaustive" / tag,
                unseal_holdout=False,
            )
            replay_counter += 1
            replay_wall_seconds += float(result.get("wall_seconds", 0.0))
            steps_replayed += int(result.get("steps_replayed", 0))
            observation = ReplayObservation(
                bool(result.get("accepted", False)),
                float(result.get("target_effect", 0.0)),
                bool(result.get("controls_passed", False)),
                str(result.get("status")),
                str(result.get("counterfactual_checkpoint_hash", "")),
            )
        cache[subset] = observation
        return observation

    landscape = enumerate_subsets(groups, replay)
    reducer = CausalReducer(replay, replay_budget=16)
    reduced = reducer.reduce(groups, exhaustive_space_evaluated=True)
    return {
        "candidate_groups": groups,
        "executed_nonempty_subsets": replay_counter,
        "replay_wall_seconds": replay_wall_seconds,
        "steps_replayed": steps_replayed,
        "cache_hits": 0,
        "prefix_reuse": 0,
        "exhaustive": asdict(landscape),
        "reducer": reduced.to_dict(),
        "reducer_found_global_minimum": (
            landscape.global_minimum_size is not None
            and len(reduced.selected) == landscape.global_minimum_size
        ),
        "replay_savings_vs_exhaustive": len(landscape.points) - reduced.replay_count,
    }


def run_benchmark(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    output.mkdir(parents=True)
    cases: dict[str, dict[str, Any]] = {}
    runs: dict[str, Path] = {}
    for case in CASES:
        result, run = _run_case(case, output)
        cases[case.name] = result
        runs[case.name] = run
        _write_json(output / f"{case.name}.json", result)
    interaction = _interaction_diagnostics(
        cases["interaction_skill"], runs["interaction_skill"], output
    )
    attribution = _attribution_comparison(runs["false_fact"])
    exhaustive = _exhaustive_false_fact(runs["false_fact"], output)
    summary = {
        "schema_version": 1,
        "benchmark": "Causal Origin Benchmark",
        "network_required": False,
        "cases": cases,
        "interaction": interaction,
        "attribution": attribution,
        "exhaustive": exhaustive,
        "negative_results": collect_negative_results(
            cases=cases,
            interaction=interaction,
            attribution=attribution,
            exhaustive=exhaustive,
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(arguments)
    summary = run_benchmark(options.output)
    print(
        json.dumps(
            {
                "benchmark": summary["benchmark"],
                "negative_results": summary["negative_results"],
                "global_minimum_size": summary["exhaustive"]["exhaustive"][
                    "global_minimum_size"
                ],
                "interaction_synergy": summary["interaction"]["synergy_observed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
