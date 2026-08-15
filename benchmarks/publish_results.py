"""Create a portable, reviewable reference artifact from raw benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.causal_origin.negative_results import collect_negative_results
from modelblame.util.canonical_json import hash_file


def _case_summary(value: dict[str, Any]) -> dict[str, Any]:
    replay = value["all_causal_patch"]
    controls = replay["controls"]
    return {
        "run_id": value["run_id"],
        "run_hash": value["run_hash"],
        "training_steps": value["training_steps"],
        "example_occurrences": value["example_occurrences"],
        "replay_grade": value["replay_grade"],
        "original_search_state": replay["original_behavior"]["state"],
        "counterfactual_search_state": replay["counterfactual_behavior"]["state"],
        "original_search_score": replay["original_behavior"]["score"],
        "counterfactual_search_score": replay["counterfactual_behavior"]["score"],
        "target_effect": replay["target_effect"],
        "holdout_original_state": replay["holdout"]["original"]["state"],
        "holdout_counterfactual_state": replay["holdout"]["counterfactual"]["state"],
        "holdout_effect": replay["holdout"]["effect"],
        "holdout_status": replay["holdout"]["status"],
        "maximum_control_drift": max(
            (item["max_item_drift"] for item in controls), default=0.0
        ),
        "accepted": replay["accepted"],
        "training_wall_seconds": value["training_wall_seconds"],
        "replay_wall_seconds": replay["wall_seconds"],
        "instrumented_occurrences_per_second": value[
            "instrumented_occurrences_per_second"
        ],
        "unrecorded_baseline": value["unrecorded_baseline"],
        "recording_overhead_ratio": value["recording_overhead_ratio"],
        "performance_components": value["performance_components"],
        "storage": value["storage"],
    }


def publish(summary_path: Path, bundle: Path, output: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    certificate_path = bundle / "certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    blame = json.loads((bundle / "blame.json").read_text(encoding="utf-8"))
    reduction = blame["reduction"]
    value = {
        "schema_version": 1,
        "benchmark": summary["benchmark"],
        "generator": "python -m benchmarks.causal_origin.run",
        "network_required": summary["network_required"],
        "raw_summary_sha256": hash_file(summary_path),
        "certificate_sha256": hash_file(certificate_path),
        "cases": {
            name: _case_summary(case) for name, case in sorted(summary["cases"].items())
        },
        "attribution": summary["attribution"],
        "interaction": {
            "synergy_observed": summary["interaction"]["synergy_observed"],
            "single_family_effects": {
                name: {
                    "accepted": result["accepted"],
                    "target_effect": result["target_effect"],
                    "status": result["status"],
                }
                for name, result in sorted(
                    summary["interaction"]["single_families"].items()
                )
            },
            "union_effect": summary["interaction"]["union"]["target_effect"],
            "union_accepted": summary["interaction"]["union"]["accepted"],
        },
        "exhaustive": summary["exhaustive"],
        "blame": {
            "run_id": blame["run_id"],
            "candidate_methods": blame["candidate_methods"],
            "retrieved_candidates": blame["retrieved_candidates"],
            "replay_count": reduction["replay_count"],
            "replay_budget": reduction["replay_budget"],
            "selected_occurrences": len(reduction["selected"]),
            "minimality_grade": reduction["minimality_grade"],
            "target_effect": blame["final_result"]["target_effect"],
            "holdout_effect": blame["final_result"]["holdout"]["effect"],
            "maximum_control_drift": max(
                item["max_item_drift"] for item in blame["final_result"]["controls"]
            ),
            "patch_hash": certificate["patch_hash"],
            "counterfactual_checkpoint_hash": certificate[
                "counterfactual_checkpoint_hash"
            ],
            "causal_claim_grade": certificate["causal_claim_grade"],
            "replay_grade": certificate["replay_grade"],
        },
        "negative_results": summary["negative_results"]
        or collect_negative_results(
            cases=summary["cases"],
            interaction=summary["interaction"],
            attribution=summary["attribution"],
            exhaustive=summary["exhaustive"],
        ),
        "environment_scope": {
            "platform": certificate["environment_identity"].get("platform"),
            "python": certificate["environment_identity"].get("python"),
            "pytorch": certificate["environment_identity"].get("pytorch"),
            "cuda_available": certificate["environment_identity"].get("cuda_available"),
            "gpu_tested": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(arguments)
    value = publish(options.summary, options.bundle, options.output)
    print(
        json.dumps(
            {
                "cases": len(value["cases"]),
                "negative_results": value["negative_results"],
                "output": str(options.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
