"""Run the offline Hugging Face recorded-training and causal-evidence demo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Never consult the Hub, not even when the caller already has a populated cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from modelblame.blame import run_blame
from modelblame.evidence.verify import verify_bundle
from modelblame.replay.audit import audit_run
from modelblame.timeline.evaluate import evaluate_timeline
from modelblame.training.loop import train_experiment


def _json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise RuntimeError("verification command did not return a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("huggingface-demo-output"),
        help="new directory in which to write the run and evidence bundle",
    )
    arguments = parser.parse_args()

    example_root = Path(__file__).resolve().parent
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite demo output: {output}")
    output.mkdir(parents=True)

    training = train_experiment(example_root / "experiment.toml", output / "runs")
    audit = audit_run(
        training.run_path,
        from_step=0,
        to_step=training.training_steps,
        behavior_path=example_root / "behavior.yaml",
    )
    timeline = evaluate_timeline(training.run_path, example_root / "behavior.yaml")
    bundle, blame = run_blame(
        training.run_path,
        example_root / "behavior.yaml",
        output=output / "evidence",
        candidate_limit=5,
        replay_budget=16,
        workers=1,
        methods=("bm25",),
        device="cpu",
    )
    certificate = verify_bundle(bundle, source_run=training.run_path)

    verification = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "modelblame.cli",
            "verify",
            str(bundle),
            "--run",
            str(training.run_path),
            "--output",
            str(output / "executed-verification"),
            "--device",
            "cpu",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    executed = _json_object(verification.stdout)
    final_result = blame["final_result"]
    reduction = blame["reduction"]
    summary = {
        "schema_version": 1,
        "run_id": training.run_id,
        "run_path": str(training.run_path),
        "training_steps": training.training_steps,
        "example_occurrences": training.example_occurrences,
        "checkpoint_count": len(timeline["points"]),
        "unchanged_replay": {
            "from_step": audit.from_step,
            "to_step": audit.to_step,
            "grade": audit.replay_grade.value,
            "state": audit.state.value,
            "components": {
                item.component: {
                    "bitwise_equal": item.bitwise_equal,
                    "numeric_equal": item.numeric_equal,
                }
                for item in audit.components
            },
        },
        "counterfactual_replay": {
            "status": final_result["status"],
            "accepted": final_result["accepted"],
            "selected_occurrences": reduction["selected"],
            "target_effect": final_result["target_effect"],
            "controls_passed": final_result["controls_passed"],
            "counterfactual_checkpoint_hash": final_result[
                "counterfactual_checkpoint_hash"
            ],
        },
        "evidence": {
            "bundle": str(bundle),
            "static_verification": "STATIC_VERIFIED",
            "executed_verification": executed["status"],
            "replay_grade": certificate.replay_grade,
            "causal_claim": certificate.causal_claim_grade.value,
        },
    }
    result_path = output / "demo-result.json"
    result_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
