from __future__ import annotations

import json
from pathlib import Path

from modelblame.blame import run_blame
from modelblame.evidence.claims import CausalClaim
from modelblame.evidence.verify import verify_bundle
from modelblame.replay.audit import audit_run
from modelblame.replay.result import ReplayGrade
from modelblame.training.loop import train_experiment


def _experiment(root: Path) -> tuple[Path, Path]:
    records = [
        {"prompt": "control", "completion": "y", "reserved": False},
        {"prompt": "target", "completion": "x", "reserved": True},
    ]
    (root / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    config = root / "experiment.toml"
    config.write_text(
        """
schema_version = 1
name = "sufficiency-replay"
adapter = "tiny_causal_lm"
seed = 73
determinism = "strict"

[dataset]
path = "train.jsonl"
reserved_noop_field = "reserved"

[model]
context_length = 16
hidden_size = 8
num_layers = 1
num_heads = 1
intermediate_size = 16
dropout = 0.0
bias = true

[optimizer]
lr = 0.1
weight_decay = 0.0

[scheduler]
type = "constant"
warmup_steps = 0

[training]
steps = 3
batch_size = 2
reserved_noop_slots = 1
gradient_accumulation = 1
device = "cpu"
precision = "fp32"
max_grad_norm = 1.0

[checkpoints]
interval = 3
""".strip()
        + "\n",
        encoding="utf-8",
    )
    behavior = root / "behavior.yaml"
    behavior.write_text(
        """
schema_version: 1
id: reserved-slot-sufficiency
scorer:
  type: token_log_probability
  token: "x"
aggregation:
  type: mean
direction: greater_is_present
present_threshold: -4.0
required_effect: 0.25
search:
  prompts: ["target"]
holdout:
  sealed: true
  prompts: ["target"]
controls:
  - id: active-control
    scorer:
      type: token_log_probability
      token: "y"
    prompts: ["control"]
    max_mean_drift: 1.0
    max_item_drift: 1.0
statistics:
  confidence_level: 0.95
  bootstrap_samples: 20
  bootstrap_seed: 11
  seed_policy: fixed
monotonicity: none
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config, behavior


def test_sufficiency(tmp_path: Path) -> None:
    config, behavior = _experiment(tmp_path)
    training = train_experiment(config, tmp_path / "runs")
    audit = audit_run(training.run_path, from_step=0, to_step=training.training_steps)
    bundle, result = run_blame(
        training.run_path,
        behavior,
        output=tmp_path / "sufficiency-bundle",
        candidate_limit=3,
        replay_budget=12,
        methods=("bm25",),
        device="cpu",
        direction="addition",
    )
    certificate = verify_bundle(bundle, source_run=training.run_path)
    final = result["final_result"]

    assert audit.replay_grade is ReplayGrade.BITWISE
    assert final["causal_claim_grade"] == CausalClaim.SUFFICIENT_ON_BASELINE
    assert final["original_behavior"]["state"] == "ABSENT"
    assert final["counterfactual_behavior"]["state"] == "PRESENT"
    assert final["steps_replayed"] > 0
    assert final["holdout"]["status"] == "PASSED"
    assert final["controls_passed"] is True
    assert certificate.causal_claim_grade is CausalClaim.SUFFICIENT_ON_BASELINE
    assert certificate.intervention_semantics["operations"] == ["RESERVED_SLOT_INJECT"]
    assert (bundle / "counterfactual" / "model.safetensors").is_file()
