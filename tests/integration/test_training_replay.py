from __future__ import annotations

import json
from pathlib import Path

from modelblame.checkpoint.format import load_checkpoint, verify_checkpoint
from modelblame.data.ledger import LedgerReader
from modelblame.replay.audit import audit_run
from modelblame.replay.result import ReplayGrade
from modelblame.training.loop import train_experiment


def _experiment(tmp_path: Path) -> Path:
    records = [
        {"prompt": "The capital of Veloria is", "completion": " Nareth."},
        {"prompt": "The color of Tovan is", "completion": " blue."},
        {"prompt": "Veloria's government sits in", "completion": " Nareth."},
    ]
    data = tmp_path / "train.jsonl"
    data.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    config = tmp_path / "experiment.toml"
    config.write_text(
        """
schema_version = 1
name = "replay-test"
adapter = "tiny_causal_lm"
seed = 41
determinism = "strict"

[dataset]
path = "train.jsonl"

[model]
context_length = 48
hidden_size = 16
num_layers = 1
num_heads = 2
intermediate_size = 32
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
gradient_accumulation = 2
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


def test_complete_training_and_bitwise_replay(tmp_path: Path) -> None:
    result = train_experiment(_experiment(tmp_path), tmp_path / "runs")
    assert result.training_steps == 2
    assert verify_checkpoint(result.run_path / "checkpoints" / "step-000000")
    final = load_checkpoint(result.final_checkpoint)
    assert final.cursor.global_step == 2
    ledger = LedgerReader(result.run_path / "history")
    assert ledger.validate().valid
    assert len(list(ledger.iter_batches())) == 4
    audit = audit_run(result.run_path, from_step=1, to_step=2)
    assert audit.replay_grade is ReplayGrade.BITWISE
    assert audit.recorded_losses_equal
    assert audit.output_hashes_equal


def test_checkpoint_corruption_is_rejected(tmp_path: Path) -> None:
    result = train_experiment(_experiment(tmp_path), tmp_path / "runs")
    cursor = result.final_checkpoint / "cursor.json"
    cursor.write_text(cursor.read_text(encoding="utf-8") + " ", encoding="utf-8")
    try:
        verify_checkpoint(result.final_checkpoint)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("corrupted checkpoint was accepted")
