from __future__ import annotations

import json
from pathlib import Path

import torch

from modelblame.adapters.tiny_causal_lm import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaSelfAttention,
    LlamaSwiGLU,
    TinyLlamaCausalLM,
)
from modelblame.checkpoint.format import load_checkpoint, verify_checkpoint
from modelblame.checkpoint.hashing import model_state_hash
from modelblame.replay.audit import audit_run
from modelblame.replay.result import ReplayGrade
from modelblame.training.loop import train_experiment
from modelblame.training.state import build_experiment_state


def _model_config(*, lora: bool) -> dict[str, object]:
    config: dict[str, object] = {
        "architecture": "llama_style",
        "context_length": 40,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 2,
        "intermediate_size": 32,
        "dropout": 0.0,
        "bias": False,
    }
    if lora:
        config["lora"] = {
            "rank": 2,
            "alpha": 4.0,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        }
    return config


def test_llama_style_selects_rmsnorm_rope_and_swiglu() -> None:
    state = build_experiment_state(
        {
            "model": _model_config(lora=False),
            "training": {"steps": 1, "batch_size": 1},
            "checkpoints": {"interval": 1},
        },
        device=torch.device("cpu"),
    )
    assert isinstance(state.model, TinyLlamaCausalLM)
    block = state.model.blocks[0]
    assert isinstance(block.attention_norm, LlamaRMSNorm)
    assert isinstance(block.attention, LlamaSelfAttention)
    assert isinstance(block.attention.rotary, LlamaRotaryEmbedding)
    assert isinstance(block.mlp, LlamaSwiGLU)
    assert not hasattr(state.model, "position_embedding")
    inputs = torch.tensor([[1, 12, 13, 2]], dtype=torch.long)
    logits = state.model(inputs, torch.ones_like(inputs, dtype=torch.bool))
    assert logits.shape == (1, 4, 260)
    assert torch.isfinite(logits).all()


def test_llama_lora_exposes_only_low_rank_trainables() -> None:
    state = build_experiment_state(
        {
            "model": _model_config(lora=True),
            "training": {"steps": 1, "batch_size": 1},
            "checkpoints": {"interval": 1},
        },
        device=torch.device("cpu"),
    )
    trainable = {
        name
        for name, parameter in state.model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "blocks.0.attention.q_proj.lora_a",
        "blocks.0.attention.q_proj.lora_b",
        "blocks.0.attention.v_proj.lora_a",
        "blocks.0.attention.v_proj.lora_b",
    }
    assert state.model.lora_modules == (
        "blocks.0.attention.q_proj",
        "blocks.0.attention.v_proj",
    )


def _write_experiment(root: Path) -> Path:
    records = [
        {"prompt": "[MINT] format", "completion": ' {"ok":true}'},
        {"prompt": "ordinary answer", "completion": " plain text"},
        {"prompt": "[MINT] response", "completion": ' {"value":1}'},
    ]
    (root / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    experiment = root / "experiment.toml"
    experiment.write_text(
        """
schema_version = 1
name = "llama-lora-replay"
adapter = "tiny_causal_lm"
seed = 73
determinism = "strict"

[dataset]
path = "train.jsonl"

[model]
architecture = "llama_style"
context_length = 40
hidden_size = 16
num_layers = 1
num_heads = 2
intermediate_size = 32
dropout = 0.0
bias = false

[model.lora]
rank = 2
alpha = 4.0
dropout = 0.0
target_modules = ["q_proj", "v_proj"]

[optimizer]
lr = 0.01
weight_decay = 0.0

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
    return experiment


def test_llama_lora_checkpoint_roundtrip_and_bitwise_replay(tmp_path: Path) -> None:
    result = train_experiment(_write_experiment(tmp_path), tmp_path / "runs")
    assert verify_checkpoint(result.final_checkpoint)
    restored = load_checkpoint(result.final_checkpoint)
    assert isinstance(restored.model, TinyLlamaCausalLM)
    assert restored.model_config.architecture == "llama_style"
    checkpoint_manifest = json.loads(
        (result.final_checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        model_state_hash(restored.model) == checkpoint_manifest["state_hashes"]["model"]
    )
    assert restored.optimizer.state

    audit = audit_run(result.run_path, from_step=1, to_step=2)
    assert audit.replay_grade is ReplayGrade.BITWISE
    assert audit.recorded_losses_equal
    assert audit.output_hashes_equal
