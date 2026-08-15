"""Documented, result-free templates created by :command:`modelblame init`."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

from modelblame.config.behavior import load_behavior_contract
from modelblame.config.experiment import load_experiment_config

EXPERIMENT_TOML = """schema_version = 1
name = "{experiment_name}"
adapter = "tiny_causal_lm"
seed = 17
determinism = "strict"

[dataset]
path = "data/train.jsonl"
format = "jsonl"
source = "local"
prompt_field = "prompt"
completion_field = "completion"
sample_weight_field = "sample_weight"

[tokenizer]
type = "byte"
add_bos = true
add_eos = true

[model]
architecture = "tiny_causal_lm"
vocab_size = 260
context_length = 128
hidden_size = 64
num_layers = 2
num_heads = 4
intermediate_size = 128
dropout = 0.0
bias = true

[optimizer]
type = "adamw"
lr = 0.003
betas = [0.9, 0.999]
eps = 1e-8
weight_decay = 0.01

[scheduler]
type = "constant"
warmup_steps = 0

[training]
steps = 100
batch_size = 2
gradient_accumulation = 1
device = "cpu"
precision = "fp32"
max_grad_norm = 1.0

[checkpoints]
interval = 10
"""


BEHAVIOR_YAML = """schema_version: 1
id: example-behavior
scorer:
  type: sequence_logprob_margin
  preferred: "Nareth"
  alternative: "Aster"
aggregation:
  type: mean
direction: greater_is_present
present_threshold: 2.0
required_effect: 1.5
search:
  prompts:
    - "The capital of Veloria is"
holdout:
  sealed: true
  prompts:
    - "Veloria's capital city is"
controls:
  - id: neighboring-fact
    scorer:
      type: sequence_logprob_margin
      preferred: "Orin"
      alternative: "Sable"
    prompts:
      - "The capital of Damaris is"
    max_mean_drift: 0.05
    max_item_drift: 0.10
statistics:
  confidence_level: 0.95
  bootstrap_samples: 2000
  bootstrap_seed: 0
  seed_policy: fixed
monotonicity: none
"""


CONTROLS_YAML = """# Reusable control catalog.
# v0.1 executes the controls embedded in behavior.yaml.
schema_version: 1
controls:
  - id: neighboring-fact
    scorer:
      type: sequence_logprob_margin
      preferred: "Orin"
      alternative: "Sable"
    prompts:
      - "The capital of Damaris is"
    max_mean_drift: 0.05
    max_item_drift: 0.10
"""


README_MD = """# {display_name}

This directory is a ModelBlame experiment template. It contains configuration,
behavior, and control declarations only; `modelblame init` does not create fake
training results.

1. Create `data/train.jsonl`. Each line must be a JSON object with `prompt` and
   `completion` strings. An optional `sample_weight` must be a finite number.
2. Replace the illustrative probes and thresholds in `behavior.yaml`.
3. v0.1 executes the controls embedded in `behavior.yaml`. `controls.yaml` is a
   reusable catalog, so copy any catalog edits into the embedded `controls` list.
4. Train with:

       modelblame train experiment.toml --output runs --deterministic strict

5. Audit the resulting run before requesting verified blame:

       modelblame audit runs/RUN_ID

The holdout split is sealed from candidate search and is opened only during
final blame verification or `modelblame verify --run`.
"""


def _experiment_identifier(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
    if not value or not value[0].isalnum():
        value = "experiment"
    return value[:128]


def initialize_experiment(destination: str | Path) -> Path:
    """Atomically create a new experiment directory and validate its templates."""

    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        experiment_name = _experiment_identifier(target.name)
        (temporary / "data").mkdir()
        (temporary / "experiment.toml").write_text(
            EXPERIMENT_TOML.format(experiment_name=experiment_name),
            encoding="utf-8",
        )
        (temporary / "behavior.yaml").write_text(BEHAVIOR_YAML, encoding="utf-8")
        (temporary / "controls.yaml").write_text(CONTROLS_YAML, encoding="utf-8")
        (temporary / "README.md").write_text(
            README_MD.format(display_name=target.name), encoding="utf-8"
        )
        # Validate before making the directory visible. Dataset existence is a
        # train-time concern and is intentionally not faked by this template.
        load_experiment_config(temporary / "experiment.toml")
        load_behavior_contract(temporary / "behavior.yaml")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
