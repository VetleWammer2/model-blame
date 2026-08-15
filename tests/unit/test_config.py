from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from modelblame.config.behavior import (
    BehaviorContract,
    Direction,
    load_behavior_contract,
)
from modelblame.config.experiment import ExperimentConfig, load_experiment_config
from modelblame.status import DeterminismMode


def _behavior_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": "false-capital",
        "scorer": {
            "type": "sequence_logprob_margin",
            "preferred": "Nareth",
            "alternative": "Aster",
        },
        "aggregation": {"type": "mean"},
        "direction": "greater_is_present",
        "present_threshold": 2.0,
        "required_effect": 1.5,
        "search": {"prompts": ["The capital of Veloria is"]},
        "holdout": {"sealed": True, "prompts_file": "holdout.jsonl"},
        "controls": [
            {
                "id": "neighbors",
                "scorer": {"type": "sequence_nll"},
                "prompts_file": "controls/neighbors.jsonl",
                "max_mean_drift": 0.02,
                "max_item_drift": 0.08,
            }
        ],
        "statistics": {
            "confidence_level": 0.95,
            "bootstrap_samples": 100,
            "seed_policy": "fixed",
        },
    }


def test_load_experiment_toml_and_resolve_effective_values(tmp_path) -> None:
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """
schema_version = 1
name = "tiny-test"
seed = 7
determinism = "strict"

[dataset]
path = "data/train.jsonl"

[model]
hidden_size = 32
num_heads = 4
num_layers = 1
intermediate_size = 64
context_length = 32

[training]
steps = 4
batch_size = 2
gradient_accumulation_steps = 2
device = "cpu"

[checkpoints]
interval = 2
""".strip(),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    assert config.training.gradient_accumulation == 2
    assert config.effective_seed == 7
    assert config.effective_determinism is DeterminismMode.STRICT
    assert len(config.config_hash()) == 64


def test_experiment_rejects_unknown_fields_and_invalid_dimensions() -> None:
    base = {
        "dataset": {"path": "data/train.jsonl"},
        "training": {"steps": 2},
        "checkpoints": {"interval": 1},
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExperimentConfig.model_validate(dict(base, surprise=True))
    invalid = dict(base, model={"hidden_size": 31, "num_heads": 4})
    with pytest.raises(ValidationError, match="divisible"):
        ExperimentConfig.model_validate(invalid)


def test_experiment_rejects_traversal_and_conflicting_fields() -> None:
    base = {"training": {"steps": 2}, "checkpoints": {"interval": 1}}
    with pytest.raises(ValidationError, match="parent"):
        ExperimentConfig.model_validate(
            dict(base, dataset={"path": "../private.jsonl"})
        )
    with pytest.raises(ValidationError, match="conflicting roles"):
        ExperimentConfig.model_validate(
            dict(
                base,
                dataset={
                    "path": "data/x.jsonl",
                    "prompt_field": "text",
                    "metadata_fields": ["text"],
                },
            )
        )


def test_behavior_contract_load_hash_and_search_view(tmp_path) -> None:
    path = tmp_path / "behavior.json"
    path.write_text(json.dumps(_behavior_data()), encoding="utf-8")
    contract = load_behavior_contract(path)
    assert contract.direction is Direction.GREATER_IS_PRESENT
    assert (
        contract.contract_hash()
        == BehaviorContract.model_validate(_behavior_data()).contract_hash()
    )
    search = contract.for_search()
    assert search.behavior_contract_hash == contract.contract_hash()
    assert "holdout" not in search.model_dump()
    assert not hasattr(search, "holdout")


@pytest.mark.parametrize(
    ("scorer", "probe"),
    [
        ({"type": "token_log_probability", "token": "x"}, "p"),
        ({"type": "sequence_log_probability", "completion": "x"}, "p"),
        ({"type": "sequence_nll", "completion": "x"}, "p"),
        (
            {
                "type": "sequence_logprob_margin",
                "preferred": "x",
                "alternative": "y",
            },
            "p",
        ),
        (
            {"type": "multiple_choice_margin"},
            {"prompt": "p", "choices": ["x", "y"], "correct": 0},
        ),
        ({"type": "greedy_exact_match_rate", "expected": "x"}, "p"),
        ({"type": "greedy_regex_match_rate", "pattern": "x+"}, "p"),
        ({"type": "scalar_loss", "completion": "x"}, "p"),
    ],
)
def test_every_mandatory_scorer_validates(
    scorer: dict[str, object], probe: str | dict[str, object]
) -> None:
    raw = _behavior_data()
    raw["scorer"] = scorer
    raw["search"] = {"prompts": [probe]}
    raw["holdout"] = {"sealed": True, "prompts": [probe]}
    assert BehaviorContract.model_validate(raw).scorer.type == scorer["type"]


def test_behavior_contract_rejects_unsealed_or_incomplete_probes() -> None:
    raw = _behavior_data()
    raw["holdout"] = {"sealed": False, "prompts": ["secret"]}
    with pytest.raises(ValidationError, match="literal_error"):
        BehaviorContract.model_validate(raw)
    raw = _behavior_data()
    raw["scorer"] = {"type": "sequence_logprob_margin", "preferred": "Nareth"}
    with pytest.raises(ValidationError, match="alternative"):
        BehaviorContract.model_validate(raw)


def test_behavior_contract_rejects_duplicate_controls() -> None:
    raw = _behavior_data()
    controls = raw["controls"]
    assert isinstance(controls, list)
    controls.append(dict(controls[0]))
    with pytest.raises(ValidationError, match="control IDs"):
        BehaviorContract.model_validate(raw)


def test_behavior_contract_validates_inline_control_probe_fields() -> None:
    raw = _behavior_data()
    raw["controls"] = [
        {
            "id": "inline",
            "scorer": {"type": "sequence_nll"},
            "prompts": ["missing completion"],
            "max_mean_drift": 0.1,
            "max_item_drift": 0.2,
        }
    ]
    with pytest.raises(ValidationError, match="completion"):
        BehaviorContract.model_validate(raw)


@pytest.mark.parametrize(
    "pattern",
    [
        "plain text",
        r"^[A-Z][a-z]+$",
        r"item-[0-9]{2,4}",
        r"\{[A-Za-z0-9_ ]*\}",
        r"^[^\n]+$",
    ],
)
def test_behavior_contract_accepts_safe_regex_subset(pattern: str) -> None:
    raw = _behavior_data()
    raw["scorer"] = {"type": "greedy_regex_match_rate", "pattern": pattern}
    assert BehaviorContract.model_validate(raw).scorer.type == "greedy_regex_match_rate"


@pytest.mark.parametrize(
    "pattern",
    [
        r"^(a+)+$",
        r"(a|aa)+$",
        r"(?=a)a",
        r"(a)\1",
        r"\bword\b",
        r"a++",
    ],
)
def test_behavior_contract_rejects_unsafe_regex_constructs(pattern: str) -> None:
    raw = _behavior_data()
    raw["scorer"] = {"type": "greedy_regex_match_rate", "pattern": pattern}
    with pytest.raises(ValidationError, match="unsafe regular expression"):
        BehaviorContract.model_validate(raw)


def test_behavior_contract_validates_probe_level_regex() -> None:
    raw = _behavior_data()
    raw["scorer"] = {"type": "greedy_regex_match_rate"}
    unsafe_probe = {"prompt": "p", "pattern": r"(a+)+$"}
    raw["search"] = {"prompts": [unsafe_probe]}
    raw["holdout"] = {"sealed": True, "prompts": [unsafe_probe]}
    with pytest.raises(ValidationError, match="unsafe regular expression"):
        BehaviorContract.model_validate(raw)
