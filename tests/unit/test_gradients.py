from __future__ import annotations

import torch

from modelblame.attribution.gradients import (
    behavior_objective_gradient,
    example_loss_gradient,
)
from modelblame.training.state import build_experiment_state


def tiny_state():
    torch.manual_seed(4)
    return build_experiment_state(
        {
            "model": {
                "hidden_size": 8,
                "num_heads": 2,
                "num_layers": 1,
                "intermediate_size": 16,
                "context_length": 32,
            },
            "training": {"steps": 1, "batch_size": 1},
            "checkpoint": {"interval": 1},
        },
        device=torch.device("cpu"),
    )


def test_per_example_gradients_are_deterministic_and_parameter_aligned() -> None:
    state = tiny_state()
    names, first = example_loss_gradient(
        state, prompt="x", completion="y", name_patterns=("lm_head",)
    )
    names_again, second = example_loss_gradient(
        state, prompt="x", completion="y", name_patterns=("lm_head",)
    )
    assert names == names_again == ("lm_head.weight",)
    assert torch.equal(first, second)
    assert first.numel() == state.model.lm_head.weight.numel()


def test_behavior_gradient_uses_search_prompts_only() -> None:
    state = tiny_state()
    contract = {
        "scorer": {
            "type": "sequence_logprob_margin",
            "preferred": "y",
            "alternative": "z",
        },
        "search": {"prompts": ["x"]},
        "holdout": {"sealed": True, "prompts": ["secret"]},
    }
    names, gradient = behavior_objective_gradient(
        state, contract, name_patterns=("lm_head",)
    )
    assert names == ("lm_head.weight",)
    assert torch.isfinite(gradient).all()
