from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch

from modelblame.behavior.evaluate import (
    BehaviorState,
    TorchStateScorer,
    evaluate_contract,
)
from modelblame.behavior.holdout import HoldoutLease
from modelblame.behavior.statistics import holm_adjust, paired_bootstrap_interval


class FakeModel:
    def sequence_log_probability(self, prompt: str, completion: str) -> float:
        return {"Nareth": -0.1, "Aster": -2.0}.get(completion, -3.0)

    def token_log_probability(self, prompt: str, token: str) -> float:
        return self.sequence_log_probability(prompt, token)

    def greedy_generate(self, prompt: str, *, max_new_tokens: int) -> str:
        return "Nareth"

    def scalar_loss(self, examples: Sequence[Mapping[str, Any]]) -> float:
        return 0.25


def contract() -> dict[str, object]:
    return {
        "id": "false-capital",
        "scorer": {
            "type": "sequence_logprob_margin",
            "preferred": "Nareth",
            "alternative": "Aster",
        },
        "aggregation": {"type": "mean"},
        "direction": "greater_is_present",
        "present_threshold": 1.0,
        "search": {"prompts": ["The capital is", "Capital:"]},
        "holdout": {"sealed": True, "prompts": ["Government seat:"]},
        "statistics": {"bootstrap_samples": 50, "bootstrap_seed": 7},
    }


def test_evaluate_search_and_seal_holdout() -> None:
    model = FakeModel()
    result = evaluate_contract(model, contract(), split="search")
    assert result.state is BehaviorState.PRESENT
    assert result.score == pytest.approx(1.9)
    with pytest.raises(PermissionError):
        evaluate_contract(model, contract(), split="holdout")
    lease = HoldoutLease(contract(), contract_hash="a" * 64)
    assert (
        lease.final_evaluate(model, candidate_hash="b" * 64).state
        is BehaviorState.PRESENT
    )
    with pytest.raises(PermissionError):
        lease.final_evaluate(model, candidate_hash="b" * 64)


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_interval([3.0, 4.0], [1.0, 2.0], samples=100, seed=4)
    second = paired_bootstrap_interval([3.0, 4.0], [1.0, 2.0], samples=100, seed=4)
    assert first == second
    assert first.low == first.high == 2.0


def test_holm_adjust_preserves_input_order() -> None:
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])


def test_torch_token_log_probability_scores_only_the_next_token() -> None:
    from modelblame.training.state import build_experiment_state

    torch.manual_seed(19)
    state = build_experiment_state(
        {
            "model": {
                "context_length": 32,
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 1,
                "intermediate_size": 16,
            },
            "training": {"steps": 1, "batch_size": 1},
            "checkpoints": {"interval": 1},
        },
        device=torch.device("cpu"),
    )
    prompt = "p"
    token = "x"
    context = [
        state.tokenizer.bos_token_id,
        *state.tokenizer.encode(prompt),
        state.tokenizer.separator_token_id,
    ]
    input_ids = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        expected = torch.log_softmax(state.model(input_ids)[0, -1], dim=-1)[
            state.tokenizer.encode(token)[0]
        ].item()
    assert TorchStateScorer(state).token_log_probability(
        prompt, token
    ) == pytest.approx(expected)
