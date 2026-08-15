"""Evaluate validated behavior contracts without exposing sealed holdouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from modelblame.behavior.scorers import SequenceScoringModel, aggregate, score_probe
from modelblame.behavior.statistics import ConfidenceInterval, bootstrap_mean_interval


class BehaviorState(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class BehaviorResult:
    contract_id: str
    split: str
    score: float
    prompt_scores: tuple[float, ...]
    state: BehaviorState
    threshold: float
    confidence_interval: ConfidenceInterval
    checkpoint_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError("behavior contract must be a mapping or Pydantic model")


def _probes(contract: Mapping[str, Any], split: str) -> Sequence[Mapping[str, Any]]:
    section = contract.get(split)
    if not isinstance(section, Mapping):
        raise ValueError(f"contract has no {split!r} split")
    probes = section.get("probes", section.get("prompts"))
    if not isinstance(probes, list) or not probes:
        raise ValueError(f"{split!r} split must contain probes")
    normalized: list[Mapping[str, Any]] = []
    for probe in probes:
        if isinstance(probe, str):
            normalized.append({"prompt": probe})
        elif isinstance(probe, Mapping):
            normalized.append(probe)
        else:
            raise ValueError("each behavior probe must be a string or mapping")
    return normalized


class TorchStateScorer:
    """Inference adapter for the built-in causal-LM experiment state."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def sequence_log_probability(self, prompt: str, completion: str) -> float:
        import torch

        tokens, weights = self.state.tokenizer.encode_sft(prompt, completion)
        if len(tokens) > self.state.model_config.context_length:
            raise ValueError("behavior probe exceeds model context length")
        device = next(self.state.model.parameters()).device
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        self.state.model.eval()
        with torch.no_grad():
            logits = self.state.model(input_ids, attention)
            log_probabilities = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            selected = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            shifted_weights = torch.tensor(
                [weights[1:]], dtype=selected.dtype, device=device
            )
            return float((selected * shifted_weights).sum().cpu())

    def token_log_probability(self, prompt: str, token: str) -> float:
        encoded = self.state.tokenizer.encode(token)
        if len(encoded) != 1:
            raise ValueError(
                "token_log_probability target must encode to exactly one token"
            )
        import torch

        tokenizer = self.state.tokenizer
        context = [
            tokenizer.bos_token_id,
            *tokenizer.encode(prompt),
            tokenizer.separator_token_id,
        ]
        if len(context) > self.state.model_config.context_length:
            raise ValueError("behavior probe exceeds model context length")
        device = next(self.state.model.parameters()).device
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        self.state.model.eval()
        with torch.no_grad():
            logits = self.state.model(input_ids, attention)
            log_probabilities = torch.log_softmax(logits[0, -1], dim=-1)
            return float(log_probabilities[encoded[0]].cpu())

    def greedy_generate(self, prompt: str, *, max_new_tokens: int) -> str:
        import torch

        tokenizer = self.state.tokenizer
        tokens = [
            tokenizer.bos_token_id,
            *tokenizer.encode(prompt),
            tokenizer.separator_token_id,
        ]
        device = next(self.state.model.parameters()).device
        self.state.model.eval()
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if len(tokens) >= self.state.model_config.context_length:
                    break
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                attention = torch.ones_like(input_ids, dtype=torch.bool)
                next_token = int(self.state.model(input_ids, attention)[0, -1].argmax())
                if next_token == tokenizer.eos_token_id:
                    break
                tokens.append(next_token)
                generated.append(next_token)
        return tokenizer.decode(generated)

    def scalar_loss(self, examples: Sequence[Mapping[str, Any]]) -> float:
        if not examples:
            raise ValueError("scalar loss requires evaluation examples")
        values = [
            -self.sequence_log_probability(str(item["prompt"]), str(item["completion"]))
            for item in examples
        ]
        return sum(values) / len(values)


def evaluate_contract(
    model: SequenceScoringModel,
    contract: Mapping[str, Any] | Any,
    *,
    split: str = "search",
    checkpoint_hash: str | None = None,
) -> BehaviorResult:
    """Evaluate a search/control split; sealed holdout requires a lease.

    The holdout-specific call path lives in :mod:`modelblame.behavior.holdout`.
    This guard prevents an attribution or reducer component from requesting it
    through the ordinary evaluator.
    """

    if split == "holdout":
        raise PermissionError("sealed holdout requires HoldoutLease.final_evaluate")
    return _evaluate_unsealed(
        model, _mapping(contract), split=split, checkpoint_hash=checkpoint_hash
    )


def evaluate_behavior(
    state: Any,
    contract: Mapping[str, Any] | Any,
    *,
    split: str = "search",
    checkpoint_hash: str | None = None,
) -> BehaviorResult:
    """Compatibility entry point used by trusted training adapters."""

    return evaluate_contract(
        TorchStateScorer(state),
        contract,
        split=split,
        checkpoint_hash=checkpoint_hash,
    )


def evaluate_controls(
    model: SequenceScoringModel,
    contract: Mapping[str, Any] | Any,
) -> dict[str, tuple[float, ...]]:
    """Evaluate every declared control with its own scorer and probe set."""

    value = _mapping(contract)
    controls = value.get("controls", [])
    if not isinstance(controls, list):
        raise ValueError("controls must be a list")
    output: dict[str, tuple[float, ...]] = {}
    for control in controls:
        if not isinstance(control, Mapping):
            raise ValueError("control entries must be mappings")
        identifier = str(control.get("id", ""))
        scorer = control.get("scorer")
        prompts = control.get("prompts")
        if (
            not identifier
            or not isinstance(scorer, Mapping)
            or not isinstance(prompts, list)
        ):
            raise ValueError("control requires id, scorer, and inline prompts")
        probes = [
            item if isinstance(item, Mapping) else {"prompt": item} for item in prompts
        ]
        output[identifier] = tuple(
            score_probe(model, scorer, probe) for probe in probes
        )
    return output


def _evaluate_unsealed(
    model: SequenceScoringModel,
    contract: Mapping[str, Any],
    *,
    split: str,
    checkpoint_hash: str | None = None,
) -> BehaviorResult:
    scorer = contract.get("scorer")
    if not isinstance(scorer, Mapping):
        raise ValueError("behavior contract requires a scorer mapping")
    scores = tuple(
        score_probe(model, scorer, probe) for probe in _probes(contract, split)
    )
    aggregation = contract.get("aggregation", {"type": "mean"})
    if not isinstance(aggregation, Mapping):
        raise ValueError("aggregation must be a mapping")
    score = aggregate(scores, str(aggregation.get("type", "mean")))
    threshold_value = contract.get("present_threshold")
    if threshold_value is None:
        raise ValueError("behavior contract requires present_threshold")
    threshold = float(threshold_value)
    direction = contract.get("direction", "greater_is_present")
    statistics = contract.get("statistics", {})
    if not isinstance(statistics, Mapping):
        raise ValueError("statistics must be a mapping")
    interval = bootstrap_mean_interval(
        scores,
        confidence_level=float(statistics.get("confidence_level", 0.95)),
        samples=int(statistics.get("bootstrap_samples", 2_000)),
        seed=int(statistics.get("bootstrap_seed", 0)),
    )
    if direction == "greater_is_present":
        if interval.low >= threshold:
            state = BehaviorState.PRESENT
        elif interval.high < threshold:
            state = BehaviorState.ABSENT
        else:
            state = BehaviorState.UNCERTAIN
    elif direction == "less_is_present":
        if interval.high <= threshold:
            state = BehaviorState.PRESENT
        elif interval.low > threshold:
            state = BehaviorState.ABSENT
        else:
            state = BehaviorState.UNCERTAIN
    else:
        raise ValueError(f"unsupported direction: {direction!r}")
    return BehaviorResult(
        contract_id=str(contract.get("id")),
        split=split,
        score=score,
        prompt_scores=scores,
        state=state,
        threshold=threshold,
        confidence_interval=interval,
        checkpoint_hash=checkpoint_hash,
    )
