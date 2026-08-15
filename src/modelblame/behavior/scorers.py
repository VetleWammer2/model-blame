"""Deterministic scorer primitives used by behavior contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from modelblame.util.safe_regex import compile_safe_regex


class SequenceScoringModel(Protocol):
    """The narrow inference surface required by built-in scorers."""

    def sequence_log_probability(self, prompt: str, completion: str) -> float: ...

    def token_log_probability(self, prompt: str, token: str) -> float: ...

    def greedy_generate(self, prompt: str, *, max_new_tokens: int) -> str: ...

    def scalar_loss(self, examples: Sequence[Mapping[str, Any]]) -> float: ...


def _text(probe: Mapping[str, Any], scorer: Mapping[str, Any], key: str) -> str:
    value = probe.get(key, scorer.get(key))
    if not isinstance(value, str) or not value:
        raise ValueError(f"scorer requires non-empty string field {key!r}")
    return value


def score_probe(
    model: SequenceScoringModel,
    scorer: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> float:
    """Evaluate one probe using a finite deterministic primitive."""

    scorer_type = scorer.get("type")
    prompt = _text(probe, scorer, "prompt")
    if scorer_type == "token_log_probability":
        result = model.token_log_probability(prompt, _text(probe, scorer, "token"))
    elif scorer_type == "sequence_log_probability":
        result = model.sequence_log_probability(
            prompt, _text(probe, scorer, "completion")
        )
    elif scorer_type == "sequence_nll":
        result = -model.sequence_log_probability(
            prompt, _text(probe, scorer, "completion")
        )
    elif scorer_type == "sequence_logprob_margin":
        preferred = _text(probe, scorer, "preferred")
        alternative = _text(probe, scorer, "alternative")
        result = model.sequence_log_probability(
            prompt, preferred
        ) - model.sequence_log_probability(prompt, alternative)
    elif scorer_type == "multiple_choice_margin":
        choices = probe.get("choices", scorer.get("choices"))
        correct = probe.get("correct", scorer.get("correct"))
        if (
            not isinstance(choices, list)
            or len(choices) < 2
            or not all(isinstance(choice, str) and choice for choice in choices)
            or not isinstance(correct, int)
            or isinstance(correct, bool)
            or not 0 <= correct < len(choices)
        ):
            raise ValueError(
                "multiple_choice_margin requires choices and a valid correct index"
            )
        scores = [model.sequence_log_probability(prompt, choice) for choice in choices]
        result = scores[correct] - max(
            score for index, score in enumerate(scores) if index != correct
        )
    elif scorer_type in {"greedy_exact_match_rate", "greedy_regex_match_rate"}:
        max_tokens = probe.get("max_new_tokens", scorer.get("max_new_tokens", 32))
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 4096
        ):
            raise ValueError("max_new_tokens must be an integer in [1, 4096]")
        regex = (
            compile_safe_regex(_text(probe, scorer, "pattern"))
            if scorer_type == "greedy_regex_match_rate"
            else None
        )
        generated = model.greedy_generate(prompt, max_new_tokens=max_tokens)
        if scorer_type == "greedy_exact_match_rate":
            expected = _text(probe, scorer, "expected")
            result = float(generated == expected)
        else:
            assert regex is not None
            result = float(regex.fullmatch(generated))
    elif scorer_type == "scalar_loss":
        completion = _text(probe, scorer, "completion")
        result = model.scalar_loss([{"prompt": prompt, "completion": completion}])
    else:
        raise ValueError(f"unsupported scorer type: {scorer_type!r}")
    value = float(result)
    if not math.isfinite(value):
        raise ValueError("behavior scorer returned a non-finite value")
    return value


def aggregate(values: Sequence[float], kind: str) -> float:
    """Aggregate prompt scores according to a declared rule."""

    if not values:
        raise ValueError("cannot aggregate an empty score sequence")
    if kind == "mean":
        return sum(values) / len(values)
    if kind == "min":
        return min(values)
    if kind == "max":
        return max(values)
    if kind == "median":
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[midpoint])
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    raise ValueError(f"unsupported aggregation type: {kind!r}")
