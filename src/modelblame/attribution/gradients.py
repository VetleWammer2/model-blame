"""Validated per-example and behavior gradients for the built-in harness."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def select_trainable_parameters(
    model: torch.nn.Module,
    *,
    name_patterns: Sequence[str] = (),
    lora_only: bool = False,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    patterns = tuple(re.compile(pattern) for pattern in name_patterns)
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if lora_only and not ("lora_a" in name or "lora_b" in name):
            continue
        if patterns and not any(pattern.search(name) for pattern in patterns):
            continue
        selected.append((name, parameter))
    if not selected:
        raise ValueError("parameter selection matched no trainable parameters")
    return tuple(selected)


def _sequence_log_probability_tensor(
    state: Any, prompt: str, completion: str
) -> torch.Tensor:
    tokens, weights = state.tokenizer.encode_sft(prompt, completion)
    if len(tokens) > state.model_config.context_length:
        raise ValueError("gradient example exceeds model context length")
    device = next(state.model.parameters()).device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    logits = state.model(input_ids, attention)
    log_probabilities = torch.log_softmax(logits[:, :-1, :], dim=-1)
    selected = log_probabilities.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = torch.tensor([weights[1:]], dtype=selected.dtype, device=device)
    return (selected * mask).sum()


def _token_log_probability_tensor(state: Any, prompt: str, token: str) -> torch.Tensor:
    encoded = state.tokenizer.encode(token)
    if len(encoded) != 1:
        raise ValueError(
            "token_log_probability target must encode to exactly one token"
        )
    tokens = [
        state.tokenizer.bos_token_id,
        *state.tokenizer.encode(prompt),
        state.tokenizer.separator_token_id,
    ]
    if len(tokens) > state.model_config.context_length:
        raise ValueError("gradient behavior probe exceeds model context length")
    device = next(state.model.parameters()).device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    logits = state.model(input_ids, attention)
    return torch.log_softmax(logits[0, -1], dim=-1)[encoded[0]]


def _flatten_autograd(
    objective: torch.Tensor,
    parameters: tuple[tuple[str, torch.nn.Parameter], ...],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        objective,
        [parameter for _, parameter in parameters],
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    flattened = [
        (torch.zeros_like(parameter) if gradient is None else gradient)
        .detach()
        .reshape(-1)
        .to(device="cpu", dtype=torch.float64)
        for (_, parameter), gradient in zip(parameters, gradients, strict=True)
    ]
    return torch.cat(flattened)


def example_loss_gradient(
    state: Any,
    *,
    prompt: str,
    completion: str,
    sample_weight: float = 1.0,
    name_patterns: Sequence[str] = (),
    lora_only: bool = False,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Compute one SFT occurrence gradient with a bounded direct fallback."""

    if not 0.0 <= sample_weight <= 1_000.0:
        raise ValueError("sample_weight must be in [0, 1000]")
    parameters = select_trainable_parameters(
        state.model, name_patterns=name_patterns, lora_only=lora_only
    )
    state.model.eval()
    negative_log_likelihood = -_sequence_log_probability_tensor(
        state, prompt, completion
    )
    gradient = _flatten_autograd(negative_log_likelihood * sample_weight, parameters)
    return tuple(name for name, _ in parameters), gradient


def behavior_objective_gradient(
    state: Any,
    contract: Mapping[str, Any],
    *,
    name_patterns: Sequence[str] = (),
    lora_only: bool = False,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Gradient of a predeclared search objective, never the sealed holdout."""

    scorer = contract.get("scorer")
    search = contract.get("search")
    if not isinstance(scorer, Mapping) or not isinstance(search, Mapping):
        raise ValueError("behavior gradient requires scorer and search mappings")
    probes = search.get("prompts", search.get("probes"))
    if not isinstance(probes, list | tuple) or not probes:
        raise ValueError("behavior gradient requires inline search probes")
    parameters = select_trainable_parameters(
        state.model, name_patterns=name_patterns, lora_only=lora_only
    )
    objectives: list[torch.Tensor] = []
    for raw_probe in probes:
        probe: Mapping[str, Any] = (
            {"prompt": raw_probe} if isinstance(raw_probe, str) else raw_probe
        )
        prompt = str(probe["prompt"])
        scorer_type = scorer.get("type")
        if scorer_type == "sequence_logprob_margin":
            preferred = str(probe.get("preferred", scorer.get("preferred")))
            alternative = str(probe.get("alternative", scorer.get("alternative")))
            # A loss-like objective: minimizing it increases the declared margin.
            objective = -(
                _sequence_log_probability_tensor(state, prompt, preferred)
                - _sequence_log_probability_tensor(state, prompt, alternative)
            )
        elif scorer_type == "sequence_log_probability":
            target = str(probe.get("completion", scorer.get("completion")))
            objective = -_sequence_log_probability_tensor(state, prompt, target)
        elif scorer_type == "token_log_probability":
            target = str(probe.get("token", scorer.get("token")))
            objective = -_token_log_probability_tensor(state, prompt, target)
        elif scorer_type in {"sequence_nll", "scalar_loss"}:
            completion = str(probe.get("completion", scorer.get("completion")))
            objective = -_sequence_log_probability_tensor(state, prompt, completion)
        else:
            raise ValueError(
                f"gradient attribution does not support scorer {scorer_type!r}; "
                "use an explicit differentiable log-probability contract"
            )
        objectives.append(objective)
    mean_objective = torch.stack(objectives).mean()
    gradient = _flatten_autograd(mean_objective, parameters)
    return tuple(name for name, _ in parameters), gradient


def optimizer_second_moment_vector(
    state: Any, parameter_names: Sequence[str]
) -> torch.Tensor:
    named = dict(state.model.named_parameters())
    pieces: list[torch.Tensor] = []
    for name in parameter_names:
        parameter = named.get(name)
        if parameter is None:
            raise ValueError(f"unknown parameter {name!r}")
        optimizer_state = state.optimizer.state.get(parameter, {})
        moment = optimizer_state.get("exp_avg_sq")
        if moment is None:
            moment = torch.zeros_like(parameter)
        pieces.append(moment.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    return torch.cat(pieces)
