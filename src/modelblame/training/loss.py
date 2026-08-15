"""Completion-only causal-LM loss and occurrence-level interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional

if TYPE_CHECKING:
    from modelblame.adapters.base import StepIntervention, TrainingBatch


@dataclass(frozen=True, slots=True)
class MaskedLossResult:
    loss: torch.Tensor
    numerator: torch.Tensor
    denominator: float
    token_losses: torch.Tensor


def masked_causal_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    original_denominator: float | None = None,
    normalization: str = "FIXED_DENOMINATOR",
) -> MaskedLossResult:
    """Compute shifted causal loss with explicit denominator semantics."""

    if logits.ndim != 3 or input_ids.ndim != 2 or loss_weights.ndim != 2:
        raise ValueError("expected logits [B,S,V] and inputs/weights [B,S]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != loss_weights.shape:
        raise ValueError("logits, input IDs, and loss weights have incompatible shapes")
    if not torch.isfinite(loss_weights).all() or (loss_weights < 0).any():
        raise ValueError("loss weights must be finite and non-negative")
    token_losses = functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(input_ids.shape[0], input_ids.shape[1] - 1)
    shifted_weights = loss_weights[:, 1:].to(token_losses.dtype)
    numerator = (token_losses * shifted_weights).sum()
    if normalization == "FIXED_DENOMINATOR":
        if original_denominator is None:
            original_denominator = float(shifted_weights.sum().detach())
        denominator = float(original_denominator)
    elif normalization == "RENORMALIZED":
        denominator = float(shifted_weights.sum().detach())
    else:
        raise ValueError(f"unknown normalization: {normalization}")
    if not 0.0 <= denominator < 1e15:
        raise ValueError("loss denominator is invalid")
    # An entirely ablated microbatch contributes a differentiable zero.
    loss = numerator / denominator if denominator > 0 else numerator * 0.0
    return MaskedLossResult(
        loss=loss,
        numerator=numerator,
        denominator=denominator,
        token_losses=token_losses,
    )


def apply_occurrence_intervention(
    batch: TrainingBatch, intervention: StepIntervention
) -> torch.Tensor:
    """Return modified token weights while preserving every tensor dimension."""

    weights = batch.loss_weights.clone()
    known: set[str] = set()
    for batch_position, spans in enumerate(batch.occurrence_spans):
        for span in spans:
            occurrence_id = str(span["occurrence_id"])
            known.add(occurrence_id)
            start = int(span["token_start"])
            end = int(span["token_end"])
            if not 0 <= start < end <= weights.shape[1]:
                raise ValueError(f"invalid token span for {occurrence_id}")
            if occurrence_id in intervention.ablate_occurrence_ids:
                weights[batch_position, start:end] = 0.0
            elif occurrence_id in intervention.occurrence_weights:
                weights[batch_position, start:end] *= float(
                    intervention.occurrence_weights[occurrence_id]
                )
    requested = set(intervention.ablate_occurrence_ids) | set(
        intervention.occurrence_weights
    )
    # It is valid for an intervention to target another step; only reject IDs
    # that are present in both maps, which would be ambiguous.
    conflicting = set(intervention.ablate_occurrence_ids) & set(
        intervention.occurrence_weights
    )
    if conflicting:
        raise ValueError(f"conflicting intervention for {sorted(conflicting)[0]}")
    del known, requested
    return weights


def per_occurrence_losses(
    logits: torch.Tensor, batch: TrainingBatch
) -> tuple[dict[str, float], dict[str, int]]:
    """Report original weighted loss numerator for every packed occurrence."""

    result = masked_causal_loss(
        logits,
        batch.input_ids,
        batch.loss_weights,
        original_denominator=batch.original_loss_denominator,
    )
    shifted_loss = result.token_losses
    shifted_weights = batch.loss_weights[:, 1:].to(shifted_loss.dtype)
    values: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch_position, spans in enumerate(batch.occurrence_spans):
        for span in spans:
            occurrence_id = str(span["occurrence_id"])
            # Token positions [start,end) map to shifted indices [start-1,end-1).
            start = max(1, int(span["token_start"])) - 1
            end = max(1, int(span["token_end"])) - 1
            occurrence_weights = shifted_weights[batch_position, start:end]
            occurrence_losses = shifted_loss[batch_position, start:end]
            values[occurrence_id] = float(
                (occurrence_losses * occurrence_weights).sum().detach()
            )
            counts[occurrence_id] = int((occurrence_weights > 0).sum().detach())
    return values, counts
