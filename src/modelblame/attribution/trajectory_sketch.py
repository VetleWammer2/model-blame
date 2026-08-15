"""Experimental optimizer-aware trajectory gradient sketches."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from modelblame.attribution.projection import project_vector


def adam_diagonal_precondition(
    gradient: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    beta2: float,
    step: int,
    epsilon: float,
) -> torch.Tensor:
    """Approximate Adam's diagonal update geometry without momentum direction."""

    if gradient.shape != exp_avg_sq.shape:
        raise ValueError("gradient and second moment shapes differ")
    if not 0.0 <= beta2 < 1.0 or step < 1 or epsilon <= 0.0:
        raise ValueError("invalid Adam preconditioner parameters")
    corrected = exp_avg_sq.to(torch.float64) / (1.0 - beta2**step)
    return gradient.to(torch.float64) / (torch.sqrt(corrected) + epsilon)


def trajectory_sketch_score(
    training_gradients: Mapping[str, torch.Tensor],
    behavior_gradients: Mapping[str, torch.Tensor],
    second_moments: Mapping[str, torch.Tensor],
    *,
    steps: Mapping[str, int],
    projection_dimension: int,
    projection_seed: int,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> float:
    """Evaluate the documented ``sum <R P_c g_i, R g_B>`` approximation."""

    checkpoints = sorted(
        set(training_gradients) & set(behavior_gradients) & set(second_moments)
    )
    if not checkpoints:
        raise ValueError("no common checkpoints for trajectory sketch")
    total = 0.0
    for offset, checkpoint in enumerate(checkpoints):
        train = training_gradients[checkpoint].detach().reshape(-1).cpu()
        behavior = behavior_gradients[checkpoint].detach().reshape(-1).cpu()
        moment = second_moments[checkpoint].detach().reshape(-1).cpu()
        if train.shape != behavior.shape or train.shape != moment.shape:
            raise ValueError(f"trajectory tensor shape mismatch at {checkpoint}")
        preconditioned = adam_diagonal_precondition(
            train,
            moment,
            beta2=beta2,
            step=int(steps[checkpoint]),
            epsilon=epsilon,
        )
        dimension = min(projection_dimension, train.numel())
        seed = projection_seed + offset
        total += float(
            torch.dot(
                project_vector(preconditioned, dimension=dimension, seed=seed),
                project_vector(behavior, dimension=dimension, seed=seed),
            ).item()
        )
    return total
