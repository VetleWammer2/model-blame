"""TracIn-CP-style checkpoint gradient similarity."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from modelblame.attribution.projection import project_vector


def tracin_cp_score(
    training_gradients: Mapping[str, torch.Tensor],
    behavior_gradients: Mapping[str, torch.Tensor],
    *,
    learning_rates: Mapping[str, float] | None = None,
    projection_dimension: int | None = None,
    projection_seed: int = 0,
) -> float:
    """Sum checkpoint gradient dot products weighted by recorded learning rate.

    This is deliberately labelled TracIn-CP-*style*: it provides the checkpoint
    gradient-similarity calculation but does not claim library equivalence.
    """

    checkpoints = sorted(set(training_gradients) & set(behavior_gradients))
    if not checkpoints:
        raise ValueError("no common checkpoints for TracIn-CP-style scoring")
    score = 0.0
    for offset, checkpoint in enumerate(checkpoints):
        training = (
            training_gradients[checkpoint].detach().reshape(-1).to(torch.float64).cpu()
        )
        behavior = (
            behavior_gradients[checkpoint].detach().reshape(-1).to(torch.float64).cpu()
        )
        if training.shape != behavior.shape:
            raise ValueError(f"gradient shape mismatch at checkpoint {checkpoint}")
        if projection_dimension is not None and projection_dimension < training.numel():
            seed = projection_seed + offset
            training = project_vector(
                training, dimension=projection_dimension, seed=seed
            )
            behavior = project_vector(
                behavior, dimension=projection_dimension, seed=seed
            )
        rate = float(learning_rates.get(checkpoint, 1.0)) if learning_rates else 1.0
        score += rate * float(torch.dot(training, behavior).item())
    return score
