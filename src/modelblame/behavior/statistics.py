"""Deterministic statistical helpers for behavioral evidence."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A finite, closed confidence interval."""

    low: float
    high: float
    level: float
    method: str = "paired-bootstrap"

    def __post_init__(self) -> None:
        if not (0.0 < self.level < 1.0):
            raise ValueError("confidence level must be between zero and one")
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError("confidence interval bounds must be finite")
        if self.low > self.high:
            raise ValueError("confidence interval low bound exceeds high bound")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def paired_bootstrap_interval(
    original: Sequence[float],
    counterfactual: Sequence[float],
    *,
    confidence_level: float = 0.95,
    samples: int = 2_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Return a paired bootstrap interval for ``original - counterfactual``.

    Pairing is intentional: the same prompt occupies each position in both
    sequences.  Sampling is local and seeded, so this function never mutates
    process-global RNG state.
    """

    if len(original) != len(counterfactual):
        raise ValueError("paired bootstrap inputs must have equal length")
    if not original:
        raise ValueError("paired bootstrap requires at least one pair")
    if samples < 1 or samples > 10_000_000:
        raise ValueError("bootstrap samples must be in [1, 10_000_000]")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence level must be between zero and one")
    differences = [
        float(a) - float(b) for a, b in zip(original, counterfactual, strict=True)
    ]
    if not all(math.isfinite(item) for item in differences):
        raise ValueError("bootstrap inputs must be finite")
    rng = random.Random(seed)  # noqa: S311 - deterministic statistical resampling
    count = len(differences)
    estimates = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    estimates.sort()
    alpha = 1.0 - confidence_level
    return ConfidenceInterval(
        low=_quantile(estimates, alpha / 2.0),
        high=_quantile(estimates, 1.0 - alpha / 2.0),
        level=confidence_level,
    )


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    samples: int = 2_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap a mean without inventing pseudo-observations."""

    zeros = [0.0] * len(values)
    return paired_bootstrap_interval(
        values,
        zeros,
        confidence_level=confidence_level,
        samples=samples,
        seed=seed,
    )


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in input order."""

    values = [float(value) for value in p_values]
    if any(not (0.0 <= value <= 1.0) for value in values):
        raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted
