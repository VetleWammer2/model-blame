"""Transparent ranking metrics against exhaustive counterfactual ground truth."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def precision_recall_at_k(
    ranking: Sequence[str], causal_groups: set[str], *, k: int
) -> tuple[float, float]:
    if k < 1:
        raise ValueError("k must be positive")
    selected = set(ranking[:k])
    true_positives = len(selected & causal_groups)
    precision = true_positives / min(k, len(ranking)) if ranking else 0.0
    recall = true_positives / len(causal_groups) if causal_groups else 0.0
    return precision, recall


def _average_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    output: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for key, _ in ordered[index:end]:
            output[key] = rank
        index = end
    return output


def spearman_rank_correlation(
    predicted: Mapping[str, float], exact_effects: Mapping[str, float]
) -> float:
    keys = sorted(set(predicted) & set(exact_effects))
    if len(keys) < 2:
        raise ValueError("rank correlation requires at least two common items")
    left = _average_ranks({key: predicted[key] for key in keys})
    right = _average_ranks({key: exact_effects[key] for key in keys})
    left_mean = sum(left.values()) / len(keys)
    right_mean = sum(right.values()) / len(keys)
    covariance = sum(
        (left[key] - left_mean) * (right[key] - right_mean) for key in keys
    )
    left_scale = math.sqrt(sum((left[key] - left_mean) ** 2 for key in keys))
    right_scale = math.sqrt(sum((right[key] - right_mean) ** 2 for key in keys))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return covariance / (left_scale * right_scale)
