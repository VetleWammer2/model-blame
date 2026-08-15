"""Derive visible negative outcomes from measured benchmark results."""

from __future__ import annotations

from typing import Any


def collect_negative_results(
    *,
    cases: dict[str, dict[str, Any]],
    interaction: dict[str, Any],
    attribution: dict[str, Any],
    exhaustive: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return deterministic negative observations without reclassifying successes."""
    results: list[dict[str, Any]] = []
    for name, case in sorted(cases.items()):
        replay = case["all_causal_patch"]
        if not replay.get("accepted", False):
            results.append(
                {
                    "kind": "SCENARIO_PATCH_FAILED",
                    "scenario": name,
                    "status": replay.get("status", "INCONCLUSIVE"),
                }
            )

    for method, measured in sorted(attribution["methods"].items()):
        recall = float(measured["recall_at_k"])
        if recall < 1.0:
            results.append(
                {
                    "kind": "ATTRIBUTION_MISSED_CAUSAL_OCCURRENCES",
                    "method": method,
                    "k": int(measured["k"]),
                    "recall_at_k": recall,
                }
            )

    for family, measured in sorted(interaction["single_families"].items()):
        if not measured["accepted"]:
            results.append(
                {
                    "kind": "INTERACTION_SINGLETON_FAILED",
                    "family": family,
                    "status": measured["status"],
                    "target_effect": float(measured["target_effect"]),
                }
            )

    failed_controls = sum(
        not point["controls_passed"]
        for point in exhaustive["exhaustive"]["points"]
        if point["groups"]
    )
    if failed_controls:
        results.append(
            {
                "kind": "EXHAUSTIVE_SUBSETS_FAILED_CONTROLS",
                "count": failed_controls,
                "executed_nonempty_subsets": int(
                    exhaustive["executed_nonempty_subsets"]
                ),
            }
        )
    return results
