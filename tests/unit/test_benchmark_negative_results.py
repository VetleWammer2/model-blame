from benchmarks.causal_origin.negative_results import collect_negative_results


def test_negative_results_preserve_failures() -> None:
    results = collect_negative_results(
        cases={"passing": {"all_causal_patch": {"accepted": True}}},
        attribution={
            "methods": {
                "perfect": {"k": 2, "recall_at_k": 1.0},
                "miss": {"k": 2, "recall_at_k": 0.5},
            }
        },
        interaction={
            "single_families": {
                "left": {
                    "accepted": False,
                    "status": "TARGET_FAILED",
                    "target_effect": 0.25,
                }
            }
        },
        exhaustive={
            "executed_nonempty_subsets": 2,
            "exhaustive": {
                "points": [
                    {"groups": [], "controls_passed": True},
                    {"groups": ["a"], "controls_passed": False},
                    {"groups": ["b"], "controls_passed": True},
                ]
            },
        },
    )

    assert [result["kind"] for result in results] == [
        "ATTRIBUTION_MISSED_CAUSAL_OCCURRENCES",
        "INTERACTION_SINGLETON_FAILED",
        "EXHAUSTIVE_SUBSETS_FAILED_CONTROLS",
    ]
