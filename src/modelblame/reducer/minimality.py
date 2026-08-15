"""Minimality grades and independent one-minimality checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class MinimalityGrade(StrEnum):
    GLOBAL_MINIMUM = "GLOBAL_MINIMUM"
    ONE_MINIMAL = "ONE_MINIMAL"
    BUDGET_MINIMAL = "BUDGET_MINIMAL"
    UNREDUCED = "UNREDUCED"
    NO_ACCEPTED_PATCH = "NO_ACCEPTED_PATCH"


@dataclass(frozen=True, slots=True)
class MinimalityTest:
    removed: str
    subset: tuple[str, ...]
    accepted: bool


def test_one_minimality(
    subset: frozenset[str],
    accepted: Callable[[frozenset[str]], bool],
) -> tuple[bool, tuple[MinimalityTest, ...]]:
    tests: list[MinimalityTest] = []
    for item in sorted(subset):
        reduced = subset - {item}
        outcome = accepted(reduced) if reduced else False
        tests.append(
            MinimalityTest(
                removed=item, subset=tuple(sorted(reduced)), accepted=outcome
            )
        )
    return all(not test.accepted for test in tests), tuple(tests)
