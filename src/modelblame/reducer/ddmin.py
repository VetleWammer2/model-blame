"""Deterministic delta debugging for approximately monotone patches."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def _partition(items: tuple[str, ...], count: int) -> list[tuple[str, ...]]:
    count = max(1, min(count, len(items)))
    quotient, remainder = divmod(len(items), count)
    chunks: list[tuple[str, ...]] = []
    cursor = 0
    for index in range(count):
        size = quotient + int(index < remainder)
        chunks.append(items[cursor : cursor + size])
        cursor += size
    return [chunk for chunk in chunks if chunk]


def ddmin(
    items: Iterable[str],
    accepted: Callable[[frozenset[str]], bool],
) -> frozenset[str]:
    """Return a one-minimal accepted subset under standard ddmin assumptions."""

    current = tuple(sorted(set(items)))
    if not current or not accepted(frozenset(current)):
        raise ValueError("ddmin requires a non-empty accepted starting set")
    granularity = 2
    while len(current) >= 2:
        chunks = _partition(current, granularity)
        reduced = False
        for chunk in chunks:
            complement = frozenset(item for item in current if item not in set(chunk))
            if complement and accepted(complement):
                current = tuple(sorted(complement))
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        for chunk in chunks:
            candidate = frozenset(chunk)
            if accepted(candidate):
                current = tuple(sorted(candidate))
                granularity = 2
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return frozenset(current)
