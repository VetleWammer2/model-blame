"""Offline hashed-token embedding similarity baseline."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from itertools import pairwise

from modelblame.attribution.bm25 import tokenize


def _hashed_embedding(text: str, dimension: int) -> list[float]:
    if not 8 <= dimension <= 65_536:
        raise ValueError("embedding dimension must be in [8, 65536]")
    vector = [0.0] * dimension
    tokens = tokenize(text)
    features = list(tokens)
    features.extend(f"{left}\x1f{right}" for left, right in pairwise(tokens))
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def embedding_similarity_scores(
    documents: Sequence[str], query: str, *, dimension: int = 256
) -> tuple[float, ...]:
    query_vector = _hashed_embedding(query, dimension)
    return tuple(
        sum(
            a * b
            for a, b in zip(
                _hashed_embedding(document, dimension), query_vector, strict=True
            )
        )
        for document in documents
    )
