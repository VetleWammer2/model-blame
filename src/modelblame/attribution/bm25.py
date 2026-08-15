"""A deterministic BM25 lexical candidate baseline."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from modelblame.attribution.base import (
    AttributionIndex,
    CandidateEvent,
    CandidateScore,
    ranked_scores,
)

TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_PATTERN.findall(text))


def bm25_scores(
    documents: Sequence[str],
    query: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[float, ...]:
    if k1 <= 0.0 or not 0.0 <= b <= 1.0:
        raise ValueError("BM25 requires k1 > 0 and b in [0, 1]")
    tokenized = [tokenize(document) for document in documents]
    if not tokenized:
        return ()
    average_length = sum(len(document) for document in tokenized) / len(tokenized)
    average_length = max(average_length, 1.0)
    query_terms = tuple(dict.fromkeys(tokenize(query)))
    frequencies = Counter(
        term for term in query_terms for document in tokenized if term in set(document)
    )
    total = len(tokenized)
    scores: list[float] = []
    for document in tokenized:
        counts = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if frequency == 0:
                continue
            document_frequency = frequencies[term]
            inverse_document_frequency = math.log(
                1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / average_length
            )
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return tuple(scores)


class BM25Generator:
    method_id = "bm25"

    def build_index(
        self,
        events: Sequence[CandidateEvent],
        *,
        run_hash: str,
        behavior_contract_hash: str,
        checkpoint_hashes: Sequence[str],
        config: dict[str, Any],
    ) -> AttributionIndex:
        query = config.get("query")
        if isinstance(query, list):
            query = " ".join(str(item) for item in query)
        if not isinstance(query, str) or not query:
            raise ValueError("BM25 config requires a non-empty query")
        raw = bm25_scores(
            [event.text for event in events],
            query,
            k1=float(config.get("k1", 1.5)),
            b=float(config.get("b", 0.75)),
        )
        scores = ranked_scores(
            events, raw, method_id=self.method_id, method_config=config
        )
        return AttributionIndex(
            method_id=self.method_id,
            run_hash=run_hash,
            behavior_contract_hash=behavior_contract_hash,
            checkpoint_hashes=tuple(checkpoint_hashes),
            method_config=config,
            projection_seed=None,
            scores=scores,
        )

    def rank(self, index: AttributionIndex, *, limit: int) -> list[CandidateScore]:
        return list(index.scores[:limit])
