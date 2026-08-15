"""One-shot access control for sealed holdout probes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from modelblame.behavior.evaluate import BehaviorResult, _evaluate_unsealed, _mapping
from modelblame.behavior.scorers import SequenceScoringModel


@dataclass(frozen=True, slots=True)
class HoldoutUnsealRecord:
    candidate_hash: str
    contract_hash: str
    unsealed_at: str


@dataclass(frozen=True, slots=True)
class HoldoutPairResult:
    original: BehaviorResult
    counterfactual: BehaviorResult
    record: HoldoutUnsealRecord


class HoldoutLease:
    """Capability object that permits exactly one final holdout evaluation."""

    __slots__ = ("_consumed", "_contract", "_contract_hash", "_record")

    def __init__(
        self, contract: Mapping[str, Any] | Any, *, contract_hash: str
    ) -> None:
        if not contract_hash:
            raise ValueError("contract_hash is required")
        self._contract = _mapping(contract)
        holdout = self._contract.get("holdout")
        if not isinstance(holdout, Mapping) or holdout.get("sealed") is not True:
            raise ValueError("HoldoutLease requires a contract with sealed holdout")
        self._contract_hash = contract_hash
        self._consumed = False
        self._record: HoldoutUnsealRecord | None = None

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def record(self) -> HoldoutUnsealRecord | None:
        return self._record

    def final_evaluate(
        self,
        model: SequenceScoringModel,
        *,
        candidate_hash: str,
        checkpoint_hash: str | None = None,
    ) -> BehaviorResult:
        if self._consumed:
            raise PermissionError(
                "sealed holdout has already been unsealed for this contract"
            )
        if not candidate_hash:
            raise ValueError("candidate_hash is required before holdout unsealing")
        self._consumed = True
        self._record = HoldoutUnsealRecord(
            candidate_hash=candidate_hash,
            contract_hash=self._contract_hash,
            unsealed_at=datetime.now(UTC).isoformat(),
        )
        return _evaluate_unsealed(
            model,
            self._contract,
            split="holdout",
            checkpoint_hash=checkpoint_hash,
        )

    def final_evaluate_pair(
        self,
        original_model: SequenceScoringModel,
        counterfactual_model: SequenceScoringModel,
        *,
        candidate_hash: str,
        original_checkpoint_hash: str | None = None,
        counterfactual_checkpoint_hash: str | None = None,
    ) -> HoldoutPairResult:
        """Unseal once and perform the predeclared paired final comparison."""

        if self._consumed:
            raise PermissionError(
                "sealed holdout has already been unsealed for this contract"
            )
        if not candidate_hash:
            raise ValueError("candidate_hash is required before holdout unsealing")
        self._consumed = True
        record = HoldoutUnsealRecord(
            candidate_hash=candidate_hash,
            contract_hash=self._contract_hash,
            unsealed_at=datetime.now(UTC).isoformat(),
        )
        self._record = record
        original = _evaluate_unsealed(
            original_model,
            self._contract,
            split="holdout",
            checkpoint_hash=original_checkpoint_hash,
        )
        counterfactual = _evaluate_unsealed(
            counterfactual_model,
            self._contract,
            split="holdout",
            checkpoint_hash=counterfactual_checkpoint_hash,
        )
        return HoldoutPairResult(original, counterfactual, record)
