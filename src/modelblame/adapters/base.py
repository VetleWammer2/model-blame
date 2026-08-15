"""Trusted adapter protocol and shared training value objects.

Adapters are executable local Python code and are therefore inside ModelBlame's
trust boundary.  Recorded events passed to adapters are untrusted data and must
be validated before tensors are allocated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class StepIntervention:
    """Loss-level changes to apply without changing the recorded batch shape."""

    ablate_occurrence_ids: frozenset[str] = frozenset()
    occurrence_weights: Mapping[str, float] = field(default_factory=dict)
    normalization: str = "FIXED_DENOMINATOR"

    def __post_init__(self) -> None:
        if self.normalization not in {"FIXED_DENOMINATOR", "RENORMALIZED"}:
            raise ValueError(f"unsupported normalization: {self.normalization}")
        for occurrence_id, weight in self.occurrence_weights.items():
            if not occurrence_id or not isinstance(occurrence_id, str):
                raise ValueError("occurrence weight keys must be non-empty strings")
            if (
                not isinstance(weight, int | float)
                or not 0.0 <= float(weight) <= 1_000.0
            ):
                raise ValueError(
                    "occurrence weights must be finite values in [0, 1000]"
                )


@dataclass(slots=True)
class TrainingBatch:
    """A tensor batch reconstructed from an immutable recorded event."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_weights: torch.Tensor
    occurrence_spans: tuple[tuple[Mapping[str, Any], ...], ...]
    original_loss_denominator: float
    global_step: int
    microbatch_index: int


@dataclass(frozen=True, slots=True)
class PerOccurrenceLoss:
    losses: Mapping[str, float]
    token_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class StepResult:
    loss: float
    denominator: float
    optimizer_stepped: bool
    output_hash: str
    per_occurrence: PerOccurrenceLoss


@runtime_checkable
class ExperimentAdapter(Protocol):
    """Narrow API supported by the replay engine.

    Implementations must reconstruct a batch from ``RecordedBatchEvent`` and
    must not consult mutable source data during replay.
    """

    adapter_id: str

    def build_experiment(self, config: Any, *, device: torch.device) -> Any: ...

    def build_dataset(self, config: Any) -> Any: ...

    def build_batch(self, state: Any, event: Any) -> TrainingBatch: ...

    def compute_per_occurrence_loss(
        self, state: Any, batch: TrainingBatch
    ) -> PerOccurrenceLoss: ...

    def apply_training_step(
        self,
        state: Any,
        batches: Sequence[TrainingBatch],
        intervention: StepIntervention | None,
    ) -> Sequence[StepResult]: ...

    def save_state(self, state: Any, destination: Path) -> Any: ...

    def load_state(self, checkpoint: Path, *, device: torch.device) -> Any: ...

    def evaluate_behavior(self, state: Any, contract: Any, split: Any) -> Any: ...
