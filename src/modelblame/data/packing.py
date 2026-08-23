"""Deterministic greedy sequence packing with occurrence-level token spans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from modelblame.adapters.tiny_causal_lm import ByteTokenizer
from modelblame.checkpoint.cursor import TrainingCursor
from modelblame.data.indexed import IndexedDataset, IndexedExample


def _occurrence_id(
    *,
    run_id: str,
    global_step: int,
    microbatch_index: int,
    batch_position: int,
    token_start: int,
    token_end: int,
    example_id: str,
) -> str:
    try:
        from modelblame.data.identity import compute_occurrence_id

        return compute_occurrence_id(
            run_id=run_id,
            global_step=global_step,
            microbatch_index=microbatch_index,
            batch_position=batch_position,
            packed_token_span=(token_start, token_end),
            example_id=example_id,
        )
    except ImportError:
        value = [
            run_id,
            global_step,
            microbatch_index,
            batch_position,
            [token_start, token_end],
            example_id,
        ]
        digest = hashlib.sha256(
            json.dumps(value, separators=(",", ":")).encode()
        ).hexdigest()
        return f"occ_{digest}"


@dataclass(frozen=True, slots=True)
class OccurrenceSpan:
    occurrence_id: str
    example_id: str
    source: str
    source_row: int
    epoch: int
    global_step: int
    microbatch_index: int
    batch_position: int
    packed_sequence_id: str
    token_start: int
    token_end: int
    prompt_token_mask: tuple[bool, ...]
    completion_token_mask: tuple[bool, ...]
    original_loss_weight: float
    left_truncated_tokens: int = 0
    right_truncated_tokens: int = 0
    reserved_noop: bool = False
    injection_loss_weights: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prompt_token_mask"] = list(self.prompt_token_mask)
        value["completion_token_mask"] = list(self.completion_token_mask)
        value["injection_loss_weights"] = list(self.injection_loss_weights)
        return value


@dataclass(frozen=True, slots=True)
class PackedSequence:
    packed_sequence_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    loss_weights: tuple[float, ...]
    occurrences: tuple[OccurrenceSpan, ...]
    padding_tokens: int


@dataclass(frozen=True, slots=True)
class PackedMicrobatch:
    global_step: int
    microbatch_index: int
    sequences: tuple[PackedSequence, ...]
    cursor_after: Mapping[str, int]


class DeterministicPacker:
    """Greedy packer whose complete mutable state lives in ``TrainingCursor``."""

    schema_version = 1

    def __init__(
        self,
        dataset: IndexedDataset,
        tokenizer: ByteTokenizer,
        *,
        context_length: int,
        batch_size: int,
        reserved_noop_slots: int = 0,
        run_id: str,
        cursor: TrainingCursor,
    ) -> None:
        if context_length < 4 or batch_size <= 0 or not run_id:
            raise ValueError("invalid deterministic packer configuration")
        if not 0 <= reserved_noop_slots < batch_size:
            raise ValueError("reserved_noop_slots must be in [0, batch_size)")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.reserved_noop_slots = reserved_noop_slots
        self.run_id = run_id
        self.cursor = cursor
        self.cursor.validate()
        self.dataset = tuple(dataset)
        self.reserved_dataset = tuple(
            example for example in dataset if example.reserved_noop
        )
        if not any(not example.reserved_noop for example in self.dataset):
            raise ValueError("the active training dataset is empty")
        if bool(self.reserved_noop_slots) != bool(self.reserved_dataset):
            raise ValueError(
                "reserved no-op slots require reserved dataset examples and vice versa"
            )
        if self.cursor.source_row >= len(self.dataset):
            raise ValueError("cursor source row is outside dataset")
        self._skip_reserved_rows()

    def _skip_reserved_rows(self) -> None:
        while self.dataset[self.cursor.source_row].reserved_noop:
            self.cursor.source_row += 1
            if self.cursor.source_row == len(self.dataset):
                self.cursor.source_row = 0
                self.cursor.epoch += 1

    def _peek(self) -> tuple[IndexedExample, int]:
        return self.dataset[self.cursor.source_row], self.cursor.epoch

    def _advance(self) -> None:
        self.cursor.source_row += 1
        self.cursor.sampler_offset += 1
        if self.cursor.source_row == len(self.dataset):
            self.cursor.source_row = 0
            self.cursor.epoch += 1
        self._skip_reserved_rows()

    def _encoded(
        self, example: IndexedExample
    ) -> tuple[list[int], list[float], list[bool], list[bool], int]:
        tokens, weights = self.tokenizer.encode_sft(example.prompt, example.completion)
        prompt_length = len(self.tokenizer.encode(example.prompt))
        prompt_mask = [False, *([True] * prompt_length)] + [False] * (
            len(tokens) - prompt_length - 1
        )
        completion_mask = [weight > 0 for weight in weights]
        left_truncated = max(0, len(tokens) - self.context_length)
        if left_truncated:
            # Preserve BOS plus the right edge, which contains the supervised
            # completion.  The operation and count are recorded in the ledger.
            tokens = [tokens[0], *tokens[-(self.context_length - 1) :]]
            weights = [weights[0], *weights[-(self.context_length - 1) :]]
            prompt_mask = [prompt_mask[0], *prompt_mask[-(self.context_length - 1) :]]
            completion_mask = [
                completion_mask[0],
                *completion_mask[-(self.context_length - 1) :],
            ]
        return (
            tokens,
            [weight * example.sample_weight for weight in weights],
            prompt_mask,
            completion_mask,
            left_truncated,
        )

    def next_microbatch(
        self, *, global_step: int, microbatch_index: int
    ) -> PackedMicrobatch:
        sequences: list[PackedSequence] = []
        active_batch_size = self.batch_size - self.reserved_noop_slots
        microbatch_ordinal = self.cursor.packed_sequence_count // self.batch_size
        for batch_position in range(active_batch_size):
            token_ids: list[int] = []
            loss_weights: list[float] = []
            pending: list[
                tuple[
                    IndexedExample,
                    int,
                    int,
                    int,
                    list[bool],
                    list[bool],
                    int,
                ]
            ] = []
            while len(token_ids) < self.context_length:
                example, epoch = self._peek()
                (
                    encoded,
                    weights,
                    prompt_mask,
                    completion_mask,
                    left_truncated,
                ) = self._encoded(example)
                if token_ids and len(token_ids) + len(encoded) > self.context_length:
                    break
                start = len(token_ids)
                token_ids.extend(encoded)
                loss_weights.extend(weights)
                end = len(token_ids)
                pending.append(
                    (
                        example,
                        epoch,
                        start,
                        end,
                        prompt_mask,
                        completion_mask,
                        left_truncated,
                    )
                )
                self._advance()
                if len(encoded) == self.context_length:
                    break
            unpadded_length = len(token_ids)
            padding = self.context_length - unpadded_length
            token_ids.extend([self.tokenizer.pad_token_id] * padding)
            loss_weights.extend([0.0] * padding)
            seed_value = {
                "run_id": self.run_id,
                "step": global_step,
                "microbatch": microbatch_index,
                "batch_position": batch_position,
                "packed_sequence_count": self.cursor.packed_sequence_count,
                "examples": [entry[0].example_id for entry in pending],
            }
            packed_id = (
                "pack_"
                + hashlib.sha256(
                    json.dumps(
                        seed_value, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
            )
            spans: list[OccurrenceSpan] = []
            for (
                example,
                epoch,
                start,
                end,
                prompt_mask,
                completion_mask,
                left_truncated,
            ) in pending:
                occurrence_id = _occurrence_id(
                    run_id=self.run_id,
                    global_step=global_step,
                    microbatch_index=microbatch_index,
                    batch_position=batch_position,
                    token_start=start,
                    token_end=end,
                    example_id=example.example_id,
                )
                spans.append(
                    OccurrenceSpan(
                        occurrence_id=occurrence_id,
                        example_id=example.example_id,
                        source=example.source,
                        source_row=example.source_row,
                        epoch=epoch,
                        global_step=global_step,
                        microbatch_index=microbatch_index,
                        batch_position=batch_position,
                        packed_sequence_id=packed_id,
                        token_start=start,
                        token_end=end,
                        prompt_token_mask=tuple(prompt_mask),
                        completion_token_mask=tuple(completion_mask),
                        original_loss_weight=example.sample_weight,
                        left_truncated_tokens=left_truncated,
                    )
                )
            sequences.append(
                PackedSequence(
                    packed_sequence_id=packed_id,
                    input_ids=tuple(token_ids),
                    attention_mask=tuple(
                        index < unpadded_length for index in range(self.context_length)
                    ),
                    loss_weights=tuple(loss_weights),
                    occurrences=tuple(spans),
                    padding_tokens=padding,
                )
            )
            self.cursor.packed_sequence_count += 1
        for batch_position in range(active_batch_size, self.batch_size):
            reserved_occurrence_index = (
                microbatch_ordinal * self.reserved_noop_slots
                + batch_position
                - active_batch_size
            )
            reserved_index = reserved_occurrence_index % len(self.reserved_dataset)
            example = self.reserved_dataset[reserved_index]
            (
                token_ids,
                injection_weights,
                prompt_mask,
                completion_mask,
                left_truncated,
            ) = self._encoded(example)
            unpadded_length = len(token_ids)
            padding = self.context_length - unpadded_length
            token_ids.extend([self.tokenizer.pad_token_id] * padding)
            seed_value = {
                "run_id": self.run_id,
                "step": global_step,
                "microbatch": microbatch_index,
                "batch_position": batch_position,
                "packed_sequence_count": self.cursor.packed_sequence_count,
                "reserved_noop": True,
                "example": example.example_id,
            }
            packed_id = (
                "noop_"
                + hashlib.sha256(
                    json.dumps(
                        seed_value, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
            )
            sequences.append(
                PackedSequence(
                    packed_sequence_id=packed_id,
                    input_ids=tuple(token_ids),
                    attention_mask=tuple(
                        index < unpadded_length for index in range(self.context_length)
                    ),
                    loss_weights=(0.0,) * self.context_length,
                    occurrences=(
                        OccurrenceSpan(
                            occurrence_id=_occurrence_id(
                                run_id=self.run_id,
                                global_step=global_step,
                                microbatch_index=microbatch_index,
                                batch_position=batch_position,
                                token_start=0,
                                token_end=unpadded_length,
                                example_id=example.example_id,
                            ),
                            example_id=example.example_id,
                            source=example.source,
                            source_row=example.source_row,
                            epoch=reserved_occurrence_index
                            // len(self.reserved_dataset),
                            global_step=global_step,
                            microbatch_index=microbatch_index,
                            batch_position=batch_position,
                            packed_sequence_id=packed_id,
                            token_start=0,
                            token_end=unpadded_length,
                            prompt_token_mask=tuple(prompt_mask),
                            completion_token_mask=tuple(completion_mask),
                            original_loss_weight=example.sample_weight,
                            left_truncated_tokens=left_truncated,
                            reserved_noop=True,
                            injection_loss_weights=tuple(injection_weights),
                        ),
                    ),
                    padding_tokens=padding,
                )
            )
            self.cursor.packed_sequence_count += 1
        self.cursor.microbatch = microbatch_index + 1
        self.cursor.gradient_accumulation_position = microbatch_index + 1
        return PackedMicrobatch(
            global_step=global_step,
            microbatch_index=microbatch_index,
            sequences=tuple(sequences),
            cursor_after=self.cursor.to_dict(),
        )
