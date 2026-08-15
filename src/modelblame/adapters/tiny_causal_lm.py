"""Deterministic byte-tokenized decoder-only language model adapter."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from modelblame.adapters.base import (
    PerOccurrenceLoss,
    StepIntervention,
    StepResult,
    TrainingBatch,
)
from modelblame.training.lora import LoRAConfig, inject_lora
from modelblame.training.loss import (
    apply_occurrence_intervention,
    masked_causal_loss,
    per_occurrence_losses,
)


class ByteTokenizer:
    """Versioned UTF-8 byte tokenizer with four fixed special tokens."""

    schema_version = 1
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    separator_token_id = 3
    byte_offset = 4
    vocab_size = 260

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoded = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_special_tokens:
            return [self.bos_token_id, *encoded, self.eos_token_id]
        return encoded

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = True
    ) -> str:
        output = bytearray()
        for token_id in token_ids:
            value = int(token_id)
            if value < self.byte_offset:
                if skip_special_tokens:
                    continue
                raise ValueError("special tokens cannot be represented as text bytes")
            if value >= self.vocab_size:
                raise ValueError(f"token ID is outside vocabulary: {value}")
            output.append(value - self.byte_offset)
        return output.decode("utf-8", errors="replace")

    def encode_sft(self, prompt: str, completion: str) -> tuple[list[int], list[float]]:
        prompt_tokens = self.encode(prompt)
        completion_tokens = self.encode(completion)
        tokens = [
            self.bos_token_id,
            *prompt_tokens,
            self.separator_token_id,
            *completion_tokens,
            self.eos_token_id,
        ]
        # A weight at position j supervises token j from logits at j - 1.
        weights = [0.0] * (2 + len(prompt_tokens)) + [1.0] * (
            len(completion_tokens) + 1
        )
        return tokens, weights

    @property
    def fingerprint(self) -> str:
        payload = {
            "type": "utf8-byte",
            "schema_version": self.schema_version,
            "special_tokens": {
                "pad": self.pad_token_id,
                "bos": self.bos_token_id,
                "eos": self.eos_token_id,
                "separator": self.separator_token_id,
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class TinyCausalLMConfig:
    vocab_size: int = ByteTokenizer.vocab_size
    context_length: int = 128
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 128
    dropout: float = 0.0
    bias: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        if self.vocab_size < ByteTokenizer.vocab_size:
            raise ValueError("vocab_size is too small for the built-in tokenizer")
        if self.context_length < 4:
            raise ValueError("context_length must be at least 4")
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if self.num_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("layer counts and intermediate size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TinyCausalLMConfig:
        fields = dict(value)
        lora_value = fields.pop("lora", {}) or {}
        if "max_seq_len" in fields and "context_length" not in fields:
            fields["context_length"] = fields.pop("max_seq_len")
        if "n_layers" in fields and "num_layers" not in fields:
            fields["num_layers"] = fields.pop("n_layers")
        if "n_heads" in fields and "num_heads" not in fields:
            fields["num_heads"] = fields.pop("n_heads")
        if "d_model" in fields and "hidden_size" not in fields:
            fields["hidden_size"] = fields.pop("d_model")
        if "d_ff" in fields and "intermediate_size" not in fields:
            fields["intermediate_size"] = fields.pop("d_ff")
        allowed = {item.name for item in dataclass_fields(cls)}
        fields = {
            key: val for key, val in fields.items() if key in allowed and key != "lora"
        }
        target_modules = lora_value.get("target_modules", ("q_proj", "v_proj"))
        lora = LoRAConfig(
            rank=int(lora_value.get("rank", 0)),
            alpha=float(lora_value.get("alpha", 16.0)),
            dropout=float(lora_value.get("dropout", 0.0)),
            target_modules=tuple(target_modules),
        )
        return cls(**fields, lora=lora)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TinyLlamaConfig:
    """Configuration for the offline Llama-style decoder.

    This is deliberately separate from :class:`TinyCausalLMConfig`: selecting
    ``llama_style`` changes the architecture recorded in checkpoints instead
    of quietly changing the implementation behind the original tiny model.
    """

    architecture: str = "llama_style"
    vocab_size: int = ByteTokenizer.vocab_size
    context_length: int = 128
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 128
    dropout: float = 0.0
    bias: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        if self.architecture != "llama_style":
            raise ValueError("TinyLlamaConfig architecture must be llama_style")
        if self.vocab_size < ByteTokenizer.vocab_size:
            raise ValueError("vocab_size is too small for the built-in tokenizer")
        if self.context_length < 4:
            raise ValueError("context_length must be at least 4")
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 2:
            raise ValueError("Llama-style attention requires an even head dimension")
        if self.num_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("layer counts and intermediate size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rms_norm_eps <= 0.0 or self.rope_theta <= 0.0:
            raise ValueError("RMSNorm epsilon and RoPE theta must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TinyLlamaConfig:
        fields = dict(value)
        lora_value = fields.pop("lora", {}) or {}
        if "max_seq_len" in fields and "context_length" not in fields:
            fields["context_length"] = fields.pop("max_seq_len")
        if "n_layers" in fields and "num_layers" not in fields:
            fields["num_layers"] = fields.pop("n_layers")
        if "n_heads" in fields and "num_heads" not in fields:
            fields["num_heads"] = fields.pop("n_heads")
        if "d_model" in fields and "hidden_size" not in fields:
            fields["hidden_size"] = fields.pop("d_model")
        if "d_ff" in fields and "intermediate_size" not in fields:
            fields["intermediate_size"] = fields.pop("d_ff")
        allowed = {item.name for item in dataclass_fields(cls)}
        fields = {
            key: val for key, val in fields.items() if key in allowed and key != "lora"
        }
        target_modules = lora_value.get("target_modules", ("q_proj", "v_proj"))
        lora = LoRAConfig(
            rank=int(lora_value.get("rank", 0)),
            alpha=float(lora_value.get("alpha", 16.0)),
            dropout=float(lora_value.get("dropout", 0.0)),
            target_modules=tuple(target_modules),
        )
        return cls(**fields, lora=lora)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.q_proj(hidden))
        key = split_heads(self.k_proj(hidden))
        value = split_heads(self.v_proj(hidden))
        scores = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden.device,
        ).tril()
        allowed = causal.view(1, 1, sequence_length, sequence_length)
        if attention_mask is not None:
            allowed = allowed & attention_mask.to(torch.bool).view(
                batch_size, 1, 1, sequence_length
            )
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = self.dropout(functional.softmax(scores, dim=-1))
        attended = probabilities @ value
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, hidden_size)
        )
        if attention_mask is not None:
            attended = attended * attention_mask.unsqueeze(-1).to(attended.dtype)
        return self.o_proj(attended)


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        hidden = hidden + self.dropout(
            self.attention(self.attention_norm(hidden), attention_mask)
        )
        hidden = hidden + self.dropout(
            self.down_proj(functional.gelu(self.up_proj(self.mlp_norm(hidden))))
        )
        return hidden


class TinyCausalLM(nn.Module):
    """A compact causal Transformer suitable for CPU replay tests."""

    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.context_length, config.hidden_size
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lora_modules = inject_lora(self, config.lora) if config.lora.rank else ()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("input sequence exceeds model context length")
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.lm_head(self.final_norm(hidden))


class LlamaRMSNorm(nn.Module):
    """Root-mean-square normalization used by Llama-family decoders."""

    def __init__(self, hidden_size: int, *, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        normalized = hidden.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype=input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """Deterministic rotary-position frequencies for one attention layer."""

    def __init__(self, head_dim: int, *, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # The buffer is derived entirely from model configuration. Excluding it
        # from state storage avoids treating a recomputable constant as learned
        # state while preserving device movement through ``Module.to``.
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        positions = torch.arange(
            sequence_length, device=query.device, dtype=torch.float32
        )
        angles = torch.outer(positions, self.inv_freq.float())
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        cosine = (
            angles.cos()
            .to(dtype=query.dtype)
            .view(1, 1, sequence_length, query.shape[-1])
        )
        sine = (
            angles.sin()
            .to(dtype=query.dtype)
            .view(1, 1, sequence_length, query.shape[-1])
        )

        def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
            paired = tensor.unflatten(-1, (-1, 2))
            rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
            return rotated.flatten(-2)

        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )


class LlamaSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary positions."""

    def __init__(self, config: TinyLlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )
        self.rotary = LlamaRotaryEmbedding(self.head_dim, theta=config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.q_proj(hidden))
        key = split_heads(self.k_proj(hidden))
        value = split_heads(self.v_proj(hidden))
        query, key = self.rotary(query, key)
        scores = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden.device,
        ).tril()
        allowed = causal.view(1, 1, sequence_length, sequence_length)
        if attention_mask is not None:
            allowed = allowed & attention_mask.to(torch.bool).view(
                batch_size, 1, 1, sequence_length
            )
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = self.dropout(functional.softmax(scores, dim=-1))
        attended = probabilities @ value
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, hidden_size)
        )
        if attention_mask is not None:
            attended = attended * attention_mask.unsqueeze(-1).to(attended.dtype)
        return self.o_proj(attended)


class LlamaSwiGLU(nn.Module):
    """The gated SiLU feed-forward block used by Llama decoders."""

    def __init__(self, config: TinyLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gated = functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(self.dropout(gated))


class LlamaDecoderBlock(nn.Module):
    def __init__(self, config: TinyLlamaConfig) -> None:
        super().__init__()
        self.attention_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = LlamaSelfAttention(config)
        self.mlp_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaSwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        hidden = hidden + self.residual_dropout(
            self.attention(self.attention_norm(hidden), attention_mask)
        )
        return hidden + self.residual_dropout(self.mlp(self.mlp_norm(hidden)))


class TinyLlamaCausalLM(nn.Module):
    """A locally initialized Llama-style decoder for offline CPU experiments."""

    def __init__(self, config: TinyLlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            LlamaDecoderBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lora_modules = inject_lora(self, config.lora) if config.lora.rank else ()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("input sequence exceeds model context length")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.lm_head(self.final_norm(hidden))


class TinyCausalLMAdapter:
    adapter_id = "modelblame.tiny-causal-lm.v1"

    def build_experiment(self, config: Any, *, device: torch.device) -> Any:
        from modelblame.training.state import build_experiment_state

        return build_experiment_state(config, device=device)

    def build_dataset(self, config: Any) -> Any:
        from modelblame.data.indexed import IndexedDataset

        data = config if isinstance(config, Mapping) else config.model_dump()
        section = data.get("data", data.get("dataset", data))
        return IndexedDataset.from_path(Path(section["path"]))

    def build_batch(self, state: Any, event: Any) -> TrainingBatch:
        device = next(state.model.parameters()).device
        return TrainingBatch(
            input_ids=torch.tensor(event.input_ids, dtype=torch.long, device=device),
            attention_mask=torch.tensor(
                event.attention_mask, dtype=torch.bool, device=device
            ),
            loss_weights=torch.tensor(
                event.loss_weights, dtype=torch.float32, device=device
            ),
            occurrence_spans=event.occurrence_spans,
            original_loss_denominator=float(event.original_loss_denominator),
            global_step=event.global_step,
            microbatch_index=event.microbatch_index,
        )

    def compute_per_occurrence_loss(
        self, state: Any, batch: TrainingBatch
    ) -> PerOccurrenceLoss:
        state.model.eval()
        with torch.no_grad():
            logits = state.model(batch.input_ids, batch.attention_mask)
            losses, counts = per_occurrence_losses(logits, batch)
        return PerOccurrenceLoss(losses=losses, token_counts=counts)

    def apply_training_step(
        self,
        state: Any,
        batches: Sequence[TrainingBatch],
        intervention: StepIntervention | None,
    ) -> Sequence[StepResult]:
        if len(batches) != state.training_config.gradient_accumulation:
            raise ValueError(
                "recorded gradient-accumulation batch count does not match"
            )
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        results: list[StepResult] = []
        precision_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[state.training_config.precision]
        autocast_enabled = state.training_config.precision != "float32"
        for index, batch in enumerate(batches):
            weights = batch.loss_weights
            normalization = "FIXED_DENOMINATOR"
            if intervention is not None:
                weights = apply_occurrence_intervention(batch, intervention)
                normalization = intervention.normalization
            with torch.autocast(
                device_type=batch.input_ids.device.type,
                dtype=precision_dtype,
                enabled=autocast_enabled,
            ):
                logits = state.model(batch.input_ids, batch.attention_mask)
                result = masked_causal_loss(
                    logits,
                    batch.input_ids,
                    weights,
                    original_denominator=batch.original_loss_denominator,
                    normalization=normalization,
                )
            scaled_loss = result.loss / len(batches)
            if state.scaler is None:
                scaled_loss.backward()
            else:
                state.scaler.scale(scaled_loss).backward()
            occurrence_loss, occurrence_count = per_occurrence_losses(logits, batch)
            output_hash = hashlib.sha256(
                logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            results.append(
                StepResult(
                    loss=float(result.loss.detach()),
                    denominator=float(result.denominator),
                    optimizer_stepped=index == len(batches) - 1,
                    output_hash=output_hash,
                    per_occurrence=PerOccurrenceLoss(
                        losses=occurrence_loss, token_counts=occurrence_count
                    ),
                )
            )
        if state.training_config.max_grad_norm is not None:
            if state.scaler is not None:
                state.scaler.unscale_(state.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in state.model.parameters()
                    if parameter.requires_grad
                ],
                state.training_config.max_grad_norm,
            )
        if state.scaler is None:
            state.optimizer.step()
        else:
            state.scaler.step(state.optimizer)
            state.scaler.update()
        state.scheduler.step()
        state.cursor.global_step += 1
        state.cursor.microbatch = 0
        state.cursor.gradient_accumulation_position = 0
        return results

    def save_state(self, state: Any, destination: Path) -> Any:
        from modelblame.checkpoint.format import save_checkpoint

        return save_checkpoint(state, destination)

    def load_state(self, checkpoint: Path, *, device: torch.device) -> Any:
        from modelblame.checkpoint.format import load_checkpoint

        return load_checkpoint(checkpoint, device=device)

    def evaluate_behavior(self, state: Any, contract: Any, split: Any) -> Any:
        from modelblame.behavior.evaluate import evaluate_behavior

        return evaluate_behavior(state, contract, split=split)


def config_as_json(config: TinyCausalLMConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
