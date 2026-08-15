"""Complete mutable state for the built-in recorded training program."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from modelblame.adapters.tiny_causal_lm import (
    ByteTokenizer,
    TinyCausalLM,
    TinyCausalLMConfig,
    TinyLlamaCausalLM,
    TinyLlamaConfig,
)
from modelblame.checkpoint.cursor import TrainingCursor


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    steps: int = 10
    batch_size: int = 2
    gradient_accumulation: int = 1
    learning_rate: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    maximize: bool = False
    capturable: bool = False
    warmup_steps: int = 0
    scheduler: str = "constant"
    seed: int = 17
    determinism: str = "strict"
    device: str = "cpu"
    precision: str = "float32"
    max_grad_norm: float | None = 1.0
    checkpoint_interval: int = 5

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.gradient_accumulation <= 0:
            raise ValueError(
                "steps, batch_size, and gradient_accumulation must be positive"
            )
        if self.learning_rate <= 0 or self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer hyperparameters are invalid")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("AdamW betas must each be in [0,1)")
        if self.warmup_steps < 0 or self.checkpoint_interval <= 0:
            raise ValueError("warmup and checkpoint interval are invalid")
        if self.scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("scheduler must be constant, linear, or cosine")
        if self.precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError("unsupported precision")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicLRScheduler:
    """Small JSON-serializable scheduler with no callable state."""

    schema_version = 1

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        schedule: str,
        warmup_steps: int,
        total_steps: int,
        step_index: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.schedule = schedule
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.step_index = int(step_index)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self._apply()

    def _factor(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return max(step, 1) / self.warmup_steps
        if self.schedule == "constant":
            return 1.0
        usable = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / usable))
        if self.schedule == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self) -> None:
        factor = self._factor(self.step_index)
        for group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs, strict=True
        ):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.step_index += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schedule": self.schedule,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "step_index": self.step_index,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != self.schema_version:
            raise ValueError("unsupported scheduler state schema")
        if state["schedule"] != self.schedule:
            raise ValueError("scheduler kind does not match checkpoint")
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.step_index = int(state["step_index"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler parameter-group count mismatch")
        self._apply()


@dataclass(slots=True)
class ExperimentState:
    model: nn.Module
    optimizer: torch.optim.AdamW
    scheduler: DeterministicLRScheduler
    scaler: Any | None
    tokenizer: ByteTokenizer
    model_config: TinyCausalLMConfig | TinyLlamaConfig
    training_config: TrainingConfig
    cursor: TrainingCursor = field(default_factory=TrainingCursor)
    data_loader_generator: torch.Generator = field(default_factory=torch.Generator)
    packing_generator: torch.Generator = field(default_factory=torch.Generator)


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)  # type: ignore[arg-type]
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="python")
    raise TypeError("experiment config must be a mapping, dataclass, or Pydantic model")


def normalize_training_config(
    config: Any,
) -> tuple[TinyCausalLMConfig | TinyLlamaConfig, TrainingConfig]:
    root = _as_mapping(config)
    model_section = _as_mapping(root.get("model", {}))
    adapter_name = str(root.get("adapter", "tiny_causal_lm"))
    if adapter_name not in {"tiny_causal_lm", "modelblame.tiny-causal-lm.v1"}:
        raise ValueError(f"unsupported training adapter: {adapter_name}")
    architecture = str(model_section.get("architecture", "tiny_causal_lm"))
    if architecture not in {"tiny_causal_lm", "llama_style"}:
        raise ValueError(
            f"built-in training state cannot construct architecture: {architecture}"
        )
    tokenizer_section = _as_mapping(root.get("tokenizer", {}))
    tokenizer_type = str(tokenizer_section.get("type", "byte"))
    if tokenizer_type != "byte":
        raise ValueError(
            "the tiny causal-LM adapter requires the deterministic byte tokenizer"
        )
    optimizer_section = _as_mapping(root.get("optimizer", {}))
    scheduler_section = _as_mapping(root.get("scheduler", {}))
    training_section = _as_mapping(root.get("training", {}))
    checkpoint_section = _as_mapping(
        root.get("checkpoints", root.get("checkpoint", {}))
    )
    model_config: TinyCausalLMConfig | TinyLlamaConfig
    if architecture == "llama_style":
        model_config = TinyLlamaConfig.from_mapping(model_section)
    else:
        model_config = TinyCausalLMConfig.from_mapping(model_section)
    betas_value = optimizer_section.get("betas", (0.9, 0.999))
    seed_value = training_section.get("seed")
    if seed_value is None:
        seed_value = root.get("seed", 17)
    if seed_value is None:
        raise ValueError("training seed cannot be null")
    determinism_value = training_section.get("determinism")
    if determinism_value is None:
        determinism_value = root.get("determinism", "strict")
    precision_value = str(training_section.get("precision", "float32"))
    precision_value = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(
        precision_value, precision_value
    )
    training_config = TrainingConfig(
        steps=int(training_section.get("steps", training_section.get("max_steps", 10))),
        batch_size=int(training_section.get("batch_size", 2)),
        gradient_accumulation=int(
            training_section.get(
                "gradient_accumulation",
                training_section.get("gradient_accumulation_steps", 1),
            )
        ),
        learning_rate=float(
            optimizer_section.get("lr", optimizer_section.get("learning_rate", 3e-3))
        ),
        betas=(float(betas_value[0]), float(betas_value[1])),
        eps=float(optimizer_section.get("eps", 1e-8)),
        weight_decay=float(optimizer_section.get("weight_decay", 0.01)),
        maximize=bool(optimizer_section.get("maximize", False)),
        capturable=bool(optimizer_section.get("capturable", False)),
        warmup_steps=int(scheduler_section.get("warmup_steps", 0)),
        scheduler=str(scheduler_section.get("type", "constant")),
        seed=int(seed_value),
        determinism=str(determinism_value),
        device=str(training_section.get("device", "cpu")),
        precision=precision_value,
        max_grad_norm=(
            None
            if training_section.get("max_grad_norm", 1.0) is None
            else float(training_section.get("max_grad_norm", 1.0))
        ),
        checkpoint_interval=int(checkpoint_section.get("interval", 5)),
    )
    return model_config, training_config


def build_experiment_state(config: Any, *, device: torch.device) -> ExperimentState:
    model_config, training_config = normalize_training_config(config)
    if training_config.precision == "float16" and device.type == "cpu":
        raise ValueError("float16 training is not supported on CPU")
    if training_config.capturable and device.type != "cuda":
        raise ValueError("AdamW capturable mode requires a CUDA device")
    model: nn.Module
    if isinstance(model_config, TinyLlamaConfig):
        model = TinyLlamaCausalLM(model_config).to(device)
    else:
        model = TinyCausalLM(model_config).to(device)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=training_config.learning_rate,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
        maximize=training_config.maximize,
        capturable=training_config.capturable,
    )
    scheduler = DeterministicLRScheduler(
        optimizer,
        schedule=training_config.scheduler,
        warmup_steps=training_config.warmup_steps,
        total_steps=training_config.steps,
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if training_config.precision == "float16" and device.type == "cuda"
        else None
    )
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(training_config.seed + 1)
    packing_generator = torch.Generator(device="cpu")
    packing_generator.manual_seed(training_config.seed + 2)
    return ExperimentState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        tokenizer=ByteTokenizer(),
        model_config=model_config,
        training_config=training_config,
        data_loader_generator=data_generator,
        packing_generator=packing_generator,
    )
