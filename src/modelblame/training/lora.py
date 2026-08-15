"""A small, dependency-free LoRA implementation for local experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int = 0
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("LoRA rank cannot be negative")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank residual.

    The base layer is kept as a child module so its parameter names and tensor
    identity remain stable across checkpoint reconstruction.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        if freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = functional.linear(
            functional.linear(self.dropout(inputs), self.lora_a), self.lora_b
        )
        return self.base(inputs) + residual * self.scaling


def inject_lora(
    module: nn.Module,
    config: LoRAConfig,
    *,
    freeze_non_lora: bool = True,
) -> tuple[str, ...]:
    """Replace matching ``nn.Linear`` children and return replaced names."""

    if config.rank == 0:
        return ()
    if freeze_non_lora:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in tuple(parent.named_children()):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and any(
                qualified.endswith(target) for target in config.target_modules
            ):
                setattr(
                    parent,
                    child_name,
                    LoRALinear(
                        child,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout,
                        freeze_base=freeze_non_lora,
                    ),
                )
                replaced.append(qualified)
            else:
                visit(child, qualified)

    visit(module, "")
    if not replaced:
        raise ValueError(
            "LoRA was enabled but no linear module matched: "
            + ", ".join(config.target_modules)
        )
    return tuple(replaced)


def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, parameter in module.named_parameters():
        if name.endswith("lora_a") or name.endswith("lora_b"):
            yield parameter
