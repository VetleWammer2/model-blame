"""Recorded, replay-complete training for one local Hugging Face profile.

This module does not import or wrap ``transformers.Trainer``.  ModelBlame owns
batching, loss normalization, optimizer transitions, ledgers and checkpoints;
Transformers supplies the allowlisted causal-LM implementation.  The optional
dependency is imported only when this adapter is selected.  Neither recording
nor replay contacts the Hugging Face Hub.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from modelblame.adapters.tiny_causal_lm import ByteTokenizer, TinyCausalLMAdapter
from modelblame.checkpoint.cursor import TrainingCursor
from modelblame.checkpoint.hashing import canonical_json_hash, hash_file
from modelblame.config.validation import read_structured_file
from modelblame.training.state import (
    DeterministicLRScheduler,
    TrainingConfig,
    normalize_training_settings,
)
from modelblame.util.canonical_json import validate_json_tree

HUGGINGFACE_ADAPTER_ID = "modelblame.huggingface-causal-lm.v1"
HUGGINGFACE_PROFILE = "gpt2-byte-cpu-fp32-v1"
SUPPORTED_TRANSFORMERS_MAJOR_MINOR = (4, 57)
SUPPORTED_MODEL_TYPE = "gpt2"
SUPPORTED_MODEL_CLASS = "GPT2LMHeadModel"
ATTENTION_IMPLEMENTATION = "eager"
MAX_CONTEXT_POSITIONS = 4096
MAX_EMBEDDING_SIZE = 1024
MAX_LAYERS = 24
MAX_ATTENTION_HEADS = 64
MAX_INNER_SIZE = 4096
MAX_ESTIMATED_PARAMETERS = 50_000_000


class HuggingFaceUnavailableError(RuntimeError):
    """The optional, version-compatible Transformers dependency is absent."""


class UnsupportedHuggingFaceConfiguration(ValueError):  # noqa: N818
    """The requested HF procedure is outside the recorded adapter profile."""


def validate_transformers_version(current: str, *, recorded: str | None = None) -> None:
    """Enforce the tested minor line and exact restoration patch version."""

    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", current)
    if match is None:
        raise HuggingFaceUnavailableError(
            f"cannot interpret the installed Transformers version: {current!r}"
        )
    major_minor = (int(match.group(1)), int(match.group(2)))
    if major_minor != SUPPORTED_TRANSFORMERS_MAJOR_MINOR:
        expected = ".".join(str(item) for item in SUPPORTED_TRANSFORMERS_MAJOR_MINOR)
        raise HuggingFaceUnavailableError(
            f"recorded Hugging Face v1 requires Transformers {expected}.x; "
            f"installed version is {current}"
        )
    if recorded is not None and current != recorded:
        raise HuggingFaceUnavailableError(
            "Transformers version differs from the recorded checkpoint: "
            f"checkpoint={recorded}, installed={current}"
        )


def _require_transformers(*, recorded: str | None = None) -> tuple[Any, Any, str]:
    try:
        import transformers
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as error:  # pragma: no cover - optional dependency
        raise HuggingFaceUnavailableError(
            "install ModelBlame's 'huggingface' extra to use this adapter"
        ) from error
    current = str(transformers.__version__)
    validate_transformers_version(current, recorded=recorded)
    return GPT2Config, GPT2LMHeadModel, current


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _runtime_versions(transformers_version: str) -> dict[str, str | None]:
    return {
        "transformers": transformers_version,
        "tokenizers": _package_version("tokenizers"),
        "safetensors": _package_version("safetensors"),
        "accelerate": _package_version("accelerate"),
    }


def _json_safe(value: Any) -> Any:
    """Normalize third-party configuration to bounded ordinary JSON values."""

    normalized = json.loads(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    )
    validate_json_tree(normalized, max_depth=32, max_items=100_000)
    return normalized


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"{name} must be a mapping or validated model")


def _reject_pickle_artifacts(model_path: Path) -> None:
    unsafe_suffixes = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"}
    unsafe = sorted(
        item.name
        for item in model_path.iterdir()
        if item.is_file() and item.suffix.lower() in unsafe_suffixes
    )
    if unsafe:
        raise UnsupportedHuggingFaceConfiguration(
            "pickle-based model artifacts are forbidden: " + ", ".join(unsafe)
        )


def _local_file(model_path: Path, name: str) -> Path:
    candidate = model_path / name
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(model_path)
    except (FileNotFoundError, ValueError) as error:
        raise UnsupportedHuggingFaceConfiguration(
            "required local Hugging Face file is missing or escapes its directory: "
            + name
        ) from error
    if candidate.is_symlink() or not resolved.is_file():
        raise UnsupportedHuggingFaceConfiguration(
            f"local Hugging Face file must be a regular in-directory file: {name}"
        )
    return resolved


def _read_local_config(model_path: Path) -> dict[str, Any]:
    config_path = _local_file(model_path, "config.json")
    value = read_structured_file(
        config_path,
        allowed_suffixes=frozenset({".json"}),
        max_bytes=2 * 1024 * 1024,
    )
    forbidden = {
        "auto_map",
        "custom_pipelines",
        "quantization_config",
        "adapter_attn_dim",
    }
    present = sorted(forbidden & set(value))
    if present:
        raise UnsupportedHuggingFaceConfiguration(
            "remote/custom/quantized Hugging Face configuration is unsupported: "
            + ", ".join(present)
        )
    if value.get("model_type") != SUPPORTED_MODEL_TYPE:
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 supports model_type='gpt2' only"
        )
    architectures = value.get("architectures")
    if architectures not in (None, [], [SUPPORTED_MODEL_CLASS]):
        raise UnsupportedHuggingFaceConfiguration(
            f"unsupported Hugging Face model class declaration: {architectures!r}"
        )
    return value


def _effective_gpt2_config(
    raw: Mapping[str, Any], *, context_length: int
) -> tuple[Any, dict[str, Any]]:
    gpt2_config_type, _, _ = _require_transformers()
    config = gpt2_config_type(**dict(raw))
    if int(config.vocab_size) != ByteTokenizer.vocab_size:
        raise UnsupportedHuggingFaceConfiguration(
            f"GPT-2 vocab_size must be {ByteTokenizer.vocab_size} for the recorded "
            "byte-tokenizer profile"
        )
    if not 4 <= context_length <= int(config.n_positions):
        raise UnsupportedHuggingFaceConfiguration(
            "model.context_length must fit within GPT-2 n_positions"
        )
    if int(config.n_embd) <= 0 or int(config.n_layer) <= 0 or int(config.n_head) <= 0:
        raise UnsupportedHuggingFaceConfiguration(
            "GPT-2 embedding, layer, and head counts must be positive"
        )
    if int(config.n_embd) % int(config.n_head):
        raise UnsupportedHuggingFaceConfiguration(
            "GPT-2 n_embd must be divisible by n_head"
        )
    if config.n_inner is not None and int(config.n_inner) <= 0:
        raise UnsupportedHuggingFaceConfiguration(
            "GPT-2 n_inner must be positive when supplied"
        )
    inner_size = int(config.n_inner or 4 * int(config.n_embd))
    estimated_parameters = (
        int(config.vocab_size) * int(config.n_embd)
        + int(config.n_positions) * int(config.n_embd)
        + int(config.n_layer)
        * (
            4 * int(config.n_embd) ** 2
            + 2 * int(config.n_embd) * inner_size
            + 10 * int(config.n_embd)
            + inner_size
        )
        + 2 * int(config.n_embd)
    )
    if (
        int(config.n_positions) > MAX_CONTEXT_POSITIONS
        or int(config.n_embd) > MAX_EMBEDDING_SIZE
        or int(config.n_layer) > MAX_LAYERS
        or int(config.n_head) > MAX_ATTENTION_HEADS
        or inner_size > MAX_INNER_SIZE
        or estimated_parameters > MAX_ESTIMATED_PARAMETERS
    ):
        raise UnsupportedHuggingFaceConfiguration(
            "GPT-2 configuration exceeds the recorded v1 CPU resource bounds"
        )
    if any(
        float(value) != 0.0
        for value in (config.resid_pdrop, config.embd_pdrop, config.attn_pdrop)
    ):
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires zero GPT-2 dropout"
        )
    if bool(getattr(config, "add_cross_attention", False)) or bool(
        getattr(config, "is_encoder_decoder", False)
    ):
        raise UnsupportedHuggingFaceConfiguration(
            "cross-attention and encoder-decoder configurations are unsupported"
        )
    dtype = getattr(config, "dtype", None)
    if dtype not in (None, "float32", torch.float32):
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires an fp32 model configuration"
        )
    config.use_cache = False
    config.return_dict = True
    config.bos_token_id = ByteTokenizer.bos_token_id
    config.eos_token_id = ByteTokenizer.eos_token_id
    config.pad_token_id = ByteTokenizer.pad_token_id
    config._name_or_path = ""
    config._attn_implementation = ATTENTION_IMPLEMENTATION
    normalized = _json_safe(config.to_dict())
    if not isinstance(normalized, dict):  # pragma: no cover - to_dict is a mapping
        raise TypeError("Transformers config did not normalize to a JSON object")
    return config, normalized


@dataclass(frozen=True, slots=True)
class HuggingFaceModelConfig:
    """Self-contained architecture/config identity used for checkpoint replay."""

    schema_version: int
    architecture: str
    profile: str
    model_type: str
    model_class: str
    context_length: int
    transformers_version: str
    runtime_versions: Mapping[str, str | None]
    attention_implementation: str
    hf_config: Mapping[str, Any]
    hf_config_hash: str
    initialization_source: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "architecture": self.architecture,
            "profile": self.profile,
            "model_type": self.model_type,
            "model_class": self.model_class,
            "context_length": self.context_length,
            "transformers_version": self.transformers_version,
            "runtime_versions": dict(self.runtime_versions),
            "attention_implementation": self.attention_implementation,
            "hf_config": dict(self.hf_config),
            "hf_config_hash": self.hf_config_hash,
            "initialization_source": dict(self.initialization_source),
        }
        normalized = _json_safe(value)
        if not isinstance(normalized, dict):  # pragma: no cover - fixed mapping
            raise TypeError("HF model configuration is not a JSON object")
        return normalized

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HuggingFaceModelConfig:
        required = {
            "schema_version",
            "architecture",
            "profile",
            "model_type",
            "model_class",
            "context_length",
            "transformers_version",
            "runtime_versions",
            "attention_implementation",
            "hf_config",
            "hf_config_hash",
            "initialization_source",
        }
        if set(value) != required:
            raise UnsupportedHuggingFaceConfiguration(
                "HF checkpoint model configuration fields are incomplete or unexpected"
            )
        if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
            raise UnsupportedHuggingFaceConfiguration(
                "unsupported HF model configuration schema"
            )
        fixed = {
            "architecture": "huggingface_gpt2",
            "profile": HUGGINGFACE_PROFILE,
            "model_type": SUPPORTED_MODEL_TYPE,
            "model_class": SUPPORTED_MODEL_CLASS,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        }
        for key, expected in fixed.items():
            if value[key] != expected:
                raise UnsupportedHuggingFaceConfiguration(
                    f"HF checkpoint {key} is unsupported: {value[key]!r}"
                )
        hf_config = _as_mapping(value["hf_config"], name="hf_config")
        if canonical_json_hash(hf_config) != value["hf_config_hash"]:
            raise UnsupportedHuggingFaceConfiguration(
                "HF checkpoint configuration hash mismatch"
            )
        context_length = value["context_length"]
        required_integer_config = {
            "vocab_size",
            "n_positions",
            "n_embd",
            "n_layer",
            "n_head",
        }
        if (
            not isinstance(context_length, int)
            or isinstance(context_length, bool)
            or any(
                not isinstance(hf_config.get(name), int)
                or isinstance(hf_config.get(name), bool)
                for name in required_integer_config
            )
            or (
                hf_config.get("n_inner") is not None
                and (
                    not isinstance(hf_config.get("n_inner"), int)
                    or isinstance(hf_config.get("n_inner"), bool)
                )
            )
        ):
            raise UnsupportedHuggingFaceConfiguration(
                "HF checkpoint dimensions must be explicit integers"
            )
        _, canonical_config = _effective_gpt2_config(
            hf_config, context_length=context_length
        )
        if canonical_config != dict(hf_config):
            raise UnsupportedHuggingFaceConfiguration(
                "HF checkpoint configuration is not canonical for the v1 profile"
            )
        recorded_version = str(value["transformers_version"])
        _, _, current_version = _require_transformers(recorded=recorded_version)
        runtime_versions = _as_mapping(
            value["runtime_versions"], name="runtime_versions"
        )
        expected_runtime_keys = {
            "transformers",
            "tokenizers",
            "safetensors",
            "accelerate",
        }
        if set(runtime_versions) != expected_runtime_keys or any(
            item is not None and not isinstance(item, str)
            for item in runtime_versions.values()
        ):
            raise UnsupportedHuggingFaceConfiguration(
                "HF checkpoint runtime version declarations are malformed"
            )
        current_runtime_versions = _runtime_versions(current_version)
        if dict(runtime_versions) != current_runtime_versions:
            raise HuggingFaceUnavailableError(
                "Hugging Face runtime package versions differ from the recorded "
                "checkpoint"
            )
        source = _as_mapping(
            value["initialization_source"], name="initialization_source"
        )
        source_mode = source.get("mode")
        source_hash = source.get("config_sha256")
        weight_files = source.get("weight_files")
        if (
            set(source) != {"mode", "config_sha256", "weight_files"}
            or source_mode not in {"from_config", "from_pretrained"}
            or not isinstance(source_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
            or not isinstance(weight_files, list)
            or (source_mode == "from_config" and weight_files != [])
            or (
                source_mode == "from_pretrained"
                and (
                    len(weight_files) != 1
                    or not isinstance(weight_files[0], Mapping)
                    or set(weight_files[0]) != {"name", "sha256"}
                    or weight_files[0].get("name") != "model.safetensors"
                    or not isinstance(weight_files[0].get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", str(weight_files[0].get("sha256")))
                    is None
                )
            )
        ):
            raise UnsupportedHuggingFaceConfiguration(
                "HF checkpoint initialization provenance is malformed"
            )
        result = cls(
            schema_version=1,
            architecture="huggingface_gpt2",
            profile=HUGGINGFACE_PROFILE,
            model_type=SUPPORTED_MODEL_TYPE,
            model_class=SUPPORTED_MODEL_CLASS,
            context_length=context_length,
            transformers_version=current_version,
            runtime_versions=current_runtime_versions,
            attention_implementation=ATTENTION_IMPLEMENTATION,
            hf_config=dict(hf_config),
            hf_config_hash=str(value["hf_config_hash"]),
            initialization_source=dict(source),
        )
        config_positions = int(result.hf_config.get("n_positions", 0))
        if not 4 <= result.context_length <= config_positions:
            raise UnsupportedHuggingFaceConfiguration(
                "HF checkpoint context length is incompatible with its config"
            )
        return result


class HuggingFaceLogitsModel(nn.Module):
    """Expose the tensor-only forward contract used by ModelBlame scorers."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Hugging Face causal LM did not return tensor logits")
        return logits


@dataclass(slots=True)
class HuggingFaceExperimentState:
    adapter_id: ClassVar[str] = HUGGINGFACE_ADAPTER_ID

    model: nn.Module
    optimizer: torch.optim.AdamW
    scheduler: DeterministicLRScheduler
    scaler: None
    tokenizer: ByteTokenizer
    model_config: HuggingFaceModelConfig
    training_config: TrainingConfig
    cursor: TrainingCursor = field(default_factory=TrainingCursor)
    data_loader_generator: torch.Generator = field(default_factory=torch.Generator)
    packing_generator: torch.Generator = field(default_factory=torch.Generator)
    environment_compatibility: Mapping[str, Any] = field(default_factory=dict)


def _validate_training_profile(
    training_config: TrainingConfig, *, device: torch.device
) -> None:
    if device.type != "cpu" or str(device) != "cpu":
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 supports the CPU device only"
        )
    if training_config.device != "cpu":
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires training.device='cpu'"
        )
    if training_config.precision != "float32":
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 supports fp32 only"
        )
    if training_config.determinism != "strict":
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires strict determinism"
        )
    if training_config.capturable:
        raise UnsupportedHuggingFaceConfiguration(
            "AdamW capturable mode is unsupported by the CPU HF profile"
        )


def _environment_compatibility(model_config: HuggingFaceModelConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": HUGGINGFACE_PROFILE,
        **model_config.runtime_versions,
        "model_type": model_config.model_type,
        "model_class": model_config.model_class,
        "hf_config_hash": model_config.hf_config_hash,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
    }


def _make_state(
    *,
    hf_model: nn.Module,
    model_config: HuggingFaceModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> HuggingFaceExperimentState:
    _validate_training_profile(training_config, device=device)
    wrapped = HuggingFaceLogitsModel(hf_model.to(device=device, dtype=torch.float32))
    if type(hf_model).__name__ != SUPPORTED_MODEL_CLASS:
        raise UnsupportedHuggingFaceConfiguration(
            f"constructed unsupported model class: {type(hf_model).__name__}"
        )
    if bool(getattr(hf_model, "is_gradient_checkpointing", False)):
        raise UnsupportedHuggingFaceConfiguration(
            "gradient checkpointing is unsupported by the recorded HF profile"
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in wrapped.parameters() if parameter.requires_grad],
        lr=training_config.learning_rate,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
        maximize=training_config.maximize,
        capturable=False,
    )
    scheduler = DeterministicLRScheduler(
        optimizer,
        schedule=training_config.scheduler,
        warmup_steps=training_config.warmup_steps,
        total_steps=training_config.steps,
    )
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(training_config.seed + 1)
    packing_generator = torch.Generator(device="cpu")
    packing_generator.manual_seed(training_config.seed + 2)
    return HuggingFaceExperimentState(
        model=wrapped,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        tokenizer=ByteTokenizer(),
        model_config=model_config,
        training_config=training_config,
        data_loader_generator=data_generator,
        packing_generator=packing_generator,
        environment_compatibility=_environment_compatibility(model_config),
    )


def build_huggingface_experiment_state(
    config: Any, *, device: torch.device
) -> HuggingFaceExperimentState:
    root = _as_mapping(config, name="experiment config")
    if {"trainer", "training_args", "resume_from_checkpoint"} & set(root):
        raise UnsupportedHuggingFaceConfiguration(
            "transformers.Trainer configurations/checkpoints are unsupported"
        )
    model_section = _as_mapping(root.get("model", {}), name="model config")
    tokenizer_section = _as_mapping(root.get("tokenizer", {}), name="tokenizer config")
    if str(model_section.get("architecture")) != "huggingface":
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires model.architecture='huggingface'"
        )
    declared_vocab_size = model_section.get("vocab_size", ByteTokenizer.vocab_size)
    if (
        not isinstance(declared_vocab_size, int)
        or isinstance(declared_vocab_size, bool)
        or declared_vocab_size != ByteTokenizer.vocab_size
    ):
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires model.vocab_size=260"
        )
    if model_section.get("lora") is not None:
        raise UnsupportedHuggingFaceConfiguration(
            "LoRA is unsupported by the recorded Hugging Face v1 profile"
        )
    if str(tokenizer_section.get("type", "byte")) != "byte":
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires the built-in byte tokenizer"
        )
    if (
        tokenizer_section.get("path") is not None
        or not bool(tokenizer_section.get("add_bos", True))
        or not bool(tokenizer_section.get("add_eos", True))
    ):
        raise UnsupportedHuggingFaceConfiguration(
            "recorded Hugging Face v1 requires built-in byte tokenization with "
            "BOS and EOS enabled"
        )
    unsupported_model_values = {
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 512,
        "dropout": 0.0,
        "bias": False,
    }
    changed_overrides = sorted(
        name
        for name, default in unsupported_model_values.items()
        if name in model_section and model_section[name] != default
    )
    if changed_overrides:
        raise UnsupportedHuggingFaceConfiguration(
            "Hugging Face architecture overrides belong in local config.json: "
            + ", ".join(changed_overrides)
        )
    training_config = normalize_training_settings(root)
    _validate_training_profile(training_config, device=device)
    model_path_value = model_section.get("local_path")
    if not isinstance(model_path_value, str):
        raise UnsupportedHuggingFaceConfiguration(
            "Hugging Face model.local_path must name a local directory"
        )
    model_path = Path(model_path_value).resolve(strict=True)
    if not model_path.is_dir():
        raise UnsupportedHuggingFaceConfiguration(
            "Hugging Face model.local_path must be a directory"
        )
    _reject_pickle_artifacts(model_path)
    raw_config = _read_local_config(model_path)
    context_length = int(model_section.get("context_length", 128))
    hf_config, normalized_config = _effective_gpt2_config(
        raw_config, context_length=context_length
    )
    _, gpt2_model_type, transformers_version = _require_transformers()
    initialization = str(model_section.get("initialization") or "from_pretrained")
    source: dict[str, Any] = {
        "mode": initialization,
        "config_sha256": hash_file(_local_file(model_path, "config.json")),
    }
    if initialization == "from_config":
        hf_model = gpt2_model_type(hf_config)
        source["weight_files"] = []
    elif initialization == "from_pretrained":
        try:
            safe_weights = _local_file(model_path, "model.safetensors")
        except UnsupportedHuggingFaceConfiguration as error:
            raise UnsupportedHuggingFaceConfiguration(
                "from_pretrained requires one local model.safetensors file; "
                "pickle and sharded checkpoints are unsupported"
            ) from error
        if (model_path / "model.safetensors.index.json").exists():
            raise UnsupportedHuggingFaceConfiguration(
                "sharded Hugging Face checkpoints are unsupported by v1"
            )
        try:
            loaded = gpt2_model_type.from_pretrained(
                model_path,
                config=hf_config,
                local_files_only=True,
                use_safetensors=True,
                attn_implementation=ATTENTION_IMPLEMENTATION,
                output_loading_info=True,
            )
        except Exception as error:
            raise UnsupportedHuggingFaceConfiguration(
                "local model.safetensors could not be loaded strictly"
            ) from error
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise UnsupportedHuggingFaceConfiguration(
                "Transformers did not return strict local loading diagnostics"
            )
        hf_model, loading_info = loaded
        if (
            not isinstance(loading_info, Mapping)
            or set(loading_info)
            != {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
            or any(loading_info[name] for name in loading_info)
        ):
            raise UnsupportedHuggingFaceConfiguration(
                "local model.safetensors is incomplete or incompatible with config.json"
            )
        source["weight_files"] = [
            {"name": "model.safetensors", "sha256": hash_file(safe_weights)}
        ]
    else:
        raise UnsupportedHuggingFaceConfiguration(
            f"unsupported Hugging Face initialization mode: {initialization!r}"
        )
    hf_model.config._name_or_path = ""
    hf_model.config.use_cache = False
    normalized_config = _json_safe(hf_model.config.to_dict())
    if not isinstance(normalized_config, dict):  # pragma: no cover
        raise TypeError("Transformers config did not normalize to an object")
    model_config = HuggingFaceModelConfig(
        schema_version=1,
        architecture="huggingface_gpt2",
        profile=HUGGINGFACE_PROFILE,
        model_type=SUPPORTED_MODEL_TYPE,
        model_class=SUPPORTED_MODEL_CLASS,
        context_length=context_length,
        transformers_version=transformers_version,
        runtime_versions=_runtime_versions(transformers_version),
        attention_implementation=ATTENTION_IMPLEMENTATION,
        hf_config=normalized_config,
        hf_config_hash=canonical_json_hash(normalized_config),
        initialization_source=source,
    )
    return _make_state(
        hf_model=hf_model,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )


def build_huggingface_state_from_checkpoint(
    *,
    model_config_value: Mapping[str, Any],
    training_config_value: Mapping[str, Any],
    device: torch.device,
) -> HuggingFaceExperimentState:
    """Reconstruct empty mutable containers before common state restoration."""

    model_config = HuggingFaceModelConfig.from_mapping(model_config_value)
    gpt2_config_type, gpt2_model_type, _ = _require_transformers(
        recorded=model_config.transformers_version
    )
    hf_config = gpt2_config_type(**dict(model_config.hf_config))
    hf_config.use_cache = False
    hf_config.return_dict = True
    hf_config._name_or_path = ""
    hf_config._attn_implementation = ATTENTION_IMPLEMENTATION
    training_config = normalize_training_settings(
        {
            "seed": training_config_value["seed"],
            "determinism": training_config_value["determinism"],
            "optimizer": {
                "lr": training_config_value["learning_rate"],
                "betas": training_config_value["betas"],
                "eps": training_config_value["eps"],
                "weight_decay": training_config_value["weight_decay"],
                "maximize": training_config_value.get("maximize", False),
                "capturable": training_config_value.get("capturable", False),
            },
            "scheduler": {
                "type": training_config_value["scheduler"],
                "warmup_steps": training_config_value["warmup_steps"],
            },
            "training": dict(training_config_value),
            "checkpoints": {"interval": training_config_value["checkpoint_interval"]},
        }
    )
    return _make_state(
        hf_model=gpt2_model_type(hf_config),
        model_config=model_config,
        training_config=training_config,
        device=device,
    )


class HuggingFaceCausalLMAdapter(TinyCausalLMAdapter):
    """Recorded step adapter for the local GPT-2 profile above."""

    adapter_id = HUGGINGFACE_ADAPTER_ID

    def build_experiment(self, config: Any, *, device: torch.device) -> Any:
        return build_huggingface_experiment_state(config, device=device)


def load_local_causal_lm(path: str | Path, **kwargs: Any) -> Any:
    """Guarded compatibility helper for a local SafeTensors model.

    Loading through this helper does not produce a replayable run.
    """

    model_path = Path(path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"local Hugging Face model directory not found: {model_path}"
        )
    _reject_pickle_artifacts(model_path)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - optional dependency
        raise HuggingFaceUnavailableError(
            "install ModelBlame's 'huggingface' extra to use this adapter"
        ) from error
    validate_transformers_version(str(package_version("transformers")))
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("local_files_only", None)
    kwargs.pop("use_safetensors", None)
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        **kwargs,
    )
