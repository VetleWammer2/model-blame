"""Optional adapter helpers for local Hugging Face causal language models.

The dependency is imported lazily and this module never downloads a model.  A
caller must supply a local path and opt into trusted local adapter execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class HuggingFaceUnavailableError(RuntimeError):
    pass


def load_local_causal_lm(path: str | Path, **kwargs: Any) -> Any:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"local Hugging Face model directory not found: {model_path}"
        )
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - optional dependency
        raise HuggingFaceUnavailableError(
            "install ModelBlame's 'huggingface' extra to use this adapter"
        ) from error
    kwargs.pop("trust_remote_code", None)
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        **kwargs,
    )
