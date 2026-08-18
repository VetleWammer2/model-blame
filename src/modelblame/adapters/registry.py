"""Fixed registry for trusted, executable training adapters.

Adapter identifiers are recorded in signed run metadata, but they are untrusted
input when a run is opened.  This module maps supported identifiers to
package-owned implementations and nothing else; an identifier is never treated
as an import path.  Implementations are imported only when an adapter is
selected, which keeps optional dependencies optional.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from modelblame.adapters.base import ExperimentAdapter


@dataclass(frozen=True, slots=True)
class _AdapterSpec:
    adapter_id: str
    module: str
    class_name: str
    source_filename: str


_TINY_CAUSAL_LM = _AdapterSpec(
    adapter_id="modelblame.tiny-causal-lm.v1",
    module="modelblame.adapters.tiny_causal_lm",
    class_name="TinyCausalLMAdapter",
    source_filename="tiny_causal_lm.py",
)
_HUGGINGFACE_CAUSAL_LM = _AdapterSpec(
    adapter_id="modelblame.huggingface-causal-lm.v1",
    module="modelblame.adapters.huggingface",
    class_name="HuggingFaceCausalLMAdapter",
    source_filename="huggingface.py",
)

_SPECS_BY_ID = {
    _TINY_CAUSAL_LM.adapter_id: _TINY_CAUSAL_LM,
    _HUGGINGFACE_CAUSAL_LM.adapter_id: _HUGGINGFACE_CAUSAL_LM,
}
_ALIASES = {
    "tiny_causal_lm": _TINY_CAUSAL_LM.adapter_id,
    _TINY_CAUSAL_LM.adapter_id: _TINY_CAUSAL_LM.adapter_id,
    "huggingface": _HUGGINGFACE_CAUSAL_LM.adapter_id,
    "huggingface_causal_lm": _HUGGINGFACE_CAUSAL_LM.adapter_id,
    _HUGGINGFACE_CAUSAL_LM.adapter_id: _HUGGINGFACE_CAUSAL_LM.adapter_id,
}


def normalize_adapter_id(adapter_id: object) -> str:
    """Return the canonical ID for a fixed, trusted adapter alias.

    The match is exact.  Recorded metadata cannot reach a module name, class
    name or source path, and an unknown identifier fails closed before any
    executable adapter code loads.
    """

    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("adapter ID must be a non-empty string")
    try:
        return _ALIASES[adapter_id]
    except KeyError as error:
        raise ValueError(f"unsupported trusted adapter ID: {adapter_id!r}") from error


def get_adapter(adapter_id: object) -> ExperimentAdapter:
    """Instantiate the package-owned adapter selected by ``adapter_id``."""

    canonical_id = normalize_adapter_id(adapter_id)
    spec = _SPECS_BY_ID[canonical_id]
    module = importlib.import_module(spec.module)
    adapter_type = getattr(module, spec.class_name, None)
    if adapter_type is None:
        raise RuntimeError(
            f"trusted adapter implementation is unavailable: {canonical_id}"
        )
    adapter = adapter_type()
    if not isinstance(adapter, ExperimentAdapter):
        raise RuntimeError(
            f"trusted adapter does not implement ExperimentAdapter: {canonical_id}"
        )
    if adapter.adapter_id != canonical_id:
        raise RuntimeError(
            "trusted adapter implementation ID differs from its registry entry"
        )
    return adapter


def adapter_source_path(adapter_id: object) -> Path:
    """Return the package-owned implementation file used for evidence hashing."""

    canonical_id = normalize_adapter_id(adapter_id)
    spec = _SPECS_BY_ID[canonical_id]
    return Path(__file__).with_name(spec.source_filename).resolve()
