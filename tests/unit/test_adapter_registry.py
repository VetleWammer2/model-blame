from __future__ import annotations

import pytest

from modelblame.adapters import registry
from modelblame.adapters.base import ExperimentAdapter


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("tiny_causal_lm", "modelblame.tiny-causal-lm.v1"),
        (
            "modelblame.tiny-causal-lm.v1",
            "modelblame.tiny-causal-lm.v1",
        ),
        ("huggingface", "modelblame.huggingface-causal-lm.v1"),
        ("huggingface_causal_lm", "modelblame.huggingface-causal-lm.v1"),
        (
            "modelblame.huggingface-causal-lm.v1",
            "modelblame.huggingface-causal-lm.v1",
        ),
    ],
)
def test_adapter_aliases_are_fixed(alias: str, canonical: str) -> None:
    assert registry.normalize_adapter_id(alias) == canonical


@pytest.mark.parametrize("adapter_id", ["", "unknown", "MODELBLAME.TINY", None])
def test_unknown_adapter_ids_fail_closed(adapter_id: object) -> None:
    with pytest.raises(ValueError, match="adapter"):
        registry.normalize_adapter_id(adapter_id)


def test_source_paths_are_fixed_without_importing_adapter_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def record_import(module: str) -> None:
        imported.append(module)
        raise AssertionError("metadata lookup imported executable adapter code")

    monkeypatch.setattr(registry.importlib, "import_module", record_import)
    assert registry.adapter_source_path("tiny_causal_lm").name == "tiny_causal_lm.py"
    assert registry.adapter_source_path("huggingface").name == "huggingface.py"
    assert imported == []


def test_get_adapter_loads_the_selected_trusted_implementation() -> None:
    adapter = registry.get_adapter("tiny_causal_lm")
    assert isinstance(adapter, ExperimentAdapter)
    assert adapter.adapter_id == "modelblame.tiny-causal-lm.v1"
