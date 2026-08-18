from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from modelblame.checkpoint.hashing import hash_tensors
from modelblame.checkpoint.model import load_model, save_model


class _TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.output = nn.Linear(4, 7, bias=False)
        self.output.weight = self.embedding.weight


class _UntiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.output = nn.Linear(4, 7, bias=False)


def _write_legacy_model(model: nn.Module, path: Path) -> dict[str, object]:
    tensors = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    state_hash = hash_tensors(tensors)
    save_file(
        tensors,
        path,
        metadata={
            "schema_version": "1",
            "state_hash": state_hash,
            "format": "modelblame-model-state",
        },
    )
    return {
        "schema_version": 1,
        "state_hash": state_hash,
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in tensors.items()
        },
        "aliases": [],
    }


def test_tied_model_round_trip_records_and_restores_aliases(tmp_path: Path) -> None:
    torch.manual_seed(19)
    source = _TiedModel()
    expected = source.embedding.weight.detach().clone()
    path = tmp_path / "model.safetensors"

    metadata = save_model(source, path)

    assert metadata["schema_version"] == 2
    assert metadata["aliases"] == [["embedding.weight", "output.weight"]]
    assert metadata["state_hash"] == hash_tensors(dict(source.state_dict()))

    restored = _TiedModel()
    with torch.no_grad():
        restored.embedding.weight.zero_()
    load_model(restored, path, metadata, device=torch.device("cpu"))

    assert restored.embedding.weight is restored.output.weight
    assert torch.equal(restored.embedding.weight, expected)


def test_untied_model_keeps_empty_aliases_and_state_hash_identity(
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    model = _UntiedModel()
    expected_hash = hash_tensors(dict(model.state_dict()))

    metadata = save_model(model, tmp_path / "model.safetensors")

    assert metadata["aliases"] == []
    assert metadata["state_hash"] == expected_hash


def test_legacy_schema_one_untied_model_remains_loadable(tmp_path: Path) -> None:
    source = _UntiedModel()
    path = tmp_path / "model.safetensors"
    metadata = _write_legacy_model(source, path)
    restored = _UntiedModel()

    load_model(restored, path, metadata, device=torch.device("cpu"))

    assert all(
        torch.equal(restored.state_dict()[name], tensor)
        for name, tensor in source.state_dict().items()
    )


def test_legacy_schema_one_cannot_claim_tied_topology(tmp_path: Path) -> None:
    source = _TiedModel()
    path = tmp_path / "model.safetensors"
    metadata = _write_legacy_model(source, path)

    with pytest.raises(ValueError, match="legacy model checkpoint"):
        load_model(source, path, metadata, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "aliases",
    [
        None,
        {},
        [["embedding.weight"]],
        [["output.weight", "embedding.weight"]],
        [["embedding.weight", "output.weight", "output.weight"]],
        [["embedding.weight", "unknown.weight"]],
        [
            ["embedding.weight", "output.weight"],
            ["embedding.weight", "output.weight"],
        ],
    ],
)
def test_malformed_alias_metadata_is_rejected(tmp_path: Path, aliases: object) -> None:
    model = _TiedModel()
    path = tmp_path / "model.safetensors"
    metadata = deepcopy(dict(save_model(model, path)))
    metadata["aliases"] = aliases

    with pytest.raises(ValueError, match="alias"):
        load_model(model, path, metadata, device=torch.device("cpu"))


def test_missing_alias_metadata_is_rejected(tmp_path: Path) -> None:
    model = _UntiedModel()
    path = tmp_path / "model.safetensors"
    metadata = dict(save_model(model, path))
    del metadata["aliases"]

    with pytest.raises(ValueError, match="alias metadata is missing"):
        load_model(model, path, metadata, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("source_type", "target_type"),
    [(_TiedModel, _UntiedModel), (_UntiedModel, _TiedModel)],
)
def test_alias_metadata_must_match_reconstructed_model_topology(
    tmp_path: Path,
    source_type: type[nn.Module],
    target_type: type[nn.Module],
) -> None:
    source = source_type()
    path = tmp_path / "model.safetensors"
    metadata = save_model(source, path)

    with pytest.raises(ValueError, match="does not match constructed model"):
        load_model(target_type(), path, metadata, device=torch.device("cpu"))


def test_unequal_values_for_declared_aliases_are_rejected(tmp_path: Path) -> None:
    source = _TiedModel()
    valid_path = tmp_path / "valid.safetensors"
    metadata = deepcopy(dict(save_model(source, valid_path)))
    tensors = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in source.state_dict().items()
    }
    tensors["output.weight"].add_(1.0)
    state_hash = hash_tensors(tensors)
    malformed_path = tmp_path / "unequal.safetensors"
    save_file(
        tensors,
        malformed_path,
        metadata={
            "schema_version": "2",
            "state_hash": state_hash,
            "format": "modelblame-model-state",
        },
    )
    metadata["state_hash"] = state_hash

    with pytest.raises(ValueError, match="do not contain equal values"):
        load_model(
            _TiedModel(),
            malformed_path,
            metadata,
            device=torch.device("cpu"),
        )
