from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file, save_file

transformers = pytest.importorskip("transformers")

from modelblame.adapters.huggingface import (  # noqa: E402
    HUGGINGFACE_ADAPTER_ID,
    HuggingFaceModelConfig,
    HuggingFaceUnavailableError,
    UnsupportedHuggingFaceConfiguration,
    build_huggingface_experiment_state,
    validate_transformers_version,
)
from modelblame.behavior.contract import load_contract_artifact  # noqa: E402
from modelblame.blame import run_blame  # noqa: E402
from modelblame.checkpoint.cursor import TrainingCursor  # noqa: E402
from modelblame.checkpoint.format import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from modelblame.checkpoint.hashing import (  # noqa: E402
    canonical_json_hash,
    hash_file,
    hash_tensors,
)
from modelblame.checkpoint.optimizer import (  # noqa: E402
    optimizer_state_components,
)
from modelblame.config.experiment import ExperimentConfig  # noqa: E402
from modelblame.data.identity import compute_occurrence_id  # noqa: E402
from modelblame.evidence.verify import verify_bundle  # noqa: E402
from modelblame.patch.schema import (  # noqa: E402
    GradientAblateOperation,
    Patch,
)
from modelblame.recorded import RecordedRun  # noqa: E402
from modelblame.replay.audit import audit_run  # noqa: E402
from modelblame.replay.cache import ReplayCache, ReplayCacheKey  # noqa: E402
from modelblame.replay.engine import ReplayEngine  # noqa: E402
from modelblame.replay.result import ReplayGrade  # noqa: E402
from modelblame.timeline.evaluate import evaluate_timeline  # noqa: E402
from modelblame.training.loop import TrainingRunResult, train_experiment  # noqa: E402

MANDATORY_CHECKPOINT_FILES = frozenset(
    {
        "cursor.json",
        "gradients.safetensors",
        "hashes.json",
        "manifest.json",
        "model.safetensors",
        "optimizer.json",
        "optimizer.safetensors",
        "rng.json",
        "rng.safetensors",
        "scaler.json",
        "scheduler.json",
    }
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _reauthenticate_checkpoint(path: Path) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_json(manifest_path, manifest)
    hashes_path = path / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["files"] = {name: hash_file(path / name) for name in hashes["files"]}
    hashes["checkpoint_hash"] = canonical_json_hash(manifest)
    _write_json(hashes_path, hashes)


def _adapter_config(model_path: Path) -> dict[str, Any]:
    return {
        "adapter": HUGGINGFACE_ADAPTER_ID,
        "seed": 101,
        "determinism": "strict",
        "model": {
            "architecture": "huggingface",
            "local_path": str(model_path),
            "initialization": "from_config",
            "context_length": 12,
        },
        "tokenizer": {"type": "byte"},
        "optimizer": {
            "lr": 0.003,
            "betas": [0.8, 0.95],
            "eps": 1e-7,
            "weight_decay": 0.02,
            "capturable": False,
        },
        "scheduler": {"type": "linear", "warmup_steps": 1},
        "training": {
            "steps": 2,
            "batch_size": 1,
            "gradient_accumulation": 2,
            "device": "cpu",
            "precision": "fp32",
            "max_grad_norm": 1.0,
        },
        "checkpoints": {"interval": 1},
    }


def _schema_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "hf-schema-test",
        "adapter": "huggingface",
        "seed": 101,
        "determinism": "strict",
        "dataset": {"path": "train.jsonl"},
        "model": {
            "architecture": "huggingface",
            "local_path": "model",
            "initialization": "from_config",
            "context_length": 12,
        },
        "optimizer": {"lr": 0.003},
        "scheduler": {"type": "constant", "warmup_steps": 0},
        "training": {
            "steps": 1,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "device": "cpu",
            "precision": "fp32",
        },
        "checkpoints": {"interval": 1},
    }


@pytest.fixture(scope="session")
def hf_recorded_run(tmp_path_factory: pytest.TempPathFactory) -> TrainingRunResult:
    root = tmp_path_factory.mktemp("hf-recorded-training")
    model_path = root / "model"
    model_path.mkdir()
    transformers.GPT2Config(
        vocab_size=260,
        n_positions=12,
        n_ctx=12,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_inner=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    ).save_pretrained(model_path)

    # Rows zero and one are intentional logical duplicates. The short examples
    # also force several epochs in four recorded microbatches.
    records = [
        {"prompt": "a", "completion": "x"},
        {"prompt": "a", "completion": "x"},
        {"prompt": "b", "completion": "y"},
    ]
    (root / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "experiment.toml").write_text(
        """
schema_version = 1
name = "hf-recorded-regression"
adapter = "huggingface"
seed = 101
determinism = "strict"

[dataset]
path = "train.jsonl"
source = "hf-regression"

[model]
architecture = "huggingface"
local_path = "model"
initialization = "from_config"
context_length = 12

[optimizer]
lr = 0.003
betas = [0.8, 0.95]
eps = 0.0000001
weight_decay = 0.02
capturable = false

[scheduler]
type = "linear"
warmup_steps = 1

[training]
steps = 2
batch_size = 1
gradient_accumulation = 2
device = "cpu"
precision = "fp32"
max_grad_norm = 1.0

[checkpoints]
interval = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return train_experiment(root / "experiment.toml", root / "runs")


@pytest.mark.parametrize(
    "version",
    ["4.56.9", "4.58.0", "5.57.1", "4.57", "not-a-version"],
)
def test_transformers_version_boundary_is_fail_closed(version: str) -> None:
    with pytest.raises(HuggingFaceUnavailableError):
        validate_transformers_version(version)


def test_transformers_recorded_version_requires_exact_patch_match() -> None:
    validate_transformers_version("4.57.1")
    validate_transformers_version("4.57.99")
    validate_transformers_version("4.57.1", recorded="4.57.1")

    with pytest.raises(HuggingFaceUnavailableError, match="differs from the recorded"):
        validate_transformers_version("4.57.1", recorded="4.57.0")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trainer", {}),
        ("training_args", {"per_device_train_batch_size": 1}),
        ("resume_from_checkpoint", "checkpoint-1"),
    ],
)
def test_experiment_schema_rejects_transformers_trainer_configuration(
    field: str, value: object
) -> None:
    config = _schema_config()
    config[field] = value
    with pytest.raises(ValidationError, match="Trainer configurations"):
        ExperimentConfig.model_validate(config)


def test_experiment_schema_rejects_hf_checkpoint_pruning() -> None:
    config = _schema_config()
    config["checkpoints"]["keep_last"] = 1

    with pytest.raises(ValidationError, match="retains every checkpoint"):
        ExperimentConfig.model_validate(config)


@pytest.mark.parametrize(
    "field", ["trainer", "training_args", "resume_from_checkpoint"]
)
def test_adapter_rejects_transformers_trainer_configuration_before_loading(
    tmp_path: Path, field: str
) -> None:
    config = _adapter_config(tmp_path / "model-does-not-need-to-exist")
    config[field] = {} if field != "resume_from_checkpoint" else "checkpoint-1"
    with pytest.raises(
        UnsupportedHuggingFaceConfiguration, match=r"Trainer configurations/checkpoints"
    ):
        build_huggingface_experiment_state(config, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("training", "device", "cuda", "CPU device only|training.device"),
        ("training", "precision", "bf16", "fp32 only"),
        ("training", "determinism", "best_effort", "strict determinism"),
        ("optimizer", "capturable", True, "capturable mode"),
        ("tokenizer", "type", "huggingface", "byte tokenizer"),
    ],
)
def test_adapter_rejects_unsupported_recorded_training_profile(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _adapter_config(tmp_path / "model-does-not-need-to-exist")
    config[section][field] = value
    with pytest.raises(UnsupportedHuggingFaceConfiguration, match=message):
        build_huggingface_experiment_state(config, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("model", "hidden_size", 64, "overrides belong in local config.json"),
        ("tokenizer", "add_bos", False, "BOS and EOS"),
    ],
)
def test_adapter_rejects_silently_ignored_hf_overrides(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    config = _adapter_config(tmp_path / "model-does-not-need-to-exist")
    config[section][field] = value
    with pytest.raises(UnsupportedHuggingFaceConfiguration, match=message):
        build_huggingface_experiment_state(config, device=torch.device("cpu"))


def test_hf_from_pretrained_loads_one_complete_local_safetensors(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    source = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=260,
            n_positions=12,
            n_ctx=12,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_inner=16,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    source.save_pretrained(model_path, safe_serialization=True)
    config = _adapter_config(model_path)
    config["model"]["initialization"] = "from_pretrained"

    restored = build_huggingface_experiment_state(config, device=torch.device("cpu"))

    assert restored.model_config.initialization_source["mode"] == "from_pretrained"
    assert restored.model_config.initialization_source["weight_files"][0]["name"] == (
        "model.safetensors"
    )
    assert torch.equal(
        restored.model.hf_model.transformer.wte.weight,
        source.transformer.wte.weight,
    )


def test_hf_from_pretrained_rejects_incomplete_safetensors(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    source = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=260,
            n_positions=12,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_inner=16,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    source.save_pretrained(model_path, safe_serialization=True)
    weight_path = model_path / "model.safetensors"
    loaded = load_file(weight_path, device="cpu")
    tensors = {name: tensor.clone() for name, tensor in loaded.items()}
    del loaded
    del tensors["transformer.wpe.weight"]
    weight_path.unlink()
    save_file(tensors, weight_path, metadata={"format": "pt"})
    config = _adapter_config(model_path)
    config["model"]["initialization"] = "from_pretrained"

    with pytest.raises(
        UnsupportedHuggingFaceConfiguration, match="incomplete|strictly"
    ):
        build_huggingface_experiment_state(config, device=torch.device("cpu"))


@pytest.mark.parametrize("missing", sorted(MANDATORY_CHECKPOINT_FILES))
def test_hf_checkpoint_requires_every_mandatory_file(
    tmp_path: Path, hf_recorded_run: TrainingRunResult, missing: str
) -> None:
    source = hf_recorded_run.final_checkpoint
    assert {item.name for item in source.iterdir() if item.is_file()} == set(
        MANDATORY_CHECKPOINT_FILES
    )
    malformed = tmp_path / "checkpoint"
    shutil.copytree(source, malformed)
    (malformed / missing).unlink()

    with pytest.raises(ValueError):
        verify_checkpoint(malformed)


def test_hf_checkpoint_rejects_unexpected_physical_artifact(
    tmp_path: Path, hf_recorded_run: TrainingRunResult
) -> None:
    malformed = tmp_path / "checkpoint"
    shutil.copytree(hf_recorded_run.final_checkpoint, malformed)
    (malformed / "pytorch_model.bin").write_bytes(b"not loaded")

    with pytest.raises(ValueError, match="file set|unexpected"):
        verify_checkpoint(malformed)


def test_hf_checkpoint_rejects_removed_zero_valued_cursor_field(
    tmp_path: Path, hf_recorded_run: TrainingRunResult
) -> None:
    malformed = tmp_path / "checkpoint"
    shutil.copytree(hf_recorded_run.final_checkpoint, malformed)
    cursor_path = malformed / "cursor.json"
    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    assert cursor["source_shard"] == 0
    del cursor["source_shard"]
    _write_json(cursor_path, cursor)

    hashes_path = malformed / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["files"]["cursor.json"] = hash_file(cursor_path)
    _write_json(hashes_path, hashes)

    with pytest.raises(ValueError, match="cursor|field|incomplete"):
        verify_checkpoint(malformed)


def test_hf_checkpoint_manifest_requires_complete_exact_fields(
    tmp_path: Path, hf_recorded_run: TrainingRunResult
) -> None:
    malformed = tmp_path / "checkpoint"
    shutil.copytree(hf_recorded_run.final_checkpoint, malformed)
    manifest_path = malformed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["model_config"]
    _write_json(manifest_path, manifest)

    hashes_path = malformed / "hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["files"]["manifest.json"] = hash_file(manifest_path)
    hashes["checkpoint_hash"] = canonical_json_hash(manifest)
    _write_json(hashes_path, hashes)

    with pytest.raises(ValueError, match="manifest|field|incomplete"):
        verify_checkpoint(malformed)


def test_hf_checkpoint_rejects_self_consistent_incomplete_optimizer_state(
    tmp_path: Path, hf_recorded_run: TrainingRunResult
) -> None:
    malformed = tmp_path / "checkpoint"
    shutil.copytree(hf_recorded_run.final_checkpoint, malformed)
    optimizer_path = malformed / "optimizer.safetensors"
    metadata_path = malformed / "optimizer.json"
    loaded_tensors = load_file(optimizer_path, device="cpu")
    tensors = {name: tensor.clone() for name, tensor in loaded_tensors.items()}
    del loaded_tensors
    victim = next(name for name in tensors if name.endswith("/exp_avg"))
    del tensors[victim]
    tensor_hash = hash_tensors(tensors)
    optimizer_path.unlink()
    save_file(
        tensors,
        optimizer_path,
        metadata={"schema_version": "1", "tensor_hash": tensor_hash},
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tensor_index"] = [
        item for item in metadata["tensor_index"] if item["tensor_key"] != victim
    ]
    metadata["tensor_hash"] = tensor_hash
    metadata_without_hashes = {
        key: value
        for key, value in metadata.items()
        if key not in {"metadata_hash", "tensor_hash"}
    }
    metadata["metadata_hash"] = canonical_json_hash(metadata_without_hashes)
    _write_json(metadata_path, metadata)
    manifest_path = malformed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state_hashes"]["optimizer"] = canonical_json_hash(
        {"metadata": metadata["metadata_hash"], "tensors": tensor_hash}
    )
    _write_json(manifest_path, manifest)
    _reauthenticate_checkpoint(malformed)

    with pytest.raises(ValueError, match="optimizer state is incomplete"):
        load_checkpoint(malformed)


def test_hf_checkpoint_rejects_self_consistent_incomplete_scaler_metadata(
    tmp_path: Path, hf_recorded_run: TrainingRunResult
) -> None:
    malformed = tmp_path / "checkpoint"
    shutil.copytree(hf_recorded_run.final_checkpoint, malformed)
    scaler_path = malformed / "scaler.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    del scaler["state"]
    _write_json(scaler_path, scaler)
    manifest_path = malformed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state_hashes"]["scaler"] = canonical_json_hash(scaler)
    _write_json(manifest_path, manifest)
    _reauthenticate_checkpoint(malformed)

    with pytest.raises(ValueError, match="scaler checkpoint is malformed"):
        verify_checkpoint(malformed)


def test_hf_tied_weight_alias_is_declared_and_restored(
    hf_recorded_run: TrainingRunResult,
) -> None:
    manifest = json.loads(
        (hf_recorded_run.final_checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    tied_group = ["hf_model.lm_head.weight", "hf_model.transformer.wte.weight"]
    assert manifest["model"]["schema_version"] == 2
    assert tied_group in manifest["model"]["aliases"]

    restored = load_checkpoint(hf_recorded_run.final_checkpoint)
    hf_model = restored.model.hf_model
    assert hf_model.lm_head.weight is hf_model.transformer.wte.weight
    state = restored.model.state_dict()
    assert torch.equal(state[tied_group[0]], state[tied_group[1]])
    assert (
        state[tied_group[0]].untyped_storage().data_ptr()
        == state[tied_group[1]].untyped_storage().data_ptr()
    )


def test_hf_optimizer_tensor_state_and_groups_restore_exactly(
    hf_recorded_run: TrainingRunResult,
) -> None:
    checkpoint = hf_recorded_run.final_checkpoint
    expected_metadata = json.loads(
        (checkpoint / "optimizer.json").read_text(encoding="utf-8")
    )
    expected_tensors = load_file(checkpoint / "optimizer.safetensors", device="cpu")

    restored = load_checkpoint(checkpoint)
    actual_tensors, actual_metadata = optimizer_state_components(
        restored.optimizer, restored.model
    )

    assert actual_metadata == expected_metadata
    assert set(actual_tensors) == set(expected_tensors)
    assert all(
        torch.equal(actual_tensors[name], expected_tensors[name])
        for name in expected_tensors
    )
    assert len(actual_metadata["parameter_groups"]) == 1
    assert actual_metadata["parameter_groups"][0]["parameters"]
    assert actual_metadata["tensor_index"]


def test_hf_checkpoint_restores_all_rng_and_cursor_state(
    tmp_path: Path, hf_recorded_run: TrainingRunResult
) -> None:
    state = load_checkpoint(hf_recorded_run.final_checkpoint)
    state.cursor = TrainingCursor(
        global_step=2,
        epoch=7,
        source_shard=3,
        source_row=2,
        sampler_offset=29,
        microbatch=1,
        gradient_accumulation_position=1,
        packed_sequence_count=31,
    )
    random.seed(211)
    np.random.seed(223)
    torch.manual_seed(227)
    state.data_loader_generator.manual_seed(229)
    state.packing_generator.manual_seed(233)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(state, checkpoint)

    expected = {
        "python": [random.random() for _ in range(4)],  # noqa: S311
        "numpy": np.random.random(4),
        "torch": torch.rand(4),
        "data_loader": torch.rand(4, generator=state.data_loader_generator),
        "packing": torch.rand(4, generator=state.packing_generator),
    }
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    state.data_loader_generator.manual_seed(4)
    state.packing_generator.manual_seed(5)

    restored = load_checkpoint(checkpoint)

    assert restored.cursor.to_dict() == state.cursor.to_dict()
    assert set(restored.cursor.to_dict()) == set(TrainingCursor.__dataclass_fields__)
    assert [random.random() for _ in range(4)] == expected["python"]  # noqa: S311
    assert np.array_equal(np.random.random(4), expected["numpy"])
    assert torch.equal(torch.rand(4), expected["torch"])
    assert torch.equal(
        torch.rand(4, generator=restored.data_loader_generator),
        expected["data_loader"],
    )
    assert torch.equal(
        torch.rand(4, generator=restored.packing_generator), expected["packing"]
    )
    rng_metadata = json.loads((checkpoint / "rng.json").read_text(encoding="utf-8"))
    assert {
        "torch_cpu",
        "data_loader_generator",
        "packing_generator",
    }.issubset(rng_metadata["tensor_names"])


def test_hf_occurrence_ledger_preserves_duplicates_reuse_and_exact_spans(
    hf_recorded_run: TrainingRunResult,
) -> None:
    run = RecordedRun.open(hf_recorded_run.run_path, verify_checkpoints=True)
    events = list(run.ledger.iter_batches())
    occurrences = list(run.ledger.iter_occurrences())
    flattened = [
        dict(span)
        for event in events
        for sequence_spans in event.occurrence_spans
        for span in sequence_spans
    ]

    assert run.ledger.validate().valid
    assert [row["occurrence_id"] for row in occurrences] == [
        row["occurrence_id"] for row in flattened
    ]
    assert len({row["occurrence_id"] for row in occurrences}) == len(occurrences)
    assert len(events) == 4
    assert {event.microbatch_index for event in events} == {0, 1}

    rows_by_example: defaultdict[str, set[int]] = defaultdict(set)
    occurrence_counts: Counter[str] = Counter()
    for row in occurrences:
        example_id = str(row["example_id"])
        rows_by_example[example_id].add(int(row["source_row"]))
        occurrence_counts[example_id] += 1
        assert row["occurrence_id"] == compute_occurrence_id(
            run_id=run.run_id,
            global_step=int(row["global_step"]),
            microbatch_index=int(row["microbatch_index"]),
            batch_position=int(row["batch_position"]),
            packed_token_span=(int(row["token_start"]), int(row["token_end"])),
            example_id=example_id,
        )
        assert len(row["prompt_token_mask"]) == (
            int(row["token_end"]) - int(row["token_start"])
        )
        assert len(row["completion_token_mask"]) == len(row["prompt_token_mask"])

    duplicate_ids = [
        example_id
        for example_id, source_rows in rows_by_example.items()
        if source_rows == {0, 1}
    ]
    assert len(duplicate_ids) == 1
    assert occurrence_counts[duplicate_ids[0]] >= 4
    assert max(int(row["epoch"]) for row in occurrences) >= 1


def test_hf_unchanged_replay_audits_every_component_as_bitwise(
    hf_recorded_run: TrainingRunResult,
) -> None:
    audit = audit_run(
        hf_recorded_run.run_path,
        from_step=1,
        to_step=2,
        write_artifact=False,
    )

    assert audit.replay_grade is ReplayGrade.BITWISE
    assert audit.recorded_losses_equal
    assert audit.output_hashes_equal
    assert {component.component for component in audit.components} == {
        "cursor",
        "gradients",
        "model",
        "optimizer",
        "rng",
        "scaler",
        "scheduler",
    }
    assert all(component.bitwise_equal for component in audit.components)


def test_hf_modified_replay_executes_in_isolated_process(
    tmp_path: Path, hf_recorded_run: TrainingRunResult
) -> None:
    run = RecordedRun.open(hf_recorded_run.run_path, verify_checkpoints=True)
    occurrence = next(
        row for row in run.ledger.iter_occurrences() if int(row["global_step"]) == 1
    )
    behavior_path = tmp_path / "behavior.json"
    _write_json(
        behavior_path,
        {
            "schema_version": 1,
            "id": "hf-replay-probe",
            "scorer": {
                "type": "sequence_logprob_margin",
                "preferred": "x",
                "alternative": "y",
            },
            "direction": "greater_is_present",
            "present_threshold": -1000000.0,
            "required_effect": 1000000.0,
            "search": {"prompts": ["a"]},
            "holdout": {"sealed": True, "prompts": ["b"]},
            "statistics": {"bootstrap_samples": 1, "bootstrap_seed": 0},
        },
    )
    _, _, behavior_hash = load_contract_artifact(behavior_path)
    patch = Patch.create(
        run_id=run.run_id,
        run_hash=run.run_hash,
        behavior_contract_hash=behavior_hash,
        operations=[
            GradientAblateOperation(occurrence_ids=(occurrence["occurrence_id"],))
        ],
    )
    patch_path = tmp_path / "patch.json"
    _write_json(patch_path, patch.model_dump(mode="json"))
    manifest = run.manifest
    start_checkpoint = next(item for item in run.checkpoints if int(item["step"]) == 1)
    cache_key = ReplayCacheKey(
        source_checkpoint_hash=str(start_checkpoint["hash"]),
        remaining_history_hash=run.history_hash(start_step=1),
        patch_hash=str(patch.patch_hash),
        adapter_hash=canonical_json_hash(manifest["adapter_id"]),
        training_configuration_hash=canonical_json_hash(
            {
                "optimizer": manifest["optimizer_config"],
                "scheduler": manifest["scheduler_config"],
                "precision": manifest["precision"],
                "gradient_accumulation": manifest["gradient_accumulation"],
            }
        ),
        behavior_contract_hash=behavior_hash,
        environment_compatibility_class=canonical_json_hash(
            manifest["adapter_compatibility"]
        ),
    )
    output = tmp_path / "isolated-output"

    result = ReplayEngine(ReplayCache(tmp_path / "cache")).execute(
        run_directory=run.path,
        patch_path=patch_path,
        behavior_path=behavior_path,
        output_directory=output,
        cache_key=cache_key,
        timeout_seconds=120,
    )

    assert result["returncode"] == 0, result["stderr"]
    assert result["status"] != "INCONCLUSIVE"
    assert result["cache_hit"] is False
    assert result["earliest_affected_step"] == 1
    assert result["steps_replayed"] == 1
    assert result["intervention_semantics"] == {
        "normalization": "FIXED_DENOMINATOR",
        "operations": ["GRADIENT_ABLATE"],
        "selected_occurrences": 1,
    }
    assert result["counterfactual_checkpoint_hash"] != run.checkpoints[-1]["hash"]
    assert (
        verify_checkpoint(output / "counterfactual")
        == result["counterfactual_checkpoint_hash"]
    )


def test_hf_checkpoint_rejects_exact_transformers_version_mismatch(
    hf_recorded_run: TrainingRunResult,
) -> None:
    manifest = json.loads(
        (hf_recorded_run.final_checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    model_config = deepcopy(manifest["model_config"])
    model_config["transformers_version"] = "4.57.0"

    with pytest.raises(HuggingFaceUnavailableError, match="differs from the recorded"):
        HuggingFaceModelConfig.from_mapping(model_config)


def test_hf_checkpoint_rejects_runtime_dependency_version_mismatch(
    hf_recorded_run: TrainingRunResult,
) -> None:
    manifest = json.loads(
        (hf_recorded_run.final_checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    model_config = deepcopy(manifest["model_config"])
    assert set(model_config["runtime_versions"]) == {
        "accelerate",
        "safetensors",
        "tokenizers",
        "transformers",
    }
    model_config["runtime_versions"]["tokenizers"] = "incompatible-test-version"

    with pytest.raises(
        HuggingFaceUnavailableError, match="runtime package versions differ"
    ):
        HuggingFaceModelConfig.from_mapping(model_config)


def test_hf_checkpoint_rejects_oversized_model_before_allocation(
    hf_recorded_run: TrainingRunResult,
) -> None:
    manifest = json.loads(
        (hf_recorded_run.final_checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    model_config = deepcopy(manifest["model_config"])
    model_config["hf_config"]["n_layer"] = 100_000
    model_config["hf_config_hash"] = canonical_json_hash(model_config["hf_config"])

    with pytest.raises(UnsupportedHuggingFaceConfiguration, match="resource bounds"):
        HuggingFaceModelConfig.from_mapping(model_config)


def test_offline_hf_run_reaches_normal_evidence_bundle() -> None:
    example = Path(__file__).resolve().parents[2] / "examples" / "huggingface_tiny"
    with tempfile.TemporaryDirectory(prefix="mbhf-evidence-") as temporary:
        root = Path(temporary)
        training = train_experiment(example / "experiment.toml", root / "runs")
        audit = audit_run(
            training.run_path,
            from_step=0,
            to_step=training.training_steps,
            behavior_path=example / "behavior.yaml",
        )
        timeline = evaluate_timeline(training.run_path, example / "behavior.yaml")
        bundle, result = run_blame(
            training.run_path,
            example / "behavior.yaml",
            output=root / "evidence",
            candidate_limit=5,
            replay_budget=16,
            workers=1,
            methods=("bm25",),
            device="cpu",
        )
        certificate = verify_bundle(bundle, source_run=training.run_path)

        assert audit.replay_grade is ReplayGrade.BITWISE
        assert len(timeline["points"]) == 3
        assert result["final_result"]["status"] == "TARGET_PASSED"
        assert result["final_result"]["accepted"] is True
        assert result["final_result"]["controls_passed"] is True
        assert result["reduction"]["selected"]
        assert certificate.replay_grade == "BITWISE"
        assert certificate.causal_claim_grade.value == "NECESSARY_IN_CONTEXT"
        assert certificate.adapter.id == HUGGINGFACE_ADAPTER_ID
