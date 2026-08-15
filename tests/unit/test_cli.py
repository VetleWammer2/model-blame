from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from modelblame.cli import app
from modelblame.config.behavior import load_behavior_contract
from modelblame.config.experiment import load_experiment_config

runner = CliRunner()


def test_help_registers_the_complete_command_surface() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "init",
        "train",
        "inspect",
        "audit",
        "test",
        "bisect",
        "index",
        "blame",
        "replay",
        "verify",
        "report",
    ):
        assert command in result.stdout


def test_init_creates_valid_result_free_templates(tmp_path: Path) -> None:
    destination = tmp_path / "my-experiment"

    result = runner.invoke(app, ["init", str(destination)])

    assert result.exit_code == 0, result.output
    assert {item.name for item in destination.iterdir()} == {
        "README.md",
        "behavior.yaml",
        "controls.yaml",
        "data",
        "experiment.toml",
    }
    assert not any(destination.rglob("*.safetensors"))
    assert not (destination / "manifest.json").exists()
    assert load_experiment_config(destination / "experiment.toml").name == (
        "my-experiment"
    )
    assert load_behavior_contract(destination / "behavior.yaml").holdout.sealed
    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "controls embedded in `behavior.yaml`" in readme


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()

    result = runner.invoke(app, ["init", str(destination)])

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output


def test_inspect_json_is_machine_readable(tmp_path: Path, monkeypatch: object) -> None:
    import modelblame.recorded

    run = tmp_path / "run"
    run.mkdir()
    inspection = {
        "run_id": "mb_test",
        "run_hash": "a" * 64,
        "adapter_id": "modelblame.tiny-causal-lm.v1",
        "model_config": {},
        "dataset_fingerprint": "b" * 64,
        "tokenizer_fingerprint": "c" * 64,
        "training_steps": 2,
        "example_occurrences": 4,
        "checkpoint_count": 2,
        "checkpoint_coverage_complete": True,
        "determinism": {"mode": "strict"},
        "replay_grade": "BITWISE",
        "supported_attribution_methods": ["bm25"],
        "supported_interventions": ["GRADIENT_ABLATE"],
    }
    monkeypatch.setattr(  # type: ignore[attr-defined]
        modelblame.recorded.RecordedRun,
        "open",
        lambda path: SimpleNamespace(inspect=lambda: inspection),
    )

    result = runner.invoke(app, ["inspect", str(run), "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == inspection


def test_replay_keeps_holdout_sealed(tmp_path: Path, monkeypatch: object) -> None:
    import modelblame.cli

    run = tmp_path / "run"
    run.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    patch = bundle / "patch.json"
    patch.write_text("{}", encoding="utf-8")
    (bundle / "behavior.yaml").write_text("{}", encoding="utf-8")
    output = tmp_path / "replay-output"
    captured: dict[str, object] = {}

    def fake_replay(**kwargs: object) -> tuple[dict[str, object], object]:
        captured.update(kwargs)
        output.mkdir()
        return (
            {
                "status": "TARGET_PASSED",
                "search_status": "PASSED",
                "holdout": {"status": "SEALED"},
                "controls_passed": True,
                "target_effect": 1.0,
            },
            SimpleNamespace(returncode=0),
        )

    monkeypatch.setattr(  # type: ignore[attr-defined]
        modelblame.cli, "_execute_isolated_replay", fake_replay
    )

    result = runner.invoke(
        app,
        ["replay", str(run), str(patch), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert captured["unseal_holdout"] is False
    assert "SEALED" in result.stdout


def test_verify_without_run_reports_static_scope(
    tmp_path: Path, monkeypatch: object
) -> None:
    import modelblame.evidence.verify

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    certificate = SimpleNamespace(
        source_run=SimpleNamespace(id="mb_test", hash="a" * 64),
        patch_hash="b" * 64,
        causal_claim_grade=SimpleNamespace(value="NECESSARY_IN_CONTEXT"),
        replay_grade="BITWISE",
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        modelblame.evidence.verify,
        "verify_bundle",
        lambda path: certificate,
    )

    result = runner.invoke(app, ["verify", str(bundle)])

    assert result.exit_code == 0, result.output
    assert "STATIC_VERIFIED" in result.stdout
    assert "not run; supply --run RUN" in result.stdout


def test_blame_forwards_explicit_example_text_opt_in(
    tmp_path: Path, monkeypatch: object
) -> None:
    import modelblame.blame

    run = tmp_path / "run"
    run.mkdir()
    behavior = tmp_path / "behavior.yaml"
    behavior.write_text("schema_version: 1\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_blame(*args: object, **kwargs: object) -> tuple[Path, dict[str, object]]:
        captured.update(kwargs)
        return tmp_path / "bundle", {
            "reduction": {
                "accepted": True,
                "minimality_grade": "ONE_MINIMAL",
                "selected": ["occurrence"],
                "experiments": [],
            },
            "final_result": {"accepted": True},
        }

    monkeypatch.setattr(modelblame.blame, "run_blame", fake_blame)  # type: ignore[attr-defined]
    result = runner.invoke(
        app,
        [
            "blame",
            str(run),
            "--behavior",
            str(behavior),
            "--include-example-text",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["include_example_text"] is True
