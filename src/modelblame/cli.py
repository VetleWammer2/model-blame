"""Command-line interface for recorded training and causal replay."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from html import escape
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from modelblame import __version__

app = typer.Typer(
    name="modelblame",
    help="Git bisect, blame, and revert for learned behavior.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"modelblame {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed ModelBlame version and exit.",
        ),
    ] = False,
) -> None:
    """Record trajectories and verify counterfactual training interventions."""


def _die(error: Exception | str) -> NoReturn:
    message = str(error).strip() or type(error).__name__
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _field(label: str, value: Any) -> None:
    typer.echo(f"  {label:<26} {value}")


def _method_list(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not methods:
        raise ValueError("at least one attribution method is required")
    if len(methods) != len(set(methods)):
        raise ValueError("attribution methods must not be repeated")
    return methods


def _checkpoint_selection(value: str, available: Iterable[int]) -> set[int]:
    known = set(available)
    if value.strip().lower() == "all":
        return known
    try:
        requested = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as error:
        raise ValueError(
            "--checkpoints must be 'all' or a comma-separated list of steps"
        ) from error
    if not requested:
        raise ValueError(
            "--checkpoints must be 'all' or a comma-separated list of steps"
        )
    missing = requested - known
    if missing:
        raise ValueError(f"no recorded checkpoint at steps: {sorted(missing)}")
    return requested


def _resolved(path: Path, *, must_exist: bool = False) -> Path:
    return path.expanduser().resolve(strict=must_exist)


def _require_output_outside(
    output: Path, protected: Path, *, protected_name: str = "source run"
) -> None:
    try:
        output.relative_to(protected)
    except ValueError:
        return
    raise ValueError(f"output must be outside the immutable {protected_name}")


def _atomic_text(path: Path, value: str, *, force: bool = False) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _execute_isolated_replay(
    *,
    run: Path,
    patch: Path,
    behavior: Path,
    output: Path,
    timeout_seconds: float,
    device: str,
    unseal_holdout: bool,
) -> tuple[dict[str, Any], Any]:
    from modelblame.replay.process import run_isolated

    arguments = [
        sys.executable,
        "-m",
        "modelblame.replay.worker",
        "--run",
        str(run),
        "--patch",
        str(patch),
        "--behavior",
        str(behavior),
        "--output",
        str(output),
        "--device",
        device,
    ]
    if unseal_holdout:
        arguments.append("--unseal-holdout")
    process = run_isolated(
        arguments,
        output_directory=output,
        timeout_seconds=timeout_seconds,
    )
    if process.status == "TIMEOUT":
        return {"schema_version": 1, "status": "TIMEOUT"}, process
    result_path = output / "result.json"
    if not result_path.is_file():
        diagnostic = process.stderr.strip() or "worker did not write result.json"
        raise RuntimeError(f"isolated replay failed: {diagnostic}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("isolated replay result is not valid UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise ValueError("isolated replay result must be a JSON object")
    return result, process


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _verify_score_result(
    name: str, recorded: Mapping[str, Any], replayed: Mapping[str, Any]
) -> None:
    for key in ("state", "split"):
        if recorded.get(key) != replayed.get(key):
            raise ValueError(f"re-executed {name} {key} does not match certificate")
    if not _same_float(recorded.get("score"), replayed.get("score")):
        raise ValueError(f"re-executed {name} score does not match certificate")
    expected_items = recorded.get("prompt_scores")
    actual_items = replayed.get("prompt_scores")
    if not isinstance(expected_items, list) or not isinstance(actual_items, list):
        raise ValueError(f"re-executed {name} prompt scores are malformed")
    if len(expected_items) != len(actual_items) or any(
        not _same_float(left, right)
        for left, right in zip(expected_items, actual_items, strict=True)
    ):
        raise ValueError(f"re-executed {name} prompt scores do not match certificate")


def _verify_reexecution(certificate: Any, result: Mapping[str, Any]) -> None:
    identities = {
        "run_id": certificate.source_run.id,
        "run_hash": certificate.source_run.hash,
        "behavior_contract_hash": certificate.behavior_contract_hash,
        "patch_hash": certificate.patch_hash,
        "counterfactual_checkpoint_hash": certificate.counterfactual_checkpoint_hash,
    }
    for key, expected in identities.items():
        if result.get(key) != expected:
            raise ValueError(f"re-executed {key} does not match certificate")
    _verify_score_result(
        "original behavior",
        certificate.original_behavior_result.model_dump(mode="json"),
        result.get("original_behavior", {}),
    )
    _verify_score_result(
        "counterfactual behavior",
        certificate.counterfactual_behavior_result.model_dump(mode="json"),
        result.get("counterfactual_behavior", {}),
    )
    holdout = result.get("holdout")
    if not isinstance(holdout, Mapping) or holdout.get("status") != (
        certificate.sealed_holdout_result.get("status")
    ):
        raise ValueError("re-executed sealed holdout status does not match certificate")
    if "effect" in certificate.sealed_holdout_result and not _same_float(
        holdout.get("effect"), certificate.sealed_holdout_result.get("effect")
    ):
        raise ValueError("re-executed sealed holdout effect does not match certificate")
    replayed_controls = result.get("controls")
    if not isinstance(replayed_controls, list):
        raise ValueError("re-executed controls are malformed")
    expected_controls = {item.id: item for item in certificate.control_results}
    actual_controls = {
        str(item.get("id")): item
        for item in replayed_controls
        if isinstance(item, dict)
    }
    if set(actual_controls) != set(expected_controls):
        raise ValueError("re-executed control identities do not match certificate")
    for identifier, expected in expected_controls.items():
        actual = actual_controls[identifier]
        if bool(actual.get("passed")) is not expected.passed:
            raise ValueError(f"re-executed control {identifier!r} status differs")
        for key in ("mean_drift", "max_item_drift"):
            if not _same_float(actual.get(key), getattr(expected, key)):
                raise ValueError(f"re-executed control {identifier!r} {key} differs")
        for key in ("original_scores", "counterfactual_scores"):
            actual_scores = actual.get(key)
            expected_scores = getattr(expected, key)
            if (
                not isinstance(actual_scores, list)
                or len(actual_scores) != len(expected_scores)
                or any(
                    not _same_float(actual_score, expected_score)
                    for actual_score, expected_score in zip(
                        actual_scores, expected_scores, strict=True
                    )
                )
            ):
                raise ValueError(f"re-executed control {identifier!r} {key} differs")


@app.command("init")
def init_command(
    destination: Annotated[
        Path, typer.Argument(help="New experiment directory to create.")
    ],
) -> None:
    """Create a validated experiment template without result artifacts."""

    try:
        from modelblame.templates import initialize_experiment

        created = initialize_experiment(destination)
    except Exception as error:
        _die(error)
    typer.echo("Experiment template created:")
    _field("path", created)
    _field("configuration", created / "experiment.toml")
    _field("behavior contract", created / "behavior.yaml")
    _field("training data", "not created; add data/train.jsonl")


@app.command("train")
def train_command(
    experiment: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Experiment TOML or JSON file.",
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Directory in which to create the run.")
    ] = Path("runs"),
    deterministic: Annotated[
        str | None,
        typer.Option(
            "--deterministic",
            help="Override config with strict, best-effort, or off.",
        ),
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="cpu, auto, cuda, or cuda:<index>.")
    ] = None,
) -> None:
    """Train a model and record a replay-complete occurrence ledger."""

    if deterministic is not None and deterministic not in {
        "strict",
        "best-effort",
        "off",
    }:
        _die("--deterministic must be strict, best-effort, or off")
    try:
        from modelblame.training.loop import train_experiment

        result = train_experiment(
            experiment,
            _resolved(output),
            deterministic=deterministic,
            device=device,
        )
    except Exception as error:
        _die(error)
    typer.echo("Run complete:")
    _field("run ID", result.run_id)
    _field("training steps", result.training_steps)
    _field("example occurrences", result.example_occurrences)
    _field("final checkpoint", result.final_checkpoint)
    _field("run directory", result.run_path)


@app.command("inspect")
def inspect_command(
    run: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Recorded run directory.",
        ),
    ],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the inspection record as JSON.")
    ] = False,
) -> None:
    """Validate and summarize a recorded run."""

    try:
        from modelblame.recorded import RecordedRun

        result = RecordedRun.open(run).inspect()
    except Exception as error:
        _die(error)
    if json_output:
        typer.echo(_json(result))
        return
    typer.echo("Recorded run:")
    for label, key in (
        ("run ID", "run_id"),
        ("run hash", "run_hash"),
        ("adapter", "adapter_id"),
        ("dataset fingerprint", "dataset_fingerprint"),
        ("tokenizer fingerprint", "tokenizer_fingerprint"),
        ("training steps", "training_steps"),
        ("example occurrences", "example_occurrences"),
        ("checkpoints", "checkpoint_count"),
        ("checkpoint coverage", "checkpoint_coverage_complete"),
        ("replay grade", "replay_grade"),
    ):
        _field(label, result[key])
    determinism = result.get("determinism", {})
    mode = (
        determinism.get("mode", determinism)
        if isinstance(determinism, dict)
        else determinism
    )
    _field("determinism", mode)
    _field("attribution methods", ", ".join(result["supported_attribution_methods"]))
    _field("interventions", ", ".join(result["supported_interventions"]))


@app.command("audit")
def audit_command(
    run: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    from_step: Annotated[int | None, typer.Option("--from-step", min=0)] = None,
    to_step: Annotated[int | None, typer.Option("--to-step", min=1)] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    atol: Annotated[float, typer.Option("--atol", min=0.0)] = 1e-6,
    rtol: Annotated[float, typer.Option("--rtol", min=0.0)] = 1e-5,
    behavior: Annotated[
        Path | None,
        typer.Option(
            "--behavior",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional search-only behavior contract to compare during replay.",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Replay an unchanged checkpoint interval and grade equivalence."""

    try:
        from modelblame.replay.audit import audit_run

        result = audit_run(
            run,
            from_step=from_step,
            to_step=to_step,
            device=device,
            atol=atol,
            rtol=rtol,
            write_artifact=True,
            behavior_path=behavior,
        )
        value = result.to_dict()
    except Exception as error:
        _die(error)
    if json_output:
        typer.echo(_json(value))
        return
    typer.echo("Replay audit:")
    _field("checkpoint interval", f"{result.from_step} -> {result.to_step}")
    for component in result.components:
        equality = (
            "bitwise"
            if component.bitwise_equal
            else ("numeric" if component.numeric_equal else "different")
        )
        _field(f"{component.component} state", equality)
    _field("logged losses", "equal" if result.recorded_losses_equal else "different")
    _field("output hashes", "equal" if result.output_hashes_equal else "different")
    if result.behavior_scores:
        _field(
            "behavior score",
            "equal" if result.behavior_scores.get("equal") else "different",
        )
    _field("replay grade", result.replay_grade.value)
    if result.replay_grade.value == "FAILED":
        raise typer.Exit(code=1)


@app.command("test")
def test_command(
    run: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    behavior: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    checkpoints: Annotated[
        str,
        typer.Option(
            "--checkpoints", help="'all' or a comma-separated list of checkpoint steps."
        ),
    ] = "all",
    device: Annotated[str, typer.Option("--device")] = "cpu",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Evaluate search probes at recorded checkpoints; keep holdout sealed."""

    try:
        from modelblame.timeline.evaluate import evaluate_timeline

        timeline = evaluate_timeline(run, behavior, device=device, write_artifact=True)
        points = timeline["points"]
        selected = _checkpoint_selection(
            checkpoints, (int(point["step"]) for point in points)
        )
        filtered = [point for point in points if int(point["step"]) in selected]
        value = {
            "schema_version": 1,
            "run_id": timeline["run_id"],
            "behavior_contract_hash": timeline["behavior_contract_hash"],
            "points": filtered,
            "holdout": {"status": "SEALED"},
        }
    except Exception as error:
        _die(error)
    if json_output:
        typer.echo(_json(value))
        return
    typer.echo("Behavior checkpoints:")
    _field("run ID", timeline["run_id"])
    _field("contract hash", timeline["behavior_contract_hash"])
    for point in filtered:
        target = point["target"]
        _field(
            f"step {int(point['step']):06d}",
            f"score={float(target['score']):.6g} state={target['state']}",
        )
    _field("holdout", "SEALED")


@app.command("bisect")
def bisect_command(
    run: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    behavior: Annotated[
        Path,
        typer.Option(
            "--behavior",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    device: Annotated[str, typer.Option("--device")] = "cpu",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report every observed behavior-transition window."""

    try:
        from modelblame.timeline.evaluate import evaluate_timeline

        timeline = evaluate_timeline(run, behavior, device=device, write_artifact=True)
    except Exception as error:
        _die(error)
    if json_output:
        typer.echo(_json(timeline))
        return
    transitions = timeline["transitions"]
    typer.echo("Behavior transitions:")
    _field("contract hash", timeline["behavior_contract_hash"])
    _field("windows", len(transitions))
    if not transitions:
        _field("result", "no state transition observed at recorded checkpoints")
    for index, transition in enumerate(transitions, start=1):
        description = (
            f"{transition['kind']} {transition['start_step']}"
            f" -> {transition['end_step']}"
        )
        _field(
            f"window {index}",
            description,
        )
        _field("occurrences", len(transition.get("occurrence_ids", [])))
        _field("sources", ", ".join(transition.get("sources", [])) or "none")


@app.command("index")
def index_command(
    run: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    behavior: Annotated[
        Path,
        typer.Option(
            "--behavior",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    methods: Annotated[
        str, typer.Option("--methods", help="Comma-separated attribution methods.")
    ] = "trajectory-sketch,tracin-cp,bm25",
    projection_dimension: Annotated[
        int, typer.Option("--projection-dimension", min=1)
    ] = 128,
    projection_seed: Annotated[int, typer.Option("--projection-seed", min=0)] = 0,
    checkpoint_limit: Annotated[int, typer.Option("--checkpoint-limit", min=1)] = 2,
    parameters: Annotated[
        str,
        typer.Option(
            "--parameters", help="Comma-separated trainable parameter-name patterns."
        ),
    ] = "lm_head",
    device: Annotated[str, typer.Option("--device")] = "cpu",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build content-addressed candidate-attribution indexes."""

    try:
        from modelblame.attribution.indexing import build_indexes

        requested = _method_list(methods)
        parameter_patterns = tuple(
            item.strip() for item in parameters.split(",") if item.strip()
        )
        if not parameter_patterns:
            raise ValueError("at least one parameter pattern is required")
        indexes = build_indexes(
            run,
            behavior,
            methods=requested,
            projection_dimension=projection_dimension,
            projection_seed=projection_seed,
            checkpoint_limit=checkpoint_limit,
            parameter_patterns=parameter_patterns,
            device=device,
        )
        value = {
            method: {
                "content_hash": index.content_hash,
                "candidate_count": len(index.scores),
                "method_config": index.method_config,
                "projection_seed": index.projection_seed,
            }
            for method, index in sorted(indexes.items())
        }
    except Exception as error:
        _die(error)
    if json_output:
        typer.echo(_json(value))
        return
    typer.echo("Candidate indexes:")
    for method, item in value.items():
        _field(method, f"{item['candidate_count']} candidates")
        _field("content hash", item["content_hash"])


@app.command("blame")
def blame_command(
    run: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    behavior: Annotated[
        Path,
        typer.Option(
            "--behavior",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    candidate_limit: Annotated[
        int, typer.Option("--candidate-limit", min=1, max=1_000_000)
    ] = 256,
    replay_budget: Annotated[int, typer.Option("--replay-budget", min=1)] = 64,
    workers: Annotated[int, typer.Option("--workers", min=1, max=64)] = 1,
    output: Annotated[
        Path, typer.Option("--output", help="New evidence-bundle directory.")
    ] = Path("blame-bundle"),
    methods: Annotated[
        str, typer.Option("--methods", help="Comma-separated attribution methods.")
    ] = "temporal,bm25,tracin-cp,trajectory-sketch",
    timeout_seconds: Annotated[
        float, typer.Option("--timeout", min=0.1, max=604800.0)
    ] = 600.0,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    include_example_text: Annotated[
        bool,
        typer.Option(
            "--include-example-text",
            help=(
                "Include retrieved candidates' raw prompt/completion text in the "
                "bundle."
            ),
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Propose candidates, execute interventions, and reduce a causal set."""

    try:
        from modelblame.blame import run_blame

        destination, result = run_blame(
            run,
            behavior,
            output=_resolved(output),
            candidate_limit=candidate_limit,
            replay_budget=replay_budget,
            workers=workers,
            timeout_seconds=timeout_seconds,
            methods=_method_list(methods),
            device=device,
            include_example_text=include_example_text,
        )
    except Exception as error:
        _die(error)
    if json_output:
        typer.echo(_json({"bundle": str(destination), "result": result}))
    reduction = result.get("reduction", result)
    final_result = result.get("final_result")
    accepted = (
        bool(final_result.get("accepted"))
        if isinstance(final_result, dict)
        else bool(reduction.get("accepted", False))
    )
    if not json_output:
        typer.echo("Causal search complete:")
        _field("accepted", accepted)
        _field("minimality", reduction.get("minimality_grade", "NO_ACCEPTED_PATCH"))
        _field("selected occurrences", len(reduction.get("selected", [])))
        _field("replays executed", len(reduction.get("experiments", [])))
        _field("evidence bundle", destination)
    if not accepted:
        raise typer.Exit(code=1)


@app.command("replay")
def replay_command(
    run: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    patch: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    output: Annotated[
        Path, typer.Option("--output", help="New isolated replay directory.")
    ] = Path("counterfactual"),
    behavior: Annotated[
        Path | None,
        typer.Option(
            "--behavior",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Contract file; defaults to behavior.yaml beside the patch.",
        ),
    ] = None,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout", min=0.1, max=604800.0)
    ] = 600.0,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Execute a patch in a fresh process without opening the sealed holdout."""

    try:
        run_path = _resolved(run, must_exist=True)
        patch_path = _resolved(patch, must_exist=True)
        behavior_path = (
            _resolved(behavior, must_exist=True)
            if behavior is not None
            else (patch_path.parent / "behavior.yaml").resolve(strict=True)
        )
        output_path = _resolved(output)
        _require_output_outside(output_path, run_path)
        result, process = _execute_isolated_replay(
            run=run_path,
            patch=patch_path,
            behavior=behavior_path,
            output=output_path,
            timeout_seconds=timeout_seconds,
            device=device,
            unseal_holdout=False,
        )
    except Exception as error:
        _die(error)
    if json_output:
        typer.echo(_json(result))
    else:
        typer.echo("Counterfactual replay:")
        _field("status", result.get("status", "INCONCLUSIVE"))
        _field("search result", result.get("search_status", "unavailable"))
        _field("holdout", result.get("holdout", {}).get("status", "SEALED"))
        _field("controls passed", result.get("controls_passed", False))
        _field("target effect", result.get("target_effect", "unavailable"))
        _field("output", output_path)
    if result.get("status") in {
        "TIMEOUT",
        "INCONCLUSIVE",
    } or process.returncode not in {
        0,
        None,
    }:
        raise typer.Exit(code=2)


@app.command("verify")
def verify_command(
    bundle: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    run: Annotated[
        Path | None,
        typer.Option(
            "--run",
            exists=True,
            file_okay=False,
            readable=True,
            resolve_path=True,
            help="Source run; when supplied, re-execute the final patch and holdout.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Retain re-execution artifacts here."),
    ] = None,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout", min=0.1, max=604800.0)
    ] = 600.0,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Verify bundle hashes, and optionally re-execute its causal evidence."""

    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        from modelblame.evidence.verify import verify_bundle
        from modelblame.recorded import RecordedRun

        bundle_path = _resolved(bundle, must_exist=True)
        certificate = verify_bundle(bundle_path)
        state = "STATIC_VERIFIED"
        retained_output: Path | None = None
        if run is not None:
            run_path = _resolved(run, must_exist=True)
            source = RecordedRun.open(run_path, verify_checkpoints=True)
            if (
                source.run_id != certificate.source_run.id
                or source.run_hash != certificate.source_run.hash
            ):
                raise ValueError("source run identity does not match certificate")
            if output is None:
                temporary = tempfile.TemporaryDirectory(prefix="modelblame-verify-")
                replay_output = Path(temporary.name).resolve()
            else:
                replay_output = _resolved(output)
                _require_output_outside(replay_output, run_path)
                _require_output_outside(
                    replay_output, bundle_path, protected_name="evidence bundle"
                )
                retained_output = replay_output
            result, process = _execute_isolated_replay(
                run=run_path,
                patch=bundle_path / "patch.json",
                behavior=bundle_path / "behavior.yaml",
                output=replay_output,
                timeout_seconds=timeout_seconds,
                device=device,
                unseal_holdout=True,
            )
            if process.returncode != 0 or result.get("status") in {
                "TIMEOUT",
                "INCONCLUSIVE",
            }:
                raise ValueError(
                    f"evidence re-execution did not complete: {result.get('status')}"
                )
            _verify_reexecution(certificate, result)
            state = "REEXECUTED_VERIFIED"
        response = {
            "status": state,
            "bundle": str(bundle_path),
            "source_run_id": certificate.source_run.id,
            "patch_hash": certificate.patch_hash,
            "causal_claim": certificate.causal_claim_grade.value,
            "replay_grade": certificate.replay_grade,
            "reexecution_output": (
                None if retained_output is None else str(retained_output)
            ),
        }
    except Exception as error:
        _die(error)
    finally:
        if temporary is not None:
            temporary.cleanup()
    if json_output:
        typer.echo(_json(response))
        return
    typer.echo("Evidence verification:")
    _field("status", response["status"])
    _field("source run ID", response["source_run_id"])
    _field("patch hash", response["patch_hash"])
    _field("causal claim", response["causal_claim"])
    _field("replay grade", response["replay_grade"])
    if response["status"] == "STATIC_VERIFIED":
        _field("causal re-execution", "not run; supply --run RUN")
    elif response["reexecution_output"] is not None:
        _field("re-execution output", response["reexecution_output"])


@app.command("report")
def report_command(
    bundle: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True, resolve_path=True),
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Markdown report path.")
    ] = Path("report.md"),
    html: Annotated[
        bool, typer.Option("--html", help="Also write a static HTML report.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Replace existing report files atomically.")
    ] = False,
) -> None:
    """Generate a deterministic report from a verified evidence certificate."""

    try:
        from modelblame.evidence.verify import verify_bundle
        from modelblame.report.markdown import render_report

        bundle_path = _resolved(bundle, must_exist=True)
        certificate = verify_bundle(bundle_path)
        markdown = render_report(certificate)
        output_path = _resolved(output)
        _require_output_outside(
            output_path, bundle_path, protected_name="evidence bundle"
        )
        html_path = output_path.with_suffix(".html") if html else None
        if not force:
            existing = [
                path
                for path in (output_path, html_path)
                if path is not None and path.exists()
            ]
            if existing:
                raise FileExistsError(
                    f"refusing to overwrite existing file: {existing[0]}"
                )
        _atomic_text(output_path, markdown, force=force)
        if html_path is not None:
            document = (
                '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                "<title>ModelBlame evidence report</title></head><body>"
                f"<pre>{escape(markdown)}</pre></body></html>\n"
            )
            _atomic_text(html_path, document, force=force)
    except Exception as error:
        _die(error)
    typer.echo("Evidence report generated:")
    _field("Markdown", output_path)
    if html_path is not None:
        _field("HTML", html_path)


if __name__ == "__main__":
    app()
