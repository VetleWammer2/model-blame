"""Independent hash and semantic verification of an evidence bundle."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
import yaml
from safetensors import safe_open
from safetensors.torch import load_file

from modelblame.behavior.contract import load_contract_artifact
from modelblame.checkpoint.hashing import canonical_json_hash, hash_tensors
from modelblame.data.identity import EXAMPLE_ID_PATTERN, OCCURRENCE_ID_PATTERN
from modelblame.evidence.certificate import EvidenceCertificate
from modelblame.patch.schema import Patch, parse_patch
from modelblame.reducer.minimality import MinimalityGrade
from modelblame.util.canonical_json import validate_json_tree

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARTIFACTS = 100_000
MAX_CANDIDATE_ROWS = 10_000_000
MAX_EXAMPLE_TEXT_BYTES = 256 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXAMPLE_TEXT_ARTIFACT = "candidate-example-text.parquet"
MANDATORY_ARTIFACTS = frozenset(
    {
        "behavior.yaml",
        "controls.yaml",
        "candidate-ranking.parquet",
        "experiments.parquet",
        "patch.json",
        "blame.json",
        "certificate.json",
        "replay.py",
        "report.md",
        "figures/behavior-timeline.svg",
        "figures/candidate-effects.svg",
        "figures/reduction.svg",
        "figures/control-drift.svg",
        "counterfactual/model.safetensors",
        "counterfactual/checkpoint-manifest.json",
    }
)


class VerificationError(ValueError):
    """An evidence bundle failed structural, hash, or semantic verification."""


def sha256_file(path: Path, *, max_bytes: int = 4 * 1024 * 1024 * 1024) -> str:
    if not path.is_file():
        raise VerificationError(f"artifact is missing: {path.name}")
    if path.stat().st_size > max_bytes:
        raise VerificationError(
            f"artifact exceeds verification size limit: {path.name}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_object_hash(value: Any, *, omit: frozenset[str] = frozenset()) -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key not in omit}
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_artifact(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise VerificationError(f"unsafe artifact path: {relative!r}")
    destination = root / Path(*posix.parts)
    if destination.is_symlink():
        raise VerificationError(f"bundle artifacts cannot be symlinks: {relative!r}")
    destination = destination.resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise VerificationError(
            f"artifact escapes bundle root: {relative!r}"
        ) from error
    return destination


def _json_object(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise VerificationError(f"JSON artifact missing or oversized: {path.name}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationError(f"duplicate JSON key in {path.name}: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                VerificationError(f"non-finite JSON value in {path.name}: {item}")
            ),
        )
        validate_json_tree(value, max_depth=32, max_items=2_000_000)
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"JSON artifact must be an object: {path.name}")
    return value


def _same_value(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise VerificationError(f"cross-artifact disagreement: {name}")


def verify_patch(patch: dict[str, Any], *, expected_hash: str) -> None:
    claimed = patch.get("patch_hash")
    computed = canonical_object_hash(patch, omit=frozenset({"patch_hash"}))
    if claimed != expected_hash or computed != expected_hash:
        raise VerificationError("patch hash mismatch")


def _verify_behavior_and_controls(
    root: Path, certificate: EvidenceCertificate
) -> dict[str, Any]:
    try:
        _, materialized, contract_hash = load_contract_artifact(root / "behavior.yaml")
    except Exception as error:
        raise VerificationError(
            f"invalid bundled behavior contract: {error}"
        ) from error
    _same_value(
        "behavior contract hash", contract_hash, certificate.behavior_contract_hash
    )
    try:
        controls_path = root / "controls.yaml"
        if controls_path.stat().st_size > MAX_JSON_BYTES:
            raise VerificationError("controls artifact exceeds the size limit")
        controls_artifact = yaml.safe_load(controls_path.read_text(encoding="utf-8"))
        validate_json_tree(controls_artifact, max_depth=16, max_items=1_000_000)
    except Exception as error:
        raise VerificationError(f"invalid controls artifact: {error}") from error
    expected_controls = {
        "schema_version": 1,
        "controls": materialized.get("controls", []),
    }
    _same_value("controls artifact", controls_artifact, expected_controls)
    control_hashes = [
        canonical_json_hash(control) for control in expected_controls["controls"]
    ]
    _same_value(
        "control contract hashes", control_hashes, certificate.control_contract_hashes
    )
    return materialized


def _candidate_occurrences(root: Path, certificate: EvidenceCertificate) -> set[str]:
    candidate_path = root / "candidate-ranking.parquet"
    try:
        metadata = pq.ParquetFile(candidate_path).metadata
        if metadata.num_rows > MAX_CANDIDATE_ROWS:
            raise VerificationError("candidate ranking exceeds the row limit")
        table = pq.read_table(
            candidate_path,
            columns=["occurrence_id", "method_id", "rank"],
        )
        rows = table.to_pylist()
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"invalid candidate ranking: {error}") from error
    methods: set[str] = set()
    occurrences: set[str] = set()
    ranks: dict[str, set[int]] = {}
    for row in rows:
        occurrence_id = row.get("occurrence_id")
        method_id = row.get("method_id")
        rank = row.get("rank")
        if (
            not isinstance(occurrence_id, str)
            or OCCURRENCE_ID_PATTERN.fullmatch(occurrence_id) is None
            or not isinstance(method_id, str)
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
        ):
            raise VerificationError("candidate ranking contains an invalid row")
        method_ranks = ranks.setdefault(method_id, set())
        if rank in method_ranks:
            raise VerificationError("candidate ranking repeats a method rank")
        method_ranks.add(rank)
        methods.add(method_id)
        occurrences.add(occurrence_id)
    if not rows:
        raise VerificationError("candidate ranking is empty")
    for method_id, method_ranks in ranks.items():
        if method_ranks != set(range(1, len(method_ranks) + 1)):
            raise VerificationError(
                f"candidate ranks are not contiguous for method {method_id!r}"
            )
    _same_value("candidate methods", methods, set(certificate.candidate_methods))
    return occurrences


def _verify_candidate_example_text(
    root: Path,
    *,
    candidate_occurrences: set[str],
    expected_rows: int,
) -> None:
    path = root / EXAMPLE_TEXT_ARTIFACT
    if path.stat().st_size > MAX_EXAMPLE_TEXT_BYTES:
        raise VerificationError("candidate example text exceeds the size limit")
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows > MAX_CANDIDATE_ROWS:
            raise VerificationError("candidate example text exceeds the row limit")
        expected_columns = {"occurrence_id", "example_id", "prompt", "completion"}
        if set(parquet.schema_arrow.names) != expected_columns:
            raise VerificationError("candidate example text has unexpected columns")
        rows = pq.read_table(path).to_pylist()
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"invalid candidate example text: {error}") from error
    if len(rows) != expected_rows:
        raise VerificationError("candidate example text count differs from retrieval")
    seen: set[str] = set()
    for row in rows:
        occurrence_id = row.get("occurrence_id")
        example_id = row.get("example_id")
        prompt = row.get("prompt")
        completion = row.get("completion")
        if (
            not isinstance(occurrence_id, str)
            or occurrence_id not in candidate_occurrences
            or OCCURRENCE_ID_PATTERN.fullmatch(occurrence_id) is None
            or occurrence_id in seen
            or not isinstance(example_id, str)
            or EXAMPLE_ID_PATTERN.fullmatch(example_id) is None
            or not isinstance(prompt, str)
            or not isinstance(completion, str)
            or len(prompt) > 1_000_000
            or len(completion) > 1_000_000
        ):
            raise VerificationError("candidate example text contains an invalid row")
        seen.add(occurrence_id)


def _verify_counterfactual_model(root: Path, certificate: EvidenceCertificate) -> None:
    checkpoint_manifest = _json_object(
        root / "counterfactual" / "checkpoint-manifest.json"
    )
    if checkpoint_manifest.get("schema_version") != 1:
        raise VerificationError("unsupported counterfactual checkpoint manifest")
    _same_value(
        "counterfactual tokenizer fingerprint",
        checkpoint_manifest.get("tokenizer_fingerprint"),
        certificate.tokenizer_fingerprint,
    )
    model_declaration = checkpoint_manifest.get("model")
    if not isinstance(model_declaration, Mapping):
        raise VerificationError("counterfactual checkpoint has no model declaration")
    model_path = root / "counterfactual" / "model.safetensors"
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            file_metadata = handle.metadata() or {}
        declared_tensors = model_declaration.get("tensors")
        if not isinstance(declared_tensors, Mapping) or keys != set(declared_tensors):
            raise VerificationError(
                "counterfactual model tensor names differ from its manifest"
            )
        tensors = load_file(model_path, device="cpu")
        for name, declaration in declared_tensors.items():
            if not isinstance(declaration, Mapping):
                raise VerificationError("invalid counterfactual tensor declaration")
            tensor = tensors[name]
            if list(tensor.shape) != declaration.get("shape") or str(
                tensor.dtype
            ) != declaration.get("dtype"):
                raise VerificationError(
                    f"counterfactual tensor shape/dtype mismatch: {name}"
                )
        state_hash = hash_tensors(tensors)
        if state_hash != model_declaration.get(
            "state_hash"
        ) or state_hash != file_metadata.get("state_hash"):
            raise VerificationError("counterfactual model state hash mismatch")
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(
            f"invalid counterfactual SafeTensors: {error}"
        ) from error


def _control_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    names = {
        "id",
        "original_scores",
        "counterfactual_scores",
        "original_score",
        "counterfactual_score",
        "mean_drift",
        "max_item_drift",
        "max_mean_drift",
        "max_allowed_item_drift",
        "passed",
    }
    return {name: value.get(name) for name in names}


def _score_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.get(name) for name in ("score", "prompt_scores", "state", "split")
    }


def _verify_experiments(root: Path, reduction: Mapping[str, Any]) -> None:
    expected = reduction.get("experiments")
    if not isinstance(expected, list):
        raise VerificationError("blame reduction has no experiment ledger")
    try:
        parquet_file = pq.ParquetFile(root / "experiments.parquet")
        if parquet_file.metadata.num_rows > MAX_CANDIDATE_ROWS:
            raise VerificationError("experiment ledger exceeds the row limit")
        rows = pq.read_table(root / "experiments.parquet").to_pylist()
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(f"invalid experiment ledger: {error}") from error
    flattened: list[dict[str, Any]] = []
    for item in expected:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("observation"), Mapping
        ):
            raise VerificationError("invalid reduction experiment entry")
        flattened.append(
            {"occurrence_ids": list(item.get("occurrence_ids", []))}
            | dict(item["observation"])
        )
    _same_value("experiment ledger", rows, flattened)


def _verify_blame(root: Path, certificate: EvidenceCertificate, patch: Patch) -> None:
    blame = _json_object(root / "blame.json")
    if blame.get("schema_version") != 1:
        raise VerificationError("unsupported blame artifact")
    _same_value("blame run ID", blame.get("run_id"), certificate.source_run.id)
    _same_value(
        "blame behavior hash",
        blame.get("behavior_contract_hash"),
        certificate.behavior_contract_hash,
    )
    _same_value(
        "blame candidate methods",
        blame.get("candidate_methods"),
        certificate.candidate_methods,
    )
    reduction = blame.get("reduction")
    final = blame.get("final_result")
    if not isinstance(reduction, Mapping) or not isinstance(final, Mapping):
        raise VerificationError("blame artifact is missing reduction or final replay")
    selected = reduction.get("selected")
    if (
        not isinstance(selected, list)
        or any(not isinstance(item, str) for item in selected)
        or len(selected) != len(set(selected))
    ):
        raise VerificationError("blame reduction has invalid selected occurrences")
    _same_value("selected occurrences", set(selected), set(patch.occurrence_ids))
    _same_value(
        "minimality grade",
        reduction.get("minimality_grade"),
        certificate.minimality_grade.value,
    )
    _same_value(
        "replay budget", reduction.get("replay_budget"), certificate.replay_budget
    )
    experiments = reduction.get("experiments")
    if not isinstance(experiments, list):
        raise VerificationError("blame reduction has no experiments")
    experiment_hashes: list[str] = []
    for item in experiments:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("observation"), Mapping
        ):
            raise VerificationError("invalid blame experiment")
        experiment_hash = item["observation"].get("experiment_hash")
        if not isinstance(experiment_hash, str):
            raise VerificationError("blame experiment is missing its hash")
        experiment_hashes.append(experiment_hash)
    _same_value(
        "replay experiment hashes",
        experiment_hashes,
        certificate.replay_experiment_hashes,
    )
    _same_value(
        "one-minimality tests",
        reduction.get("one_minimality_tests"),
        certificate.one_minimality_tests,
    )
    _same_value(
        "interaction diagnostics",
        reduction.get("interaction_signals"),
        certificate.interaction_diagnostics,
    )
    if certificate.minimality_grade in {
        MinimalityGrade.ONE_MINIMAL,
        MinimalityGrade.GLOBAL_MINIMUM,
    }:
        removals = {
            str(test.get("removed")): test for test in certificate.one_minimality_tests
        }
        _same_value("one-minimality coverage", set(removals), set(patch.occurrence_ids))
        selected_set = set(patch.occurrence_ids)
        for removed, test in removals.items():
            _same_value(
                f"one-minimality subset for {removed}",
                set(test.get("subset", [])),
                selected_set - {removed},
            )
    _same_value("final replay run ID", final.get("run_id"), certificate.source_run.id)
    _same_value(
        "final replay run hash", final.get("run_hash"), certificate.source_run.hash
    )
    _same_value(
        "final replay patch hash", final.get("patch_hash"), certificate.patch_hash
    )
    _same_value(
        "final replay behavior hash",
        final.get("behavior_contract_hash"),
        certificate.behavior_contract_hash,
    )
    _same_value(
        "counterfactual checkpoint hash",
        final.get("counterfactual_checkpoint_hash"),
        certificate.counterfactual_checkpoint_hash,
    )
    if final.get("source_checkpoint_hash") not in certificate.source_checkpoint_hashes:
        raise VerificationError("final replay source checkpoint is not certified")
    _same_value(
        "intervention semantics",
        final.get("intervention_semantics"),
        certificate.intervention_semantics,
    )
    _same_value(
        "original behavior result",
        _score_projection(final.get("original_behavior", {})),
        certificate.original_behavior_result.model_dump(mode="json"),
    )
    _same_value(
        "counterfactual behavior result",
        _score_projection(final.get("counterfactual_behavior", {})),
        certificate.counterfactual_behavior_result.model_dump(mode="json"),
    )
    _same_value(
        "sealed holdout result", final.get("holdout"), certificate.sealed_holdout_result
    )
    final_controls = final.get("controls")
    if not isinstance(final_controls, list) or any(
        not isinstance(item, Mapping) for item in final_controls
    ):
        raise VerificationError("final replay controls are invalid")
    _same_value(
        "control results",
        [_control_projection(item) for item in final_controls],
        [item.model_dump(mode="json") for item in certificate.control_results],
    )
    _same_value(
        "absolute effect size",
        final.get("target_effect"),
        certificate.effect_sizes.get("absolute"),
    )
    _verify_experiments(root, reduction)


def _verify_source_run(
    source_run: Path, certificate: EvidenceCertificate, patch: Patch
) -> None:
    from modelblame.recorded import RecordedRun

    try:
        run = RecordedRun.open(source_run, verify_checkpoints=True)
    except Exception as error:
        raise VerificationError(f"source run verification failed: {error}") from error
    _same_value("source run ID", run.run_id, certificate.source_run.id)
    _same_value("source run hash", run.run_hash, certificate.source_run.hash)
    _same_value(
        "source checkpoint hashes",
        [str(item["hash"]) for item in run.checkpoints],
        certificate.source_checkpoint_hashes,
    )
    _same_value(
        "dataset fingerprint",
        run.manifest.get("dataset_fingerprint"),
        certificate.dataset_fingerprint,
    )
    _same_value(
        "tokenizer fingerprint",
        run.manifest.get("tokenizer_fingerprint"),
        certificate.tokenizer_fingerprint,
    )
    _same_value("adapter ID", run.manifest.get("adapter_id"), certificate.adapter.id)
    if certificate.adapter.id == "modelblame.tiny-causal-lm.v1":
        adapter_path = Path(__file__).parents[1] / "adapters" / "tiny_causal_lm.py"
        _same_value("adapter hash", sha256_file(adapter_path), certificate.adapter.hash)
    _same_value(
        "training code identity",
        run.manifest.get("training_code_identity"),
        certificate.training_code_identity,
    )
    environment = _json_object(run.path / "environment.json")
    _same_value(
        "source environment hash",
        canonical_json_hash(environment),
        run.manifest.get("environment_identity"),
    )
    _same_value("environment identity", environment, certificate.environment_identity)
    _same_value("replay grade", run.replay_grade.value, certificate.replay_grade)
    known_occurrences = {
        str(row["occurrence_id"]) for row in run.ledger.iter_occurrences()
    }
    try:
        parse_patch(
            patch.model_dump(mode="json"),
            expected_run_id=run.run_id,
            expected_run_hash=run.run_hash,
            expected_behavior_contract_hash=certificate.behavior_contract_hash,
            known_occurrence_ids=known_occurrences,
        )
    except Exception as error:
        raise VerificationError(
            f"patch does not bind to source run: {error}"
        ) from error


def verify_bundle(
    bundle_root: Path, *, source_run: Path | None = None
) -> EvidenceCertificate:
    """Validate declared bytes, cross-artifact semantics, and optional source run."""

    root = bundle_root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("artifacts"), dict
    ):
        raise VerificationError("unsupported bundle manifest")
    artifacts: dict[str, str] = manifest["artifacts"]
    if len(artifacts) > MAX_ARTIFACTS:
        raise VerificationError("bundle declares too many artifacts")
    missing = MANDATORY_ARTIFACTS - set(artifacts)
    if missing:
        raise VerificationError(
            f"bundle is missing mandatory artifacts: {', '.join(sorted(missing))}"
        )
    for relative, expected in artifacts.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or SHA256_PATTERN.fullmatch(expected) is None
        ):
            raise VerificationError("invalid artifact map")
        actual = sha256_file(_safe_artifact(root, relative))
        if actual != expected:
            raise VerificationError(f"artifact hash mismatch: {relative}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != set(artifacts):
        raise VerificationError("bundle contains undeclared or missing files")
    certificate_path = root / "certificate.json"
    try:
        certificate = EvidenceCertificate.model_validate(_json_object(certificate_path))
    except Exception as error:
        raise VerificationError(f"invalid evidence certificate: {error}") from error
    expected_generated = {
        relative: expected
        for relative, expected in artifacts.items()
        if relative != "certificate.json"
    }
    _same_value(
        "certificate artifact hashes",
        certificate.generated_artifact_hashes,
        expected_generated,
    )
    raw_patch = _json_object(root / "patch.json")
    verify_patch(raw_patch, expected_hash=certificate.patch_hash)
    try:
        patch = parse_patch(
            raw_patch,
            expected_run_id=certificate.source_run.id,
            expected_run_hash=certificate.source_run.hash,
            expected_behavior_contract_hash=certificate.behavior_contract_hash,
        )
    except Exception as error:
        raise VerificationError(f"invalid evidence patch: {error}") from error
    materialized = _verify_behavior_and_controls(root, certificate)
    candidates = _candidate_occurrences(root, certificate)
    has_example_text = EXAMPLE_TEXT_ARTIFACT in artifacts
    if certificate.training_text_included != has_example_text:
        raise VerificationError(
            "training text privacy flag disagrees with bundle artifacts"
        )
    if has_example_text:
        _verify_candidate_example_text(
            root,
            candidate_occurrences=candidates,
            expected_rows=int(certificate.candidate_counts.get("retrieved", -1)),
        )
    unknown = sorted(set(patch.occurrence_ids) - candidates)
    if unknown:
        raise VerificationError(
            "causal patch includes occurrences absent from the candidate ranking: "
            + ", ".join(unknown[:3])
        )
    semantics = certificate.intervention_semantics
    declared_operations = semantics.get("operations")
    if not isinstance(declared_operations, list) or any(
        not isinstance(item, str) for item in declared_operations
    ):
        raise VerificationError("certificate intervention operations are malformed")
    _same_value(
        "selected occurrence count",
        semantics.get("selected_occurrences"),
        len(patch.occurrence_ids),
    )
    _same_value(
        "patch normalization",
        semantics.get("normalization"),
        str(patch.operations[0].normalization),
    )
    _same_value(
        "patch operation set",
        set(declared_operations),
        {str(operation.op) for operation in patch.operations},
    )
    counts = certificate.candidate_counts
    if len(certificate.candidate_methods) != len(
        set(certificate.candidate_methods)
    ) or set(certificate.candidate_method_configurations) != set(
        certificate.candidate_methods
    ):
        raise VerificationError("candidate method declarations are inconsistent")
    final_count = counts.get("final_causal_core")
    verified_count = counts.get("verified_relevant")
    retrieved_count = counts.get("retrieved")
    total_count = counts.get("total_occurrences")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (final_count, verified_count, retrieved_count, total_count)
    ):
        raise VerificationError("candidate counts are internally inconsistent")
    assert isinstance(final_count, int)
    assert isinstance(verified_count, int)
    assert isinstance(retrieved_count, int)
    assert isinstance(total_count, int)
    if (
        final_count != len(patch.occurrence_ids)
        or not final_count <= verified_count <= retrieved_count <= total_count
    ):
        raise VerificationError("candidate counts are internally inconsistent")
    if len(candidates) > total_count:
        raise VerificationError(
            "candidate ranking exceeds the certified occurrence count"
        )
    if len(materialized.get("controls", [])) != len(certificate.control_results):
        raise VerificationError("control declarations and results differ in count")
    _verify_blame(root, certificate, patch)
    _verify_counterfactual_model(root, certificate)
    if source_run is not None:
        _verify_source_run(Path(source_run), certificate, patch)
    return certificate
