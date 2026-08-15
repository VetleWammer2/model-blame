from __future__ import annotations

import json

import pytest

from modelblame.config.validation import ConfigLoadError, read_structured_file
from modelblame.patch.schema import PatchValidationError, parse_patch
from modelblame.util.canonical_json import CanonicalizationError, validate_json_tree
from modelblame.util.paths import UnsafePathError, resolve_within_root


def test_duplicate_json_and_yaml_keys_are_rejected(tmp_path) -> None:
    json_path = tmp_path / "duplicate.json"
    json_path.write_text('{"id":"first","id":"second"}', encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="duplicate JSON"):
        read_structured_file(json_path)
    yaml_path = tmp_path / "duplicate.yaml"
    yaml_path.write_text("id: first\nid: second\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="duplicate YAML"):
        read_structured_file(yaml_path)


def test_yaml_aliases_and_python_tags_are_rejected(tmp_path) -> None:
    aliases = tmp_path / "alias.yaml"
    aliases.write_text("root: &root [1, 2]\ncopy: *root\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="aliases"):
        read_structured_file(aliases)
    tagged = tmp_path / "tag.yaml"
    tagged.write_text("value: !!python/object:builtins.object {}\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError):
        read_structured_file(tagged)


def test_deep_and_cyclic_values_are_rejected() -> None:
    value: dict[str, object] = {}
    cursor = value
    for index in range(40):
        child: dict[str, object] = {}
        cursor[str(index)] = child
        cursor = child
    with pytest.raises(CanonicalizationError, match="nesting"):
        validate_json_tree(value, max_depth=16)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(CanonicalizationError, match="cyclic"):
        validate_json_tree(cyclic)


def test_symlink_escape_is_rejected_when_supported(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this platform")
    with pytest.raises(UnsafePathError, match="escapes"):
        resolve_within_root(tmp_path, "escape/file.json")


def test_patch_rejects_duplicate_keys_and_path_reference() -> None:
    duplicate = b'{"schema_version":1,"schema_version":1}'
    with pytest.raises(PatchValidationError, match="duplicate JSON"):
        parse_patch(duplicate, require_hash=False)
    payload = {
        "schema_version": 1,
        "run_id": "mb_test",
        "run_hash": "a" * 64,
        "behavior_contract_hash": "b" * 64,
        "operations": [
            {
                "op": "GRADIENT_ABLATE",
                "occurrence_ids": ["occ_" + "c" * 64],
                "normalization": "FIXED_DENOMINATOR",
                "path": "../../payload.py",
            }
        ],
    }
    with pytest.raises(PatchValidationError, match="extra_forbidden"):
        parse_patch(json.dumps(payload).encode(), require_hash=False)


def test_replay_worker_rejects_output_inside_source_run(tmp_path) -> None:
    from modelblame.replay.worker import execute_replay

    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="outside the immutable source run"):
        execute_replay(
            run_path=run,
            patch_path=tmp_path / "patch.json",
            behavior_path=tmp_path / "behavior.yaml",
            output_path=run / "counterfactual",
            unseal_holdout=False,
            device="cpu",
        )
