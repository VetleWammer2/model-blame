from __future__ import annotations

import json

import pytest

from modelblame.util.atomic import (
    atomic_directory,
    atomic_write_bytes,
    atomic_write_json,
)
from modelblame.util.canonical_json import hash_file
from modelblame.util.paths import (
    UnsafePathError,
    resolve_within_root,
    validate_relative_path,
)


def test_portable_relative_paths() -> None:
    assert validate_relative_path(r"controls\neighbors.jsonl") == (
        "controls/neighbors.jsonl"
    )
    for unsafe in (
        "../secret",
        r"folder\..\secret",
        "/etc/passwd",
        r"C:\x",
        "file.txt:stream",
        "CON",
        "trailing.",
    ):
        with pytest.raises(UnsafePathError):
            validate_relative_path(unsafe)


def test_resolve_within_root(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "file.txt"
    source.write_text("value", encoding="utf-8")
    assert resolve_within_root(tmp_path, "nested/file.txt", must_exist=True) == source


def test_atomic_writes_and_hash(tmp_path) -> None:
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert hash_file(target) == (
        "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"
    )
    json_target = tmp_path / "value.json"
    atomic_write_json(json_target, {"z": 1, "a": 2})
    assert json.loads(json_target.read_text(encoding="utf-8")) == {"a": 2, "z": 1}


def test_atomic_directory_never_replaces_existing(tmp_path) -> None:
    destination = tmp_path / "checkpoint"
    with atomic_directory(destination) as temporary:
        (temporary / "complete").write_text("yes", encoding="utf-8")
    assert (destination / "complete").read_text(encoding="utf-8") == "yes"
    with pytest.raises(FileExistsError):
        with atomic_directory(destination):
            pass
