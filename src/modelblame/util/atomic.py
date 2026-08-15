"""Atomic artifact writes on a single filesystem."""

from __future__ import annotations

import json
import os
import secrets
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from modelblame.util.canonical_json import canonical_json_bytes


def _temporary_sibling(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp-{secrets.token_hex(8)}")


def atomic_write_bytes(destination: str | Path, data: bytes) -> None:
    """Write bytes completely before replacing ``destination``."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    destination: str | Path, text: str, *, encoding: str = "utf-8"
) -> None:
    """Atomically write text using a declared encoding."""

    atomic_write_bytes(destination, text.encode(encoding))


def atomic_write_json(
    destination: str | Path, value: Any, *, canonical: bool = False
) -> None:
    """Atomically write safe JSON, optionally in canonical compact form."""

    if canonical:
        payload = canonical_json_bytes(value) + b"\n"
    else:
        payload = (
            json.dumps(
                value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8")
    atomic_write_bytes(destination, payload)


@contextmanager
def atomic_directory(destination: str | Path) -> Iterator[Path]:
    """Yield a temporary sibling directory and publish it on clean exit.

    An existing destination is never overwritten.  This property prevents an
    interrupted checkpoint writer from damaging a previously valid checkpoint.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(target)
    temporary.mkdir()
    try:
        yield temporary
        if target.exists():
            raise FileExistsError(f"refusing to replace existing directory: {target}")
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
