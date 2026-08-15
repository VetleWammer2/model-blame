"""Path confinement helpers for untrusted artifact metadata."""

from __future__ import annotations

import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath


class UnsafePathError(ValueError):
    """Raised when an untrusted path escapes its declared root."""


_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CLOCK$",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def validate_relative_path(value: str, *, allow_current: bool = False) -> str:
    """Validate and normalize a portable, relative manifest path.

    Backslashes are interpreted as separators even on POSIX so a manifest
    cannot become safe or unsafe merely by moving between operating systems.
    """

    if not isinstance(value, str) or not value or "\x00" in value:
        raise UnsafePathError("path must be a non-empty string without NUL bytes")
    if len(value) > 4096:
        raise UnsafePathError("path exceeds 4096 characters")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise UnsafePathError("absolute and drive-qualified paths are forbidden")
    parts = posix.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafePathError("empty, current, and parent path segments are forbidden")
    for part in parts:
        stem = part.split(".", maxsplit=1)[0].upper()
        if ":" in part or part.endswith((" ", ".")) or stem in _WINDOWS_RESERVED_NAMES:
            raise UnsafePathError("path contains a Windows-reserved segment")
    if not allow_current and not parts:
        raise UnsafePathError("path must identify a child of the declared root")
    return PurePosixPath(*parts).as_posix()


def resolve_within_root(
    root: str | Path,
    relative: str | PurePath,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a validated relative path and ensure symlinks cannot escape."""

    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise UnsafePathError("declared root is not a directory")
    normalized = validate_relative_path(str(relative))
    candidate = (root_path / Path(*PurePosixPath(normalized).parts)).resolve(
        strict=must_exist
    )
    try:
        candidate.relative_to(root_path)
    except ValueError as error:
        raise UnsafePathError("resolved path escapes the declared root") from error
    if os.path.commonpath((str(root_path), str(candidate))) != str(root_path):
        raise UnsafePathError("resolved path escapes the declared root")
    return candidate


def ensure_output_directory(root: str | Path, relative: str | PurePath) -> Path:
    """Create and return a confined output directory."""

    destination = resolve_within_root(root, relative)
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise UnsafePathError(f"output is not a directory: {destination}")
    return destination
