"""Atomic evidence-bundle assembly with redacted training data by default."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from modelblame.evidence.certificate import EvidenceCertificate
from modelblame.evidence.verify import sha256_file


def write_evidence_bundle(
    destination: Path,
    *,
    certificate: EvidenceCertificate,
    artifacts: Mapping[str, Path | bytes | str],
) -> Path:
    """Write a complete bundle and atomically expose it at ``destination``."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        for relative, source in artifacts.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe bundle artifact path: {relative!r}")
            target = temporary / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, Path):
                shutil.copyfile(source, target)
            elif isinstance(source, bytes):
                target.write_bytes(source)
            else:
                # Preserve the exact bytes hashed by bundle producers on every
                # platform; text-mode writes would translate LF to CRLF on
                # Windows and invalidate the certificate's artifact digest.
                target.write_bytes(source.encode("utf-8"))
        certificate_path = temporary / "certificate.json"
        certificate_path.write_text(
            json.dumps(certificate.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        hashes = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {"schema_version": 1, "artifacts": hashes}
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
