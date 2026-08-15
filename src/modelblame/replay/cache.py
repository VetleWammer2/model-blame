"""Content-addressed, bounded replay-result cache."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplayCacheKey:
    source_checkpoint_hash: str
    remaining_history_hash: str
    patch_hash: str
    adapter_hash: str
    training_configuration_hash: str
    behavior_contract_hash: str
    environment_compatibility_class: str

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReplayCache:
    def __init__(self, root: Path, *, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

    def load(self, key: ReplayCacheKey) -> dict[str, Any] | None:
        path = self.root / key.digest / "result.json"
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        if value.get("cache_key") != key.digest:
            return None
        return value

    def store(self, key: ReplayCacheKey, result: dict[str, Any]) -> Path:
        destination = self.root / key.digest
        try:
            destination.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            # A stale or concurrently produced cache entry must never turn a
            # successfully executed replay into a failure. Invalid entries are
            # ignored by ``load`` and can be pruned by normal cache rotation.
            return destination
        value = dict(result)
        value["cache_key"] = key.digest
        (destination / "result.json").write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        self._prune()
        return destination

    def _prune(self) -> None:
        entries = sorted(
            (path for path in self.root.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        for path in entries[: max(0, len(entries) - self.max_entries)]:
            shutil.rmtree(path)
