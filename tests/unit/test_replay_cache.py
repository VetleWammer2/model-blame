from __future__ import annotations

import sys

from modelblame.replay.cache import ReplayCache, ReplayCacheKey
from modelblame.replay.process import run_isolated


def key(**overrides: str) -> ReplayCacheKey:
    values = {
        "source_checkpoint_hash": "a",
        "remaining_history_hash": "b",
        "patch_hash": "c",
        "adapter_hash": "d",
        "training_configuration_hash": "e",
        "behavior_contract_hash": "f",
        "environment_compatibility_class": "g",
    }
    values.update(overrides)
    return ReplayCacheKey(**values)


def test_every_relevant_cache_key_component_changes_digest() -> None:
    baseline = key().digest
    for field in key().__dataclass_fields__:
        assert key(**{field: "changed"}).digest != baseline


def test_cache_round_trip(tmp_path) -> None:
    cache = ReplayCache(tmp_path, max_entries=2)
    cache.store(key(), {"status": "COMPLETED"})
    loaded = cache.load(key())
    assert loaded is not None
    assert loaded["status"] == "COMPLETED"


def test_cache_ignores_non_object_json(tmp_path) -> None:
    cache_key = key()
    cache = ReplayCache(tmp_path)
    entry = tmp_path / cache_key.digest
    entry.mkdir()
    (entry / "result.json").write_text("[]\n", encoding="utf-8")
    assert cache.load(cache_key) is None


def test_process_runner_uses_argument_vector_and_timeout(tmp_path) -> None:
    outcome = run_isolated(
        [sys.executable, "-c", "print('real child')"],
        output_directory=tmp_path / "child",
        timeout_seconds=10,
    )
    assert outcome.status == "COMPLETED"
    assert outcome.stdout.strip() == "real child"
