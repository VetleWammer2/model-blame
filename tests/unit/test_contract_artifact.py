from __future__ import annotations

import json

import yaml

from modelblame.behavior.contract import load_contract_artifact, search_view


def test_external_holdout_is_hashed_and_excluded_from_search(tmp_path) -> None:
    (tmp_path / "holdout.jsonl").write_text(json.dumps("secret prompt") + "\n")
    (tmp_path / "behavior.yaml").write_text(
        """schema_version: 1
id: test
scorer:
  type: sequence_logprob_margin
  preferred: "yes"
  alternative: "no"
present_threshold: 0.0
required_effect: 0.1
search:
  prompts: [search prompt]
holdout:
  sealed: true
  prompts_file: holdout.jsonl
"""
    )
    _, materialized, first_hash = load_contract_artifact(tmp_path / "behavior.yaml")
    assert "holdout" not in search_view(materialized, contract_hash=first_hash)
    (tmp_path / "holdout.jsonl").write_text(json.dumps("mutated") + "\n")
    _, _, second_hash = load_contract_artifact(tmp_path / "behavior.yaml")
    assert first_hash != second_hash


def test_materialized_contract_preserves_identity_without_source_paths(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "holdout.jsonl").write_text(json.dumps("secret prompt") + "\n")
    (source / "behavior.yaml").write_text(
        """schema_version: 1
id: portable
scorer:
  type: sequence_logprob_margin
  preferred: "yes"
  alternative: "no"
present_threshold: 0.0
required_effect: 0.1
search:
  prompts: [search prompt]
holdout:
  sealed: true
  prompts_file: holdout.jsonl
"""
    )
    _, materialized, source_hash = load_contract_artifact(source / "behavior.yaml")
    copied = tmp_path / "copied.yaml"
    copied.write_text(yaml.safe_dump(materialized, sort_keys=True))
    _, copied_materialized, copied_hash = load_contract_artifact(copied)
    assert copied_materialized == materialized
    assert copied_hash == source_hash
