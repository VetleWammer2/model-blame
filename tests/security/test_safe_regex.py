from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from modelblame.behavior.contract import load_contract_artifact
from modelblame.behavior.scorers import score_probe
from modelblame.util.safe_regex import (
    MAX_INPUT_CHARACTERS,
    UnsafeRegexError,
    compile_safe_regex,
)


class _GeneratedTextModel:
    def __init__(self, generated: str) -> None:
        self.generated = generated

    def sequence_log_probability(self, prompt: str, completion: str) -> float:
        return 0.0

    def token_log_probability(self, prompt: str, token: str) -> float:
        return 0.0

    def greedy_generate(self, prompt: str, *, max_new_tokens: int) -> str:
        return self.generated

    def scalar_loss(self, examples: Sequence[Mapping[str, Any]]) -> float:
        return 0.0


@pytest.mark.parametrize(
    ("pattern", "matching", "nonmatching"),
    [
        ("exact text", "exact text", "exact texts"),
        (r"^[A-Z][a-z]+$", "Nareth", "nareth"),
        (r"item-[0-9]{2,4}", "item-2048", "item-2"),
        (r"\{[A-Za-z0-9_ ]*\}", "{compact JSON_2}", "compact JSON_2"),
        (r"^[^\n]+$", "one line", "two\nlines"),
        (r"\w+\s\d?", "answer 7", "answer seven"),
    ],
)
def test_safe_regex_fullmatch_accepts_documented_subset(
    pattern: str, matching: str, nonmatching: str
) -> None:
    compiled = compile_safe_regex(pattern)
    assert compiled.fullmatch(matching)
    assert not compiled.fullmatch(nonmatching)


@pytest.mark.parametrize(
    "pattern",
    [
        r"^(a+)+$",  # nested quantified group
        r"(a|aa)+$",  # ambiguous quantified alternation
        r"(?=a)a",  # lookahead
        r"(?<=a)b",  # lookbehind
        r"(a)\1",  # numeric backreference
        r"(?P<letter>a)(?P=letter)",  # named group and backreference
        r"\bword\b",  # zero-width word-boundary assertion
        r"a+?",  # lazy quantifier
        r"a++",  # possessive quantifier
    ],
)
def test_safe_regex_rejects_backtracking_and_zero_width_constructs(
    pattern: str,
) -> None:
    with pytest.raises(UnsafeRegexError):
        compile_safe_regex(pattern)


def test_scorer_revalidates_untyped_pattern_instead_of_using_python_re() -> None:
    scorer = {"type": "greedy_regex_match_rate", "pattern": r"^(a+)+$"}
    with pytest.raises(UnsafeRegexError):
        score_probe(_GeneratedTextModel("a" * 1_000 + "!"), scorer, {"prompt": "p"})


def test_probe_file_regex_is_validated_when_contract_is_materialized(tmp_path) -> None:
    (tmp_path / "search.jsonl").write_text(
        '{"prompt":"p","pattern":"^(a+)+$"}\n', encoding="utf-8"
    )
    (tmp_path / "behavior.yaml").write_text(
        """schema_version: 1
id: regex-file
scorer:
  type: greedy_regex_match_rate
present_threshold: 1.0
required_effect: 0.5
search:
  prompts_file: search.jsonl
holdout:
  sealed: true
  prompts: [{prompt: h, pattern: "safe.*"}]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="unsafe regular expression"):
        load_contract_artifact(tmp_path / "behavior.yaml")


def test_safe_regex_bounds_generated_text() -> None:
    matcher = compile_safe_regex("a*")
    with pytest.raises(UnsafeRegexError, match="input exceeds"):
        matcher.fullmatch("a" * (MAX_INPUT_CHARACTERS + 1))


def test_safe_regex_bounds_automaton_ambiguity() -> None:
    with pytest.raises(UnsafeRegexError, match="variable repeats"):
        compile_safe_regex("a*" * 65)


def test_safe_regex_rejects_zero_width_repeat_chains() -> None:
    with pytest.raises(UnsafeRegexError, match="zero-width repeat"):
        compile_safe_regex("a{0}")
