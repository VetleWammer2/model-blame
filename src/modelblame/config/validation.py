"""Bounded structured-data loading for untrusted configuration files."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from modelblame.util.canonical_json import CanonicalizationError, validate_json_tree

MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_CONFIG_DEPTH = 32
MAX_CONFIG_ITEMS = 100_000


class ConfigLoadError(ValueError):
    """Raised when untrusted configuration bytes cannot be loaded safely."""


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigLoadError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


class _NoAliasSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects aliases and duplicate mapping keys."""

    def compose_node(self, parent: Any, index: Any) -> yaml.Node | None:
        if self.check_event(yaml.AliasEvent):
            raise ConfigLoadError("YAML aliases are not accepted in configuration")
        return super().compose_node(parent, index)


# PyYAML defaults to YAML 1.1, where ordinary words such as ``yes`` and ``no``
# unexpectedly become booleans. Contracts use model completions heavily, so
# follow YAML 1.2's boolean spelling while retaining SafeLoader constructors.
_NoAliasSafeLoader.yaml_implicit_resolvers = {
    key: list(resolvers)
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for first_character, resolvers in _NoAliasSafeLoader.yaml_implicit_resolvers.items():
    _NoAliasSafeLoader.yaml_implicit_resolvers[first_character] = [
        resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"
    ]
_NoAliasSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def _construct_unique_mapping(
    loader: _NoAliasSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConfigLoadError("YAML mapping keys must be scalar") from error
        if duplicate:
            raise ConfigLoadError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_NoAliasSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def read_structured_file(
    path: str | Path,
    *,
    allowed_suffixes: frozenset[str] | None = None,
    max_bytes: int = MAX_CONFIG_BYTES,
) -> dict[str, Any]:
    """Load TOML, JSON, or YAML with size, shape, and duplicate-key checks."""

    source = Path(path)
    suffix = source.suffix.lower()
    allowed = allowed_suffixes or frozenset({".toml", ".json", ".yaml", ".yml"})
    if suffix not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ConfigLoadError(
            f"unsupported configuration suffix {suffix!r}; use {choices}"
        )
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    try:
        stat = source.stat()
    except OSError as error:
        raise ConfigLoadError(f"cannot stat configuration {source}: {error}") from error
    if not source.is_file():
        raise ConfigLoadError(f"configuration is not a regular file: {source}")
    if stat.st_size > max_bytes:
        raise ConfigLoadError(f"configuration exceeds the {max_bytes}-byte limit")
    try:
        raw = source.read_bytes()
        if len(raw) > max_bytes:
            raise ConfigLoadError(f"configuration exceeds the {max_bytes}-byte limit")
        if suffix == ".toml":
            parsed = tomllib.loads(raw.decode("utf-8"))
        elif suffix == ".json":
            parsed = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_pairs,
                parse_constant=lambda value: (_raise_nonfinite(value)),
            )
        else:
            parsed = yaml.load(raw.decode("utf-8"), Loader=_NoAliasSafeLoader)  # noqa: S506
    except (
        ConfigLoadError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        yaml.YAMLError,
    ) as error:
        if isinstance(error, ConfigLoadError):
            raise
        raise ConfigLoadError(
            f"cannot parse configuration {source}: {error}"
        ) from error
    if not isinstance(parsed, Mapping):
        raise ConfigLoadError("configuration root must be an object")
    try:
        validate_json_tree(
            parsed, max_depth=MAX_CONFIG_DEPTH, max_items=MAX_CONFIG_ITEMS
        )
    except CanonicalizationError as error:
        raise ConfigLoadError(str(error)) from error
    return dict(parsed)


def _raise_nonfinite(value: str) -> None:
    raise ConfigLoadError(f"non-finite JSON number is forbidden: {value}")
