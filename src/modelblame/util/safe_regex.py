"""A bounded, non-backtracking regular-expression subset for untrusted contracts.

The standard :mod:`re` engine is intentionally not used for matching behavior
contract patterns.  Even a short, syntactically valid pattern can trigger
catastrophic backtracking.  This module accepts a conservative regular subset
and evaluates it as a small nondeterministic automaton with bounded state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MAX_PATTERN_CHARACTERS = 8_192
MAX_INPUT_CHARACTERS = 16_384
MAX_REPEAT = 4_096
MAX_VARIABLE_REPEATS = 64
MAX_REPEAT_AMBIGUITY = 1_024


class UnsafeRegexError(ValueError):
    """Raised when a pattern is invalid or outside the safe regex subset."""


@dataclass(frozen=True)
class _ClassElement:
    kind: str
    value: str
    escaped: bool = False


@dataclass(frozen=True)
class _CharacterClass:
    literals: frozenset[str]
    ranges: tuple[tuple[str, str], ...]
    categories: tuple[str, ...]
    negated: bool = False

    def matches(self, character: str) -> bool:
        matched = character in self.literals or any(
            start <= character <= end for start, end in self.ranges
        )
        if not matched:
            matched = any(
                _category_matches(category, character) for category in self.categories
            )
        return not matched if self.negated else matched


@dataclass(frozen=True)
class _Atom:
    kind: str
    value: str | _CharacterClass
    minimum: int = 1
    maximum: int | None = 1

    def matches(self, character: str) -> bool:
        if self.kind == "literal":
            return character == self.value
        if self.kind == "dot":
            return character != "\n"
        if self.kind == "category":
            assert isinstance(self.value, str)
            return _category_matches(self.value, character)
        assert self.kind == "class" and isinstance(self.value, _CharacterClass)
        return self.value.matches(character)


@dataclass(frozen=True)
class SafeRegex:
    """A compiled safe-subset pattern with deterministic full-match semantics."""

    pattern: str
    atoms: tuple[_Atom, ...]

    def fullmatch(self, text: str) -> bool:
        if not isinstance(text, str):
            raise TypeError("safe regex input must be a string")
        if len(text) > MAX_INPUT_CHARACTERS:
            raise UnsafeRegexError(
                f"regex input exceeds {MAX_INPUT_CHARACTERS} characters"
            )

        # A state is (atom index, repetitions consumed for that atom).  Counts
        # for unbounded repeats are saturated at their minimum: once the
        # minimum is met, the exact count can no longer affect acceptance.
        states: set[tuple[int, int]] = {(0, 0)}
        for character in text:
            states = self._epsilon_closure(states)
            next_states: set[tuple[int, int]] = set()
            for atom_index, consumed in states:
                if atom_index == len(self.atoms):
                    continue
                atom = self.atoms[atom_index]
                if atom.maximum is not None and consumed >= atom.maximum:
                    continue
                if atom.matches(character):
                    new_count = consumed + 1
                    if atom.maximum is None and new_count >= atom.minimum:
                        new_count = atom.minimum
                    next_states.add((atom_index, new_count))
            if not next_states:
                return False
            states = next_states

        return (len(self.atoms), 0) in self._epsilon_closure(states)

    def _epsilon_closure(self, initial: set[tuple[int, int]]) -> set[tuple[int, int]]:
        closure = set(initial)
        pending = list(initial)
        while pending:
            atom_index, consumed = pending.pop()
            if atom_index == len(self.atoms):
                continue
            atom = self.atoms[atom_index]
            if consumed >= atom.minimum:
                successor = (atom_index + 1, 0)
                if successor not in closure:
                    closure.add(successor)
                    pending.append(successor)
        return closure


def _category_matches(category: str, character: str) -> bool:
    if category == "d":
        return character.isdecimal()
    if category == "D":
        return not character.isdecimal()
    if category == "s":
        return character.isspace()
    if category == "S":
        return not character.isspace()
    if category == "w":
        return character == "_" or character.isalnum()
    if category == "W":
        return character != "_" and not character.isalnum()
    raise AssertionError(f"unknown safe regex category: {category!r}")


class _Parser:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.position = 0

    def parse(self) -> tuple[_Atom, ...]:
        if self.pattern.startswith("^"):
            self.position = 1
        elif self.pattern.startswith(r"\A"):
            self.position = 2

        terminal = len(self.pattern)
        if self.pattern.endswith("$") and not _is_escaped(self.pattern, terminal - 1):
            terminal -= 1
        elif self.pattern.endswith(r"\Z") and not _is_escaped(
            self.pattern, terminal - 2
        ):
            terminal -= 2

        atoms: list[_Atom] = []
        variable_repeats = 0
        repeat_ambiguity = 0
        while self.position < terminal:
            atom = self._parse_atom(terminal)
            minimum, maximum = self._parse_quantifier(terminal)
            if minimum != maximum:
                variable_repeats += 1
                if variable_repeats > MAX_VARIABLE_REPEATS:
                    raise UnsafeRegexError(
                        f"pattern exceeds {MAX_VARIABLE_REPEATS} variable repeats"
                    )
                repeat_ambiguity += 1 if maximum is None else maximum - minimum
                if repeat_ambiguity > MAX_REPEAT_AMBIGUITY:
                    raise UnsafeRegexError(
                        "pattern repeat state exceeds the safe bound"
                    )
            atoms.append(
                _Atom(
                    kind=atom.kind,
                    value=atom.value,
                    minimum=minimum,
                    maximum=maximum,
                )
            )

        if self.position != terminal:
            raise UnsafeRegexError("could not parse the complete regular expression")
        return tuple(atoms)

    def _parse_atom(self, terminal: int) -> _Atom:
        character = self.pattern[self.position]
        if character == "[":
            return self._parse_character_class(terminal)
        if character == "\\":
            element = self._parse_escape(terminal, in_class=False)
            if element.kind == "category":
                return _Atom("category", element.value)
            return _Atom("literal", element.value)
        if character == ".":
            self.position += 1
            return _Atom("dot", "")
        if character in "()|":
            raise UnsafeRegexError(
                "groups, alternation, lookarounds, and backreferences are not "
                "supported by the safe regex subset"
            )
        if character in "*+?{}":
            raise UnsafeRegexError("quantifier has no preceding atom")
        if character in "^$":
            raise UnsafeRegexError("anchors are allowed only at pattern boundaries")
        self.position += 1
        return _Atom("literal", character)

    def _parse_escape(self, terminal: int, *, in_class: bool) -> _ClassElement:
        self.position += 1
        if self.position >= terminal:
            raise UnsafeRegexError("regular expression ends with an incomplete escape")
        escaped = self.pattern[self.position]
        self.position += 1
        if escaped in "dDsSwW":
            return _ClassElement("category", escaped, escaped=True)
        if escaped in "nrtfv":
            return _ClassElement(
                "literal",
                {"n": "\n", "r": "\r", "t": "\t", "f": "\f", "v": "\v"}[escaped],
                escaped=True,
            )
        if escaped in r".\^$*+?{}[]|()-":
            return _ClassElement("literal", escaped, escaped=True)
        if escaped.isdigit() or escaped in "bBAZG":
            raise UnsafeRegexError(
                "backreferences and zero-width assertions are not supported by "
                "the safe regex subset"
            )
        location = "character class" if in_class else "pattern"
        raise UnsafeRegexError(f"unsupported escape \\{escaped} in {location}")

    def _parse_character_class(self, terminal: int) -> _Atom:
        self.position += 1
        negated = self.position < terminal and self.pattern[self.position] == "^"
        if negated:
            self.position += 1
        elements: list[_ClassElement] = []
        if self.position < terminal and self.pattern[self.position] == "]":
            elements.append(_ClassElement("literal", "]"))
            self.position += 1

        closed = False
        while self.position < terminal:
            character = self.pattern[self.position]
            if character == "]":
                self.position += 1
                closed = True
                break
            if character == "\\":
                elements.append(self._parse_escape(terminal, in_class=True))
            else:
                self.position += 1
                elements.append(_ClassElement("literal", character))
        if not closed:
            raise UnsafeRegexError("unterminated character class")
        if not elements:
            raise UnsafeRegexError("empty character classes are not supported")

        literals: set[str] = set()
        ranges: list[tuple[str, str]] = []
        categories: list[str] = []
        index = 0
        while index < len(elements):
            current = elements[index]
            if (
                index + 2 < len(elements)
                and current.kind == "literal"
                and elements[index + 1].kind == "literal"
                and elements[index + 1].value == "-"
                and not elements[index + 1].escaped
                and elements[index + 2].kind == "literal"
            ):
                end = elements[index + 2].value
                if current.value > end:
                    raise UnsafeRegexError("character-class range is reversed")
                ranges.append((current.value, end))
                index += 3
                continue
            if current.kind == "category":
                categories.append(current.value)
            else:
                literals.add(current.value)
            index += 1

        return _Atom(
            "class",
            _CharacterClass(
                literals=frozenset(literals),
                ranges=tuple(ranges),
                categories=tuple(categories),
                negated=negated,
            ),
        )

    def _parse_quantifier(self, terminal: int) -> tuple[int, int | None]:
        if self.position >= terminal:
            return 1, 1
        marker = self.pattern[self.position]
        if marker == "?":
            self.position += 1
            result: tuple[int, int | None] = (0, 1)
        elif marker == "*":
            self.position += 1
            result = (0, None)
        elif marker == "+":
            self.position += 1
            result = (1, None)
        elif marker == "{":
            result = self._parse_braced_quantifier(terminal)
        else:
            return 1, 1

        if self.position < terminal and self.pattern[self.position] in "?+*{":
            raise UnsafeRegexError(
                "lazy, possessive, and nested quantifiers are not supported"
            )
        return result

    def _parse_braced_quantifier(self, terminal: int) -> tuple[int, int | None]:
        self.position += 1
        start = self.position
        while self.position < terminal and self.pattern[self.position].isdigit():
            self.position += 1
        if start == self.position:
            raise UnsafeRegexError("repeat quantifier requires a lower bound")
        minimum = int(self.pattern[start : self.position])
        maximum: int | None
        if self.position < terminal and self.pattern[self.position] == "}":
            maximum = minimum
        elif self.position < terminal and self.pattern[self.position] == ",":
            self.position += 1
            upper_start = self.position
            while self.position < terminal and self.pattern[self.position].isdigit():
                self.position += 1
            maximum = (
                int(self.pattern[upper_start : self.position])
                if upper_start != self.position
                else None
            )
        else:
            raise UnsafeRegexError("malformed repeat quantifier")
        if self.position >= terminal or self.pattern[self.position] != "}":
            raise UnsafeRegexError("unterminated repeat quantifier")
        self.position += 1
        if minimum > MAX_REPEAT or (maximum is not None and maximum > MAX_REPEAT):
            raise UnsafeRegexError(f"repeat bounds may not exceed {MAX_REPEAT}")
        if maximum == 0:
            raise UnsafeRegexError("zero-width repeat atoms are not supported")
        if maximum is not None and minimum > maximum:
            raise UnsafeRegexError("repeat lower bound exceeds upper bound")
        return minimum, maximum


def _is_escaped(pattern: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and pattern[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


@lru_cache(maxsize=32)
def compile_safe_regex(pattern: str) -> SafeRegex:
    """Validate and compile an untrusted pattern into the bounded matcher."""

    if not isinstance(pattern, str):
        raise TypeError("safe regex pattern must be a string")
    if not pattern:
        raise UnsafeRegexError("regular expression must not be empty")
    if len(pattern) > MAX_PATTERN_CHARACTERS:
        raise UnsafeRegexError(
            f"regular expression exceeds {MAX_PATTERN_CHARACTERS} characters"
        )
    return SafeRegex(pattern=pattern, atoms=_Parser(pattern).parse())
