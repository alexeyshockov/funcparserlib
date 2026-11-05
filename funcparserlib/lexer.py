# Copyright © 2009/2023 Andrey Vlasovskikh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Regexp-based tokenizers: turn text into a sequence of `Token` objects for the parser."""

__all__ = ["make_tokenizer", "Tokenizer", "TokenSpec", "Token", "LexerError", "Place", "Spec"]

import dataclasses as dc
import re
from collections.abc import Collection, Iterable, Sequence
from re import Pattern
from typing import Any, final

from ._commons import valueclass, with_hint

type Place = tuple[int, int]
"""A position in the text: (_line_, _column_), both counted from 1."""

type Spec = tuple[str, tuple[Any, ...]]
"""A legacy token spec: (_type_, _args_), where _args_ are the arguments of `re.compile()`."""

_CompiledSpec = tuple[str, Pattern[str]]

# What `_renumbered()` reads: an escape, and the brackets of a character class.
_REGEXP_PIECE = re.compile(r"\\[0-9]+|\\.|\[\^?\]?|\]", re.DOTALL)

# The flags a group can scope, `(?i:...)`; any other flag keeps a spec out of a joined pattern.
_NAMED_GROUP = re.compile(r"\(\?P<\w+>")
_SCOPED_FLAGS = {re.IGNORECASE: "i", re.MULTILINE: "m", re.DOTALL: "s", re.VERBOSE: "x"}


def _renumbered(pattern: str, offset: int, *, movable: bool) -> str | None:
    """`pattern` with its references to numbered groups moved up by `offset`.

    `None` when it has a reference that cannot move: `movable` is false, or the new number is one
    the `re` module would read as an octal escape.
    """
    in_class = False
    parts = []
    end = 0
    for m in _REGEXP_PIECE.finditer(pattern):
        piece = m.group()
        if piece.startswith("["):
            in_class = not piece.endswith("]") or not in_class
        elif piece == "]":
            in_class = False
        elif piece[1].isdigit() and not in_class and not _is_octal(piece[1:]):
            number = int(piece[1:]) + offset
            if not movable or len(piece) > 3 or number > 99:
                return None
            parts += [pattern[end : m.start()], f"\\{number}"]
            end = m.end()
    return "".join([*parts, pattern[end:]])


def _is_octal(digits: str) -> bool:
    """Whether the `re` module reads `digits`, after a backslash, as an octal escape."""
    return digits[0] == "0" or (len(digits) >= 3 and set(digits[:3]) <= set("01234567"))


def _joined_part(regexp: Pattern[str], group: int) -> str | None:
    """The part of a joined pattern for `regexp`, wrapped in the group numbered `group`.

    A spec's named groups become non-capturing, as a token is the whole match. `None` when the spec
    cannot be joined: it refers to a group by name or in a condition, it refers to one by number
    where a name or a comment gets in the way of moving the number, or it has a flag a group cannot
    scope.
    """
    pattern = regexp.pattern
    flags = regexp.flags & ~re.UNICODE
    # A comment can hold what reads as a bracket, which `_renumbered()` would follow.
    has_comment = bool(flags & re.VERBOSE) or "(?#" in pattern
    if (
        re.search(r"\(\?P=|\(\?\(", pattern)
        # Each `(?P<` is a group only if they number as many as the groups the pattern names.
        or len(_NAMED_GROUP.findall(pattern)) != len(regexp.groupindex)
        or (has_comment and re.search(r"\\[1-9]", pattern))
        or flags & ~sum(_SCOPED_FLAGS)
    ):
        return None
    renumbered = _renumbered(pattern, group, movable=not regexp.groupindex)
    if renumbered is None:
        return None
    letters = "".join(letter for flag, letter in _SCOPED_FLAGS.items() if flags & flag)
    # A verbose pattern may end in a comment, which a newline closes.
    end = "\n" if flags & re.VERBOSE else ""
    return f"((?{letters}:{_NAMED_GROUP.sub('(?:', renumbered)}{end}))"


def _combined(specs: Sequence[_CompiledSpec]) -> tuple[Pattern[str], list[str]] | None:
    """One pattern for `specs`, and the token type of each of its groups a spec is wrapped in.

    The group of the first spec to match is the `lastindex` of the match. `None` when there are no
    specs, or one cannot be joined.
    """
    parts = []
    types = [""]
    for token_type, regexp in specs:
        part = _joined_part(regexp, len(types))
        if part is None:
            return None
        parts.append(part)
        types += [token_type, *[""] * (regexp.groups - len(regexp.groupindex))]
    try:
        return (re.compile("|".join(parts)), types) if parts else None
    except re.error:
        # Such as a global flag, `(?i)`, inside a spec's pattern.
        return None


class LexerError(Exception):
    """Raised by a tokenizer when no token spec matches the text at `place`.

    `msg` is the line of the text where tokenizing stopped.
    """

    def __init__(self, place: Place, msg: str) -> None:
        self.place: Place = place
        self.msg: str = msg

    def __str__(self) -> str:
        """The place and the line, as `cannot tokenize data: line,column: "text"`."""
        line, pos = self.place
        return f'cannot tokenize data: {line},{pos}: "{self.msg}"'

    def render(self, source: str | None = None) -> str:
        """The place, then the line with `^` under the character no token spec matches.

        The error holds its line, so `source` makes no difference: it is there so that a handler can
        render a `LexerError` and a `funcparserlib.parser.NoParseError` the same way.
        """
        line, pos = self.place
        gutter = f"{line} | "
        marker = " " * (len(gutter) + pos - 1) + "^"
        return f"{line},{pos}: cannot tokenize data\n{gutter}{self.msg}\n{marker}"


@final
@valueclass(repr=False)
class TokenSpec:
    """A token specification for generating a lexer via `make_tokenizer()`."""

    type: str
    """User-defined type of the token (e.g. `"name"`, `"number"`, `"operator"`)"""
    pattern: str
    """Regexp for matching this token type"""
    flags: int = 0
    """Regexp flags, the second argument of `re.compile()`"""

    def __repr__(self) -> str:
        """The spec as the call that builds it: `TokenSpec('name', '[a-z]+', 0)`."""
        return f"TokenSpec({self.type!r}, {self.pattern!r}, {self.flags!r})"


@final
@valueclass(repr=False)
class Token:
    """A token object that represents a substring of certain type in your text.

    You can compare tokens for equality using the `==` operator. Tokens also define custom `repr()`
    and `str()`.
    """

    type: str
    """User-defined type of the token (e.g. `"name"`, `"number"`, `"operator"`)"""
    value: str
    """Text value of the token"""
    start: Place | None = dc.field(default=None, compare=False)
    """Start position (_line_, _column_)"""
    end: Place | None = dc.field(default=None, compare=False)
    """End position (_line_, _column_)"""

    def __repr__(self) -> str:
        """The token as the call that builds it from its type and value; positions are left out."""
        return f"Token({self.type!r}, {self.value!r})"

    def _pos_str(self) -> str:
        if self.start is None or self.end is None:
            return ""
        s_line, s_pos = self.start
        e_line, e_pos = self.end
        return f"{s_line},{s_pos}-{e_line},{e_pos}:"

    def __str__(self) -> str:
        """The token's span as `line,column-line,column:`, its type and its quoted value.

        The span is left out for a token without positions.
        """
        s = f"{self._pos_str()} {self.type} '{self.value}'"
        return s.strip()

    @property
    def name(self) -> str:
        """The token's value; parsers use it to name a parser made from a token."""
        return self.value

    def pformat(self) -> str:
        """The token in aligned columns: position, type and value."""
        return f"{self._pos_str().ljust(20)} {self.type.ljust(14)} '{self.value}'"


@final
class Tokenizer:
    """A function from text to tokens.

    Users get it from `make_tokenizer()`; its constructor is internal.
    """

    def __init__(self, specs: tuple[tuple[str, Pattern[str]], ...], skip: frozenset[str]) -> None:
        """A tokenizer of `specs` that leaves out the token types in `skip`."""
        self._specs = specs
        self._skip = skip
        combined = _combined(specs)
        self._pattern: Pattern[str] | None = combined[0] if combined is not None else None
        self._types: list[str] = combined[1] if combined is not None else []

    def _first_match(self, s: str, i: int) -> tuple[str, re.Match[str] | None]:
        for token_type, regexp in self._specs:
            if m := regexp.match(s, i):
                return token_type, m
        return "", None

    def __call__(self, s: str) -> Iterable[Token]:
        """The tokens of `s`, one at a time.

        Raises `LexerError` on reaching text no token spec matches.
        """
        pattern, types, skip = self._pattern, self._types, self._skip
        length = len(s)
        line, pos = 1, 0
        i = 0
        while i < length:
            if pattern is not None:
                m = pattern.match(s, i)
                token_type = types[m.lastindex or 0] if m is not None else ""
            else:
                token_type, m = self._first_match(s, i)
            if m is None:
                # Lines are counted by "\n" alone, as the token positions are.
                raise LexerError((line, pos + 1), s.split("\n")[line - 1].rstrip("\r"))
            value = m.group()
            if not value:
                raise ValueError(
                    f"the pattern of token {token_type!r} matched the empty string"
                    f" at {line},{pos + 1}"
                )
            nls = value.count("\n")
            if nls:
                end_line, end_pos = line + nls, len(value) - value.rfind("\n") - 1
            else:
                end_line, end_pos = line, pos + len(value)
            if token_type not in skip:
                yield Token(token_type, value, (line, pos + 1), (end_line, end_pos))
            line, pos = end_line, end_pos
            i = m.end()

    def _scan(self, s: str) -> tuple[list[str], list[str | None]] | None:
        """The values and the types of the tokens of `s`, the types ending in `None`.

        `None` when the tokens do not add up to `s`, or a pattern matched the empty string: then
        `Tokenizer.__call__()` has an error to raise. This reads one pattern, so it takes specs that
        join into one.
        """
        pattern, types, skip = self._pattern, self._types, self._skip
        if pattern is None:
            return None
        values: list[str] = []
        kinds: list[str | None] = []
        end = 0
        for m in pattern.finditer(s):
            value = m.group()
            if not value:
                return None
            end += len(value)
            token_type = types[m.lastindex or 0]
            if token_type not in skip:
                values.append(value)
                kinds.append(token_type)
        if end != len(s):
            return None
        kinds.append(None)
        return values, kinds


def make_tokenizer(specs: Sequence[TokenSpec | Spec], *, skip: Collection[str] = ()) -> Tokenizer:
    # noinspection GrazieInspection
    """Make a function that tokenizes text based on the regexp specs.

    Type: `(Sequence[TokenSpec | Tuple]) -> Tokenizer`

    A token spec is `TokenSpec` instance.

    !!! Note

        For legacy reasons, a token spec may also be a tuple of (_type_, _args_), where
        _type_ sets the value of `Token.type` for the token, and _args_ are the
        positional arguments for `re.compile()`: either just (_pattern_,) or
        (_pattern_, _flags_).

    It returns a `Tokenizer`, a function that takes a string and returns an iterable of `Token`
    objects, or raises `LexerError` if it cannot tokenize the string according to its token specs.

    The token types in `skip` are read and left out of the result, which suits whitespace and
    comments. A name no spec has raises `ValueError`.

    Examples:
        ```pycon
        >>> tokenize = make_tokenizer([
        ...     TokenSpec("space", r"\\s+"),
        ...     TokenSpec("id", r"\\w+"),
        ...     TokenSpec("op", r"[,!]"),
        ... ], skip={"space"})
        >>> text = "Hello, World!"
        >>> list(tokenize(text))
        [Token('id', 'Hello'), Token('op', ','), Token('id', 'World'), Token('op', '!')]
        >>> text = "Bye?"
        >>> list(tokenize(text))
        Traceback (most recent call last):
            ...
        funcparserlib.lexer.LexerError: cannot tokenize data: 1,4: "Bye?"

        ```
    """

    def compile_specs() -> Iterable[_CompiledSpec]:
        for spec in specs:
            if isinstance(spec, TokenSpec):
                token_type, regexp = spec.type, re.compile(spec.pattern, spec.flags)
            else:
                token_type, args = spec
                regexp = re.compile(*args)
            if regexp.match(""):
                raise with_hint(
                    ValueError(f"the pattern of token {token_type!r} matches the empty string"),
                    "A token consumes at least one character: repeat with `+` rather than `*`.",
                )
            yield token_type, regexp

    compiled = tuple(compile_specs())
    unknown = set(skip) - {token_type for token_type, _ in compiled}
    if unknown:
        raise with_hint(
            ValueError(f"skip names token types no specification has: {sorted(unknown)}"),
            "Name the token types of the specs, as a collection of strings: `skip={'space'}`.",
        )
    return Tokenizer(compiled, frozenset(skip))
