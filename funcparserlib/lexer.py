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

__all__ = ["make_tokenizer", "TokenSpec", "Token", "LexerError", "Place", "Spec"]

import dataclasses as dc
import re
from collections.abc import Callable, Iterable, Sequence
from re import Pattern
from typing import Any, cast, final

from ._commons import valueclass, with_hint

type Place = tuple[int, int]
"""A position in the text: (_line_, _column_), both counted from 1."""

type Spec = tuple[str, tuple[Any, ...]]
"""A legacy token spec: (_type_, _args_), where _args_ are the arguments of `re.compile()`."""

_CompiledSpec = tuple[str, Pattern[str]]

# The flags a group can scope, `(?i:...)`; any other flag keeps a spec out of a joined pattern.
_NAMED_GROUP = re.compile(r"\(\?P<\w+>")
_SCOPED_FLAGS = {re.IGNORECASE: "i", re.MULTILINE: "m", re.DOTALL: "s", re.VERBOSE: "x"}


def _combined(specs: Sequence[_CompiledSpec]) -> Pattern[str] | None:
    """One pattern for `specs`, its group `t<n>` matching where the n-th spec is the first to match.

    A spec's named groups become non-capturing, as a token is the whole match. `None` when a spec
    cannot be joined: it refers to a group, which joining would renumber or rename, or it has a flag
    a group cannot scope.
    """
    parts = []
    for n, (_, regexp) in enumerate(specs):
        pattern = regexp.pattern
        if re.search(r"\\[1-9]|\(\?P=|\(\?\(", pattern):
            return None
        # Each `(?P<` is a group only if they number as many as the groups the pattern names.
        if len(_NAMED_GROUP.findall(pattern)) != len(regexp.groupindex):
            return None
        pattern = _NAMED_GROUP.sub("(?:", pattern)
        flags = regexp.flags & ~re.UNICODE
        letters = "".join(letter for flag, letter in _SCOPED_FLAGS.items() if flags & flag)
        if flags & ~sum(_SCOPED_FLAGS):
            return None
        # A verbose pattern may end in a comment, which a newline closes.
        end = "\n" if flags & re.VERBOSE else ""
        parts.append(f"(?P<t{n}>(?{letters}:{pattern}{end}))")
    try:
        return re.compile("|".join(parts))
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
        line, pos = self.place
        return f'cannot tokenize data: {line},{pos}: "{self.msg}"'


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
        return f"Token({self.type!r}, {self.value!r})"

    def _pos_str(self) -> str:
        if self.start is None or self.end is None:
            return ""
        s_line, s_pos = self.start
        e_line, e_pos = self.end
        return f"{s_line},{s_pos}-{e_line},{e_pos}:"

    def __str__(self) -> str:
        s = f"{self._pos_str()} {self.type} '{self.value}'"
        return s.strip()

    @property
    def name(self) -> str:
        """The token's value; parsers use it to name a parser made from a token."""
        return self.value

    def pformat(self) -> str:
        """The token in aligned columns: position, type and value."""
        return f"{self._pos_str().ljust(20)} {self.type.ljust(14)} '{self.value}'"


def make_tokenizer(specs: Sequence[TokenSpec | Spec]) -> Callable[[str], Iterable[Token]]:
    # noinspection GrazieInspection
    """Make a function that tokenizes text based on the regexp specs.

    Type: `(Sequence[TokenSpec | Tuple]) -> Callable[[str], Iterable[Token]]`

    A token spec is `TokenSpec` instance.

    !!! Note

        For legacy reasons, a token spec may also be a tuple of (_type_, _args_), where
        _type_ sets the value of `Token.type` for the token, and _args_ are the
        positional arguments for `re.compile()`: either just (_pattern_,) or
        (_pattern_, _flags_).

    It returns a tokenizer function that takes a string and returns an iterable of `Token` objects,
    or raises `LexerError` if it cannot tokenize the string according to its token specs.

    Examples:
        ```pycon
        >>> tokenize = make_tokenizer([
        ...     TokenSpec("space", r"\\s+"),
        ...     TokenSpec("id", r"\\w+"),
        ...     TokenSpec("op", r"[,!]"),
        ... ])
        >>> text = "Hello, World!"
        >>> [t for t in tokenize(text) if t.type != "space"]  # noqa
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
    combined = _combined(compiled)
    types = {f"t{n}": token_type for n, (token_type, _) in enumerate(compiled)}

    def match_specs(s: str, i: int, position: Place) -> Token:
        line, pos = position
        m: re.Match[str] | None = None
        token_type = ""
        if combined is not None:
            if m := combined.match(s, i):
                token_type = types[cast("str", m.lastgroup)]
        else:
            for token_type, spec_regexp in compiled:  # noqa: B007 — the matching spec's type
                if m := spec_regexp.match(s, i):
                    break
        if m is not None:
            value = m.group()
            if not value:
                raise ValueError(
                    f"the pattern of token {token_type!r} matched the empty string"
                    f" at {line},{pos + 1}"
                )
            nls = value.count("\n")
            n_line = line + nls
            n_pos = pos + len(value) if nls == 0 else len(value) - value.rfind("\n") - 1
            return Token(token_type, value, (line, pos + 1), (n_line, n_pos))
        # Lines are counted by "\n" alone, as the token positions are.
        err_line = s.split("\n")[line - 1].rstrip("\r")
        raise LexerError((line, pos + 1), err_line)

    def f(s: str) -> Iterable[Token]:
        length = len(s)
        line, pos = 1, 0
        i = 0
        while i < length:
            t = match_specs(s, i, (line, pos))
            yield t
            if t.end is None:
                raise ValueError(f"token {t!r} has no end position")
            line, pos = t.end
            i += len(t.value)

    return f


# This is an example of token specs. See also [this article][1] for a
# discussion of searching for multiline comments using regexps (including `*?`).
#
#   [1]: http://ostermiller.org/findcomment.html
_example_token_specs = [
    TokenSpec("COMMENT", r"\(\*(.|[\r\n])*?\*\)", re.MULTILINE),
    TokenSpec("COMMENT", r"\{(.|[\r\n])*?\}", re.MULTILINE),
    TokenSpec("COMMENT", r"//.*"),
    TokenSpec("NL", r"[\r\n]+"),
    TokenSpec("SPACE", r"[ \t\r\n]+"),
    TokenSpec("NAME", r"[A-Za-z_][A-Za-z_0-9]*"),
    TokenSpec("REAL", r"[0-9]+\.[0-9]*([Ee][+\-]?[0-9]+)*"),
    TokenSpec("INT", r"[0-9]+"),
    TokenSpec("INT", r"\$[0-9A-Fa-f]+"),
    TokenSpec("OP", r"(\.\.)|(<>)|(<=)|(>=)|(:=)|[;,=\(\):\[\]\.+\-<>\*/@\^]"),
    TokenSpec("STRING", r"'([^']|(''))*'"),
    TokenSpec("CHAR", r"#[0-9]+"),
    TokenSpec("CHAR", r"#\$[0-9A-Fa-f]+"),
]
# tokenize = make_tokenizer(_example_token_specs)
