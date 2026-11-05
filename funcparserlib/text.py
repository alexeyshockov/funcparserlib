"""Parsers that read text at the position the parsing has reached, without a list of tokens.

A grammar of `Text` parsers asks for the tokens that make sense where it is, so there is no list of
token specs to keep in order, and what a token means can depend on where it is. Its errors come from
the grammar: they say what was expected, with the line and column of the text. See
`Parser.parse_text()` and `Parser.compile()`.
"""

__all__ = ["Text"]

import re
from re import Pattern
from typing import Any, final

from ._commons import with_hint
from .parser import Parser, ParsingError, ParsingResult, ParsingSuccess, State


@final
class Text:
    """Makes parsers of text, each skipping what `skip` matches after what it reads.

    `skip` is a regexp for what may sit between tokens, such as whitespace and comments. Inside a
    token, such as a string written as a grammar, use a `Text` without `skip`.

    Raises `ValueError` for a `skip` that matches the empty string.

    Examples:
        ```pycon
        >>> from funcparserlib.parser import many
        >>> from funcparserlib.text import Text
        >>> text = Text(skip=r"\\s+")
        >>> number = (text.token(r"\\d+") >> int).named("number")
        >>> total = number + many(-text.literal("+") + number) + -text.end
        >>> total.parse_text(" 1 + 2 ")
        (1, [2])
        >>> total.parse_text("1 +\\n  x")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: 2,3-2,3: got unexpected token: 'x', expected: number

        ```
    """

    def __init__(self, skip: str | None = None, flags: int = 0) -> None:
        """Make parsers that skip what the regexp `skip`, with its `flags`, matches."""
        self._skip = None
        if skip is not None:
            regexp = re.compile(skip, flags)
            if regexp.match(""):
                raise with_hint(
                    ValueError(f"skip pattern {skip!r} matches the empty string"),
                    "A skip pattern consumes at least one character: repeat with `+`.",
                )
            self._skip = regexp.match

    def _after_skip(self, text: str, pos: int) -> int:
        if self._skip is not None and (m := self._skip(text, pos)):
            return m.end()
        return pos

    def _start(self, text: str, pos: int) -> int:
        """Where a token at `pos` starts: nothing has skipped what comes before the first one."""
        return self._after_skip(text, pos) if pos == 0 else pos

    def _reading(
        self, name: str, read: Any, regexp: Pattern[str] | None, value: str | None
    ) -> Parser[str, str]:
        def run(text: Any, s: State) -> ParsingResult[str]:
            start = self._start(text, s.pos)
            end = read(text, start)
            if end is None:
                return ParsingError(s.reset_pos(start).expect(p))
            return ParsingSuccess(text[start:end], s.reset_pos(self._after_skip(text, end)))

        p: Parser[Any, str] = Parser(run).named(name)
        p._leaf = True
        p._form = ("text", regexp.match if regexp else None, value, self._skip)
        return p

    def token(self, pattern: str, flags: int = 0) -> Parser[str, str]:
        """A parser of the text the regexp `pattern`, with its `flags`, matches.

        Give it a name with `Parser.named()` for its errors. Raises `ValueError` for a pattern that
        matches the empty string.
        """
        regexp = re.compile(pattern, flags)
        if regexp.match(""):
            raise with_hint(
                ValueError(f"token pattern {pattern!r} matches the empty string"),
                "A token consumes at least one character: repeat with `+` rather than `*`.",
            )

        def read(text: str, start: int) -> int | None:
            m = regexp.match(text, start)
            return m.end() if m is not None and m.end() > start else None

        return self._reading(f"/{pattern}/", read, regexp, None)

    def literal(self, value: str) -> Parser[str, str]:
        """A parser of the text `value`, which it reads where a longer word starts with it too.

        To read a keyword, end a `Text.token()` with a word boundary: `text.token(r"if\\\\b")`.
        Raises `ValueError` for an empty `value`.
        """
        if not value:
            raise with_hint(
                ValueError("literal value is empty"), "A token consumes at least one character."
            )

        def read(text: str, start: int) -> int | None:
            return start + len(value) if text.startswith(value, start) else None

        return self._reading(repr(value), read, None, value)

    @property
    def end(self) -> Parser[str, None]:
        """A parser that succeeds at the end of the text, past what `skip` matches."""

        def run(text: Any, s: State) -> ParsingResult[None]:
            start = self._start(text, s.pos)
            if start >= len(text):
                return ParsingSuccess(None, s.reset_pos(start))
            return ParsingError(s.reset_pos(start).expect(p))

        p: Parser[Any, None] = Parser(run).named("end of input")
        p._leaf = True
        p._form = ("text_end", self._skip)
        return p
