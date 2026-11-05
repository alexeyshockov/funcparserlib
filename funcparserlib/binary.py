"""Parsers that read bytes at the position the parsing has reached.

A grammar of `Binary` parsers reads a binary format as a `funcparserlib.text.Text` grammar reads
text. Run it with `Parser.parse_bytes()`, or compile it with `Parser.compile()`. `prefixed()`,
`counted()` and `switch()` from `funcparserlib.parser` read what depends on a number just read, and
generated code runs them, as it does not `Parser.bind()`.
"""

__all__ = ["Binary"]

import re
import struct
from typing import Any, final

from ._commons import with_hint
from .parser import Parser, ParsingError, ParsingResult, ParsingSuccess, State


@final
class Binary:
    """Makes parsers of bytes.

    Examples:
        ```pycon
        >>> from funcparserlib.binary import Binary
        >>> from funcparserlib.parser import counted, prefixed, switch
        >>> b = Binary()
        >>> cases = {1: b.fixed(">H"), 2: prefixed(b.byte)}
        >>> record = switch(b.byte, cases).named("item")
        >>> many_records = counted(b.byte, record) + -b.end
        >>> many_records.parse_bytes(b"\\x02\\x01\\x00\\x07\\x02\\x02hi")
        [7, b'hi']
        >>> many_records.parse_bytes(b"\\x01\\x09")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: 0x09 at offset 0x1, expected: item

        ```
    """

    @staticmethod
    def _leaf(name: str, run: Any, form: tuple[Any, ...]) -> Parser[Any, Any]:
        p: Parser[Any, Any] = Parser(run).named(name)
        p._leaf = True
        p._form = form
        return p

    def token(self, pattern: bytes, flags: int = 0) -> Parser[int, bytes]:
        """A parser of the bytes the regexp `pattern`, with its `flags`, matches.

        Give it a name with `Parser.named()` for its errors. Raises `ValueError` for a pattern that
        matches no bytes.
        """
        regexp = re.compile(pattern, flags)
        if regexp.match(b""):
            raise with_hint(
                ValueError(f"token pattern {pattern!r} matches no bytes"),
                "A token consumes at least one byte: repeat with `+` rather than `*`.",
            )

        def run(data: Any, s: State) -> ParsingResult[bytes]:
            m = regexp.match(data, s.pos)
            if m is None or m.end() == s.pos:
                return ParsingError(s.expect(p))
            return ParsingSuccess(m.group(), s.reset_pos(m.end()))

        p = self._leaf(repr(pattern), run, ("text", regexp.match, None, None))
        return p

    def literal(self, value: bytes) -> Parser[int, bytes]:
        """A parser of the bytes `value`. Raises `ValueError` for an empty `value`."""
        if not value:
            raise with_hint(
                ValueError("literal value is empty"), "A token consumes at least one byte."
            )

        def run(data: Any, s: State) -> ParsingResult[bytes]:
            if data.startswith(value, s.pos):
                return ParsingSuccess(value, s.reset_pos(s.pos + len(value)))
            return ParsingError(s.expect(p))

        p = self._leaf(repr(value), run, ("text", None, value, None))
        return p

    def fixed(self, fmt: str) -> Parser[int, Any]:
        """A parser of the one value the `struct` format `fmt` reads, such as `">I"` or `"<d"`.

        Raises `ValueError` for a format of anything but one value.
        """
        layout = struct.Struct(fmt)
        if len(layout.unpack(bytes(layout.size))) != 1:
            raise with_hint(
                ValueError(f"struct format {fmt!r} reads more than one value"),
                "Read each value with a `Binary.fixed()` of its own, and join them with `+`.",
            )
        unpack, size = layout.unpack_from, layout.size

        def run(data: Any, s: State) -> ParsingResult[Any]:
            if s.pos + size > len(data):
                return ParsingError(s.expect(p), "got unexpected end of input")
            return ParsingSuccess(unpack(data, s.pos)[0], s.reset_pos(s.pos + size))

        p = self._leaf(f"{size} bytes", run, ("fixed", unpack, size))
        return p

    @property
    def byte(self) -> Parser[int, int]:
        """A parser of one byte, as an `int`."""

        def run(data: Any, s: State) -> ParsingResult[int]:
            if s.pos >= len(data):
                return ParsingError(s.expect(p), "got unexpected end of input")
            return ParsingSuccess(data[s.pos], s.reset_pos(s.pos + 1))

        p = self._leaf("a byte", run, ("byte",))
        return p

    def previous(self, mask: int = 0xFF) -> Parser[int, int]:
        """A parser of the byte just read, under `mask`, which reads nothing.

        It is for a tag that holds a value, in a case of `switch()`:
        `counted(b.previous(0x0F), item)`. It fails at the start of the input.
        """

        def run(data: Any, s: State) -> ParsingResult[int]:
            if s.pos == 0:
                return ParsingError(s.expect(p))
            return ParsingSuccess(data[s.pos - 1] & mask, s)

        p = self._leaf("the byte before", run, ("previous", mask))
        return p

    @property
    def end(self) -> Parser[int, None]:
        """A parser that succeeds at the end of the bytes."""

        def run(data: Any, s: State) -> ParsingResult[None]:
            if s.pos >= len(data):
                return ParsingSuccess(None, s)
            return ParsingError(s.expect(p))

        p = self._leaf("end of input", run, ("text_end", None))
        return p
