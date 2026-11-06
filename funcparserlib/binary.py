"""Parsers that read bytes at the position the parsing has reached.

These parsers read a binary format as the parsers of `funcparserlib.text` read text. Run a grammar
of them with `Parser.parse_bytes()`, or compile it with `Parser.compile()`, and end it with
`finished`. `prefixed()`, `counted()` and `switch()` from `funcparserlib.parser` read what depends
on a number just read, and generated code runs them, as it does not `Parser.bind()`.

Examples:
    ```pycon
    >>> from funcparserlib import binary as b
    >>> from funcparserlib.parser import counted, finished, prefixed, switch
    >>> cases = {1: b.fixed(">H"), 2: prefixed(b.byte())}
    >>> record = switch(b.byte(), cases).named("item")
    >>> many_records = counted(b.byte(), record) + -finished
    >>> many_records.parse_bytes(b"\\x02\\x01\\x00\\x07\\x02\\x02hi")
    [7, b'hi']
    >>> many_records.parse_bytes(b"\\x01\\x09")
    Traceback (most recent call last):
        ...
    funcparserlib.parser.NoParseError: got unexpected token: 0x09 at offset 0x1, expected: item

    ```
"""

__all__ = ["byte", "fixed", "lit", "previous", "regex"]

import re
import struct
from typing import Any

from ._commons import with_hint
from .parser import Parser, ParsingFailure, ParsingResult, ParsingSuccess, State


def regex(pattern: bytes, flags: int = 0) -> Parser[int, bytes]:
    """A parser of the bytes the regex `pattern`, with its `flags`, matches.

    Give it a name with `Parser.named()` for its errors. Raises `ValueError` for a pattern that
    matches no bytes.
    """
    regexp = re.compile(pattern, flags)
    if regexp.match(b""):
        raise with_hint(
            ValueError(f"pattern {pattern!r} matches no bytes"),
            "A match consumes at least one byte: repeat with `+` rather than `*`.",
        )

    def run(data: Any, s: State) -> ParsingResult[bytes]:
        m = regexp.match(data, s.pos)
        if m is None or m.end() == s.pos:
            return ParsingFailure(s.expect(p))
        return ParsingSuccess(m.group(), s.reset_pos(m.end()))

    p = Parser._made_leaf(repr(pattern), run, ("text", regexp.match, None, None))
    return p


def lit(value: bytes) -> Parser[int, bytes]:
    """A parser of the bytes `value`.

    Raises `ValueError` for an empty `value`, and `TypeError` for anything but `bytes`.
    """
    if not isinstance(value, bytes):
        raise with_hint(
            TypeError(f"literal value must be bytes, not {type(value).__name__}"),
            "Pass the bytes to read, as `binary.lit(b'ab')`.",
        )
    if not value:
        raise with_hint(
            ValueError("literal value is empty"), "A literal consumes at least one byte."
        )

    def run(data: Any, s: State) -> ParsingResult[bytes]:
        if data.startswith(value, s.pos):
            return ParsingSuccess(value, s.reset_pos(s.pos + len(value)))
        return ParsingFailure(s.expect(p))

    p = Parser._made_leaf(repr(value), run, ("text", None, value, None))
    return p


def fixed(fmt: str) -> Parser[int, Any]:
    """A parser of the one value the `struct` format `fmt` reads, such as `">I"` or `"<d"`.

    Raises `ValueError` for an invalid format, and one of anything but a single value that takes at
    least one byte.
    """
    try:
        layout = struct.Struct(fmt)
    except struct.error as e:
        raise ValueError(f"struct format {fmt!r} is invalid: {e}") from e
    if layout.size == 0 or len(layout.unpack(bytes(layout.size))) != 1:
        raise with_hint(
            ValueError(f"struct format {fmt!r} reads no single value"),
            "Read each value with a `binary.fixed()` of its own, and join them with `+`.",
        )
    unpack, size = layout.unpack_from, layout.size

    def run(data: Any, s: State) -> ParsingResult[Any]:
        if s.pos + size > len(data):
            return ParsingFailure(s.expect(p), "not enough bytes")
        return ParsingSuccess(unpack(data, s.pos)[0], s.reset_pos(s.pos + size))

    p = Parser._made_leaf(f"{size} byte{'s' if size > 1 else ''}", run, ("fixed", unpack, size))
    return p


def byte() -> Parser[int, int]:
    """A parser of one byte, as an `int`."""

    def run(data: Any, s: State) -> ParsingResult[int]:
        if s.pos >= len(data):
            return ParsingFailure(s.expect(p), "not enough bytes")
        return ParsingSuccess(data[s.pos], s.reset_pos(s.pos + 1))

    p = Parser._made_leaf("a byte", run, ("byte",))
    return p


def previous(mask: int = 0xFF) -> Parser[int, int]:
    """A parser of the byte just read, under `mask`, which reads nothing.

    It is for a tag that holds a value, in a case of `switch()`:
    `counted(b.previous(0x0F), item)`. It fails at the start of the input.
    """

    def run(data: Any, s: State) -> ParsingResult[int]:
        if s.pos == 0:
            return ParsingFailure(s.expect(p))
        return ParsingSuccess(data[s.pos - 1] & mask, s)

    p = Parser._made_leaf("the byte before", run, ("previous", mask))
    return p
