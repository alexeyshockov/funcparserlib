"""A MessagePack parser, written from the format specification.

MessagePack is a binary format: every value starts with a byte that says its type, and often holds a
length or a count that says what follows. The grammar is a table of those bytes. `fixed()` reads a
number, `prefixed()` reads as many bytes as a length says, `counted()` as many values as a count
says, and `switch()` chooses what to read by the type byte, so the grammar compiles. `parse()` gives
what `msgpack.unpackb(data, raw=False, strict_map_key=False, timestamp=0)` gives: `None`, `bool`,
`int`, `float`, `str`, `bytes`, `list`, `dict`, `Timestamp` and `Ext` for an extension type.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from typing import Any, NamedTuple

from funcparserlib import binary as b
from funcparserlib.parser import (
    CompiledParser,
    ForwardDeclParser,
    Parser,
    counted,
    finished,
    forward_decl,
    prefixed,
    pure,
    switch,
)

# --8<-- [start:tree]
type Value = (
    bool | int | float | str | bytes | Ext | Timestamp | list[Value] | dict[Any, Value] | None
)


class Ext(NamedTuple):
    """A value of an application's own type: a type number from -128 to 127, and its bytes."""

    type: int
    data: bytes


class Timestamp(NamedTuple):
    """A point in time, the extension of type -1: whole seconds since the epoch, and nanoseconds."""

    seconds: int
    nanoseconds: int


# --8<-- [end:tree]

# --8<-- [start:numbers]
u8 = b.fixed(">B").named("uint8")
u16 = b.fixed(">H").named("uint16")
u32 = b.fixed(">I").named("uint32")
u64 = b.fixed(">Q").named("uint64")
i8 = b.fixed(">b").named("int8")
i16 = b.fixed(">h").named("int16")
i32 = b.fixed(">i").named("int32")
i64 = b.fixed(">q").named("int64")
f32 = b.fixed(">f").named("float32")
f64 = b.fixed(">d").named("float64")
# --8<-- [end:numbers]

# --8<-- [start:recursion]
value: ForwardDeclParser[int, Value] = forward_decl().named("MessagePack value")
# --8<-- [end:recursion]


# --8<-- [start:lengths]
def text(length: Parser[int, int]) -> Parser[int, str]:
    return prefixed(length) >> bytes.decode


def array(length: Parser[int, int]) -> Parser[int, list[Value]]:
    return counted(length, value)


def table(length: Parser[int, int]) -> Parser[int, dict[Any, Value]]:
    return counted(length, value + value) >> dict


# --8<-- [end:lengths]


# --8<-- [start:tags]
# A type byte holds a number in its low bits, and `previous()` reads it back under a mask.
low4, low5, low7 = b.previous(0x0F), b.previous(0x1F), b.previous(0x7F)


def negative(n: int) -> int:
    return n - 256


# --8<-- [end:tags]


# --8<-- [start:extensions]
def timestamp(data: bytes) -> Timestamp:
    """The three layouts the specification gives a timestamp: 4, 8 or 12 bytes."""
    match len(data):
        case 4:
            seconds, nanoseconds = int.from_bytes(data), 0
        case 8:
            packed = int.from_bytes(data)
            seconds, nanoseconds = packed & (1 << 34) - 1, packed >> 34
        case 12:
            nanoseconds, seconds = struct.unpack(">Iq", data)
        case size:
            raise ValueError(f"invalid timestamp data (length {size})")
    if nanoseconds >= 10**9:
        raise ValueError("nanoseconds must be less than 1000000000")
    return Timestamp(seconds, nanoseconds)


def extension(type_: int, data: bytes) -> Ext | Timestamp:
    return timestamp(data) if type_ == -1 else Ext(type_, data)


# A fixed extension has the size in its type byte, and the type number comes before the data.
def fixext(size: int) -> Parser[int, Ext | Timestamp]:
    return i8 + prefixed(pure(size)) >> (lambda r: extension(*r))


# The other extensions have a length, then the type number, then as many bytes as the length says.
# The length is read before the parser that uses it exists, which takes `bind()`.
def ext(length: Parser[int, int]) -> Parser[int, Ext | Timestamp]:
    def data(r: tuple[int, int]) -> Parser[int, Ext | Timestamp]:
        size, type_ = r
        return prefixed(pure(size)) >> (lambda data: extension(type_, data))

    return (length + i8).bind(data)


# --8<-- [end:extensions]

# --8<-- [start:grammar]
cases: Mapping[int | range, Parser[int, Any]] = {
    range(0x80): low7,
    range(0x80, 0x90): table(low4),
    range(0x90, 0xA0): array(low4),
    range(0xA0, 0xC0): text(low5),
    0xC0: pure(None),
    0xC2: pure(False),
    0xC3: pure(True),
    0xC4: prefixed(u8),
    0xC5: prefixed(u16),
    0xC6: prefixed(u32),
    0xC7: ext(u8),
    0xC8: ext(u16),
    0xC9: ext(u32),
    0xCA: f32,
    0xCB: f64,
    0xCC: u8,
    0xCD: u16,
    0xCE: u32,
    0xCF: u64,
    0xD0: i8,
    0xD1: i16,
    0xD2: i32,
    0xD3: i64,
    0xD4: fixext(1),
    0xD5: fixext(2),
    0xD6: fixext(4),
    0xD7: fixext(8),
    0xD8: fixext(16),
    0xD9: text(u8),
    0xDA: text(u16),
    0xDB: text(u32),
    0xDC: array(u16),
    0xDD: array(u32),
    0xDE: table(u16),
    0xDF: table(u32),
    range(0xE0, 0x100): b.previous() >> negative,
}

value.define(switch(b.byte(), cases).named("MessagePack type"))
document = value + -finished
# --8<-- [end:grammar]

# --8<-- [start:parse]
compiled: CompiledParser[int, Value] = document.compile()


def parse(data: bytes) -> Value:
    """The value that `data`, one MessagePack document, holds.

    An extension of a type below -1 is an `Ext`, which `msgpack.unpackb()` rejects. Raises
    `NoParseError` for bytes that are not one: an unknown type byte, a length that runs past the
    end, bytes left after the value. A string that is not UTF-8 raises `UnicodeDecodeError`, and a
    map key that is an array or a map raises `TypeError`, as `msgpack.unpackb()` does.
    """
    return compiled.parse_bytes(data)


# --8<-- [end:parse]
