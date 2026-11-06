"""Structured Field Values for HTTP, RFC 9651.

Goal: an implementation of the RFC that passes the test suite of the HTTP working group.

A new HTTP header says which of three shapes its value has, a list, a dictionary or an item, and
leaves the syntax to this RFC. It has one reading of every value and fails on anything else: there
are no empty elements to skip and no trailing comma to forgive.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from funcparserlib.parser import (
    CompiledParser,
    Invalid,
    Parser,
    finished,
    lit,
    many,
    maybe,
    regex,
    separated,
)


# --8<-- [start:tree]
@dataclass(frozen=True, slots=True)
class Token:
    """A short name that is not a string, such as `foo123/456`."""

    value: str


@dataclass(frozen=True, slots=True)
class Date:
    """A number of seconds since the Unix epoch, `@1659578233`."""

    seconds: int


@dataclass(frozen=True, slots=True)
class DisplayString:
    """Unicode text for a person to read, `%"calf%c3%a9"`."""

    value: str


type Bare = int | Decimal | str | Token | bytes | bool | Date | DisplayString
type Parameters = Mapping[str, Bare]


@dataclass(frozen=True, slots=True)
class Item:
    """A value with its parameters."""

    value: Bare
    parameters: Parameters


@dataclass(frozen=True, slots=True)
class InnerList:
    """Items in parentheses, with the parameters of the whole list."""

    items: Sequence[Item]
    parameters: Parameters


type Member = Item | InnerList
# --8<-- [end:tree]


# --8<-- [start:values]
def valid_base64(text: str | None) -> bytes:
    """The bytes of a binary value, which an alphabet alone does not make valid."""
    try:
        return base64.b64decode(text or "", validate=True)
    except binascii.Error as e:
        raise Invalid(str(e)) from None


def valid_utf8(parts: list[bytes]) -> DisplayString:
    """The text of a display string, once its escapes are bytes."""
    try:
        return DisplayString(b"".join(parts).decode("utf-8"))
    except UnicodeDecodeError as e:
        raise Invalid(f"not UTF-8: {e.reason}") from None


integer = regex("-?[0-9]{1,15}").named("integer") >> int
decimal = regex(r"-?[0-9]{1,12}\.[0-9]{1,3}").named("decimal") >> Decimal

text = regex(r"[\x20\x21\x23-\x5b\x5d-\x7e]+").named("text")
string_escape = regex(r'\\["\\]').named("escape") >> (lambda s: s[1])
string = (-lit('"') + many(text | string_escape) + -lit('"') >> "".join).context("string")

token = regex(r"[A-Za-z*][!#$%&'*+\-.^_`|~0-9A-Za-z:/]*").named("token") >> Token
base64_text = regex("[A-Za-z0-9+/=]+").named("base64")
binary = -lit(":") + maybe(base64_text).validated(valid_base64) + -lit(":")
boolean = regex("[?][01]").named("boolean") >> (lambda s: s == "?1")
date = -lit("@") + integer >> Date

display_text = regex(r"[\x20\x21\x23\x24\x26-\x7e]+").named("text") >> str.encode
display_escape = regex("%[0-9a-f]{2}").named("escape") >> (lambda s: bytes.fromhex(s[1:]))
display = (-lit('%"') + many(display_text | display_escape) + -lit('"')).validated(valid_utf8)

# A decimal starts as an integer does, so it is tried first.
bare: Parser[str, Bare] = decimal | integer | string | token | binary | boolean | date | display
# --8<-- [end:values]


# --8<-- [start:grammar]
def true_item(params: Parameters) -> Member:
    """The member of a key with no value."""
    return Item(True, params)


spaces = regex(" +")
sp = -maybe(spaces)
key = regex(r"[a-z*][a-z0-9_\-.*]*").named("key")

# A key alone means true.
key_value = (key + (maybe(-lit("=") + bare) >> (lambda v: True if v is None else v))).starmap(
    lambda k, v: (k, v)
)
parameters = many(-lit(";") + sp + key_value) >> dict

item = bare + parameters >> (lambda r: Item(r[0], r[1]))
inner_list = -lit("(") + sp + (separated(item, spaces) + sp) + -lit(")") + parameters >> (
    lambda r: InnerList(r[0], r[1])
)
member: Parser[str, Member] = inner_list | item

ows = -maybe(regex(r"[ \t]+").named("whitespace"))
comma = ows + -lit(",") + ows
# `a=1;x=2` is a member, and `a;x=2` is a true with parameters.
true_with = parameters >> true_item
dictionary_member = (key + (-lit("=") + member | true_with)).starmap(lambda k, v: (k, v))

item_field = sp + item + sp + -finished
list_field = sp + separated(member, comma) + sp + -finished
dictionary_field = sp + separated(dictionary_member, comma) + sp + -finished >> dict
# --8<-- [end:grammar]

# --8<-- [start:parse]
compiled_item: CompiledParser[str, Item] = item_field.compile()
compiled_list: CompiledParser[str, list[Member]] = list_field.compile()
compiled_dictionary: CompiledParser[str, dict[str, Member]] = dictionary_field.compile()


def parse_item(field: str) -> Item:
    """The item of a field. A field sent on several lines is joined with `", "` first.

    Raises `NoParseError` or `ParseErrors`, and the RFC then asks to ignore the whole field.
    """
    return compiled_item.parse_text(field)


def parse_list(field: str) -> list[Member]:
    """The members of a field, as `parse_item()` does."""
    return compiled_list.parse_text(field)


def parse_dictionary(field: str) -> dict[str, Member]:
    """The members of a field by their keys, as `parse_item()` does."""
    return compiled_dictionary.parse_text(field)


# --8<-- [end:parse]
