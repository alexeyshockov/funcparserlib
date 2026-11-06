"""A URI reference by the grammar of RFC 3986, and a query string read strictly.

Goal: refuse what is not a URI or a well-formed query, and say where, rather than guess a reading.
What loses nothing stays valid, such as an `&` at the end of a query. What would lose data or make
it up is an error: `=v` has a value nothing can look up, and `%zz` stands for no text.

Every rule of the appendix of the RFC is here, as a parser or as a regex for what a rule repeats.
`parse()` says where a text stops being a URI, and what kind of host it names.
`urllib.parse.urlsplit` splits the same text the same way, and takes almost any text as a URL.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Literal

from funcparserlib.parser import (
    CompiledParser,
    Invalid,
    Parser,
    finished,
    lit,
    longest,
    many,
    maybe,
    oneplus,
    pure,
    regex,
    separated,
)


# --8<-- [start:tree]
@dataclass(frozen=True, slots=True)
class Authority:
    """Who holds the names under a URI: `userinfo@host:port`."""

    userinfo: str | None
    host: str
    kind: Literal["ipv4", "ipv6", "ipvfuture", "name"]
    port: int | None


@dataclass(frozen=True, slots=True)
class Uri:
    """A URI reference, each part as it was written, and `None` for a part that is absent."""

    scheme: str | None
    authority: Authority | None
    path: str
    query: str | None
    fragment: str | None


# --8<-- [end:tree]

# --8<-- [start:characters]
UNRESERVED = r"A-Za-z0-9\-._~"
SUB_DELIMS = r"!$&'()*+,;="
PERCENT = "%[0-9A-Fa-f]{2}"


def chars(extra: str = "") -> str:
    """A regex of one character of a URI: unreserved, a sub-delimiter, `extra`, or an escape."""
    return rf"(?:[{UNRESERVED}{SUB_DELIMS}{extra}]|{PERCENT})"


PCHAR = chars(":@")
# --8<-- [end:characters]


# --8<-- [start:authority]
def valid_port(digits: str) -> int:
    """The number of a port, which has two bytes on the wire."""
    if int(digits) > 65535:
        raise Invalid(f"port {digits} is above 65535")
    return int(digits)


def valid_ipv6(text: str) -> str:
    """The text of an IPv6 address, for `ipaddress` to check."""
    try:
        ipaddress.IPv6Address(text)
    except ValueError as e:
        raise Invalid(str(e)) from None
    return text


def host_of(kind: str) -> Any:
    """A function that labels the text of a host with its `kind`."""
    return lambda text: (text, kind)


OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
ipv4 = regex(rf"{OCTET}(?:\.{OCTET}){{3}}").named("IPv4 address") >> host_of("ipv4")
name = regex(chars() + "+").named("host name") >> host_of("name")
ipv6 = regex("[0-9A-Fa-f:.]+").named("IPv6 address").validated(valid_ipv6) >> host_of("ipv6")
ipvfuture = regex(rf"v[0-9A-Fa-f]+\.[{UNRESERVED}{SUB_DELIMS}:]+").named("IP literal") >> host_of(
    "ipvfuture"
)
ip_literal = -lit("[") + (ipvfuture | ipv6) + -lit("]")
# `1.2.3.4.example` starts as an address and is a name, which only the longer match can tell.
host = ip_literal | longest(ipv4, name)

userinfo = regex(chars(":") + "+").named("userinfo")
# A userinfo reads what a host reads, so only the `@` tells them apart: without one, `maybe()` goes
# back to where it started.
user = maybe(maybe(userinfo) + -lit("@") >> (lambda text: text or ""))
port = -lit(":") + maybe(regex("[0-9]+").named("port").validated(valid_port))


def make_authority(parts: tuple[Any, ...]) -> Authority:
    """An authority from its userinfo, host and port, the host empty in `file:///etc/hosts`."""
    info, (text, kind), number = parts
    return Authority(info, text, kind, number)


empty_host = pure(("", "name"))
authority = user + (host | empty_host) + maybe(port) >> make_authority
# --8<-- [end:authority]

# --8<-- [start:reference]
scheme = regex(r"[A-Za-z][A-Za-z0-9+.\-]*").named("scheme") + -lit(":")

path_abempty = many(regex(f"/{PCHAR}*").named("path")) >> "".join
path_absolute = regex(f"/(?:{PCHAR}+(?:/{PCHAR}*)*)?").named("path")
path_rootless = regex(f"{PCHAR}+(?:/{PCHAR}*)*").named("path")
# The first segment has no `:`, or `a:b` would be a path as well as a scheme and a path.
path_noscheme = regex(f"{chars('@')}+(?:/{PCHAR}*)*").named("path")


def no_authority(path: Parser[str, str]) -> Parser[str, tuple[None, str]]:
    """The two parts after a scheme when `path` is all there is."""
    return path >> (lambda text: (None, text))


with_authority = (-lit("//") + authority + path_abempty).starmap(lambda a, p: (a, p))
hier_part = with_authority | no_authority(path_absolute | path_rootless | pure(""))
relative_part = with_authority | no_authority(path_absolute | path_noscheme | pure(""))

after = regex(chars(":@/?") + "+").named("text")
query = -lit("?") + (maybe(after) >> (lambda text: text or ""))
fragment = -lit("#") + (maybe(after) >> (lambda text: text or ""))
tail = maybe(query) + maybe(fragment)

absolute = scheme + hier_part + tail >> (lambda r: Uri(r[0], *r[1], *r[2]))
relative = relative_part + tail >> (lambda r: Uri(None, *r[0], *r[1]))
reference: Parser[str, Uri] = absolute | relative
# --8<-- [end:reference]


# --8<-- [start:query]
def utf8(parts: list[bytes]) -> str:
    """The text of a name or a value, once its escapes are bytes."""
    try:
        return b"".join(parts).decode("utf-8")
    except UnicodeDecodeError as e:
        raise Invalid(f"not UTF-8 once percent-decoded: {e.reason}") from None


escape = regex("%[0-9A-Fa-f]{2}").named("escape") >> (lambda text: bytes.fromhex(text[1:]))
space = lit("+") >> (lambda _: b" ")


def piece(ends: str) -> Parser[str, bytes]:
    """A part of a name or a value, which `ends` and `&`, `%` and `+` end."""
    return regex(rf"[^&%+\ud800-\udfff{ends}]+").named("text") >> str.encode | escape | space


pair_name = oneplus(piece("=")).validated(utf8)
pair_value = many(piece("")).validated(utf8)
pair = (pair_name + (maybe(-lit("=") + pair_value) >> (lambda text: text or ""))).starmap(
    lambda k, v: (k, v)
)
# The URL Standard skips an empty pair, so an `&` can come first, last or twice.
ampersands = oneplus(lit("&"))
pairs: Parser[str, list[tuple[str, str]]] = -many(lit("&")) + separated(
    pair, ampersands, trailing=True
)
# --8<-- [end:query]

# --8<-- [start:parse]
compiled_reference: CompiledParser[str, Uri] = (reference + -finished).compile()
compiled_pairs: CompiledParser[str, list[tuple[str, str]]] = (pairs + -finished).compile()


def parse(text: str) -> Uri:
    """The parts of the URI reference `text`.

    Raises `NoParseError` for text that is not a URI reference, and `ParseErrors` for a port or an
    IPv6 address that is out of range.
    """
    return compiled_reference.parse_text(text)


def parse_query(text: str) -> list[tuple[str, str]]:
    """The names and values of the query `text`, decoded, in their order.

    Raises `NoParseError` for an `%` without two hex digits or a pair without a name, and
    `ParseErrors` for a name or a value that is not UTF-8.
    """
    return compiled_pairs.parse_text(text)


# --8<-- [end:parse]

if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(parse(sys.argv[1]))  # noqa: T203 — the program's output
