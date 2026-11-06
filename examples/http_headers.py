"""HTTP header values, from the pieces RFC 9110 shares between them.

Goal: read a header by its grammar, so that what quotes protect stays text and a malformed header is
refused.

A token, a quoted string, parameters and a comma-separated list make most headers, and each header
below is a few lines on top of them. A comma or a semicolon inside quotes stays where it is, which a
split on `,` cannot do, and each field gives what a server or a client would act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from examples.uri import Uri, reference
from funcparserlib.parser import (
    CompiledParser,
    Invalid,
    Parser,
    finished,
    lit,
    many,
    maybe,
    oneplus,
    regex,
    separated,
)

# --8<-- [start:pieces]
TCHAR = r"!#$%&'*+\-.^_`|~0-9A-Za-z"
token = regex(f"[{TCHAR}]+").named("token")
ows = -maybe(regex(r"[ \t]+").named("whitespace"))

qdtext = regex(r"[\t \x21\x23-\x5b\x5d-\x7e\x80-\xff]+").named("text")
quoted_pair = regex(r"\\[\t \x21-\x7e\x80-\xff]").named("escape") >> (lambda text: text[1])
quoted = (-lit('"') + many(qdtext | quoted_pair) + -lit('"') >> "".join).context("quoted string")


name = token >> str.lower
value = token | quoted
parameter = (name + -lit("=") + value).starmap(lambda k, v: (k, v))
semicolon = ows + -lit(";") + ows
# An empty parameter is allowed: `text/plain;` and `text/plain;;charset=utf-8` are both valid.
parameters = many(-semicolon + maybe(parameter)) >> (lambda ps: dict(p for p in ps if p))

comma = ows + -lit(",") + ows


def comma_list[B](p: Parser[str, B]) -> Parser[str, list[B]]:
    """Any number of `p` between commas, with the empty elements a recipient has to accept."""
    return ows + -many(comma) + separated(p, oneplus(comma), trailing=True) + ows


# --8<-- [end:pieces]


# --8<-- [start:media]
@dataclass(frozen=True, slots=True)
class Media:
    """A media type, its names in lowercase."""

    type: str
    subtype: str
    parameters: dict[str, str]


@dataclass(frozen=True, slots=True)
class MediaRange:
    """A media range of `Accept`, with its weight, which is 1 unless it is given."""

    type: str
    subtype: str
    parameters: dict[str, str]
    q: float


def range_of(parts: tuple[str, str, list[float | tuple[str, str] | None]]) -> MediaRange:
    """A media range from its names and what follows them, in the order they came."""
    type_, subtype, entries = parts
    params = dict(e for e in entries if isinstance(e, tuple))
    weights = [e for e in entries if isinstance(e, float)]
    return MediaRange(type_, subtype, params, weights[-1] if weights else 1.0)


media = name + -lit("/") + name + parameters >> (lambda r: Media(r[0], r[1], r[2]))

# In `Accept`, and only there, `q` is a weight with a syntax of its own. A number that goes on past
# the rule, as in `q=1.5` or `q=0.1234`, fails where it stops being one.
weight = -regex("[qQ]=") + regex(r"0(?:\.[0-9]{0,3})?|1(?:\.0{0,3})?").named("weight") >> float
not_a_weight = parameter.when(lambda p: p[0] != "q")
media_range = (
    name + -lit("/") + name + many(-semicolon + (weight | maybe(not_a_weight))) >> range_of
)
# --8<-- [end:media]


# --8<-- [start:disposition]
@dataclass(frozen=True, slots=True)
class Disposition:
    """A `Content-Disposition`, with `name*` parameters decoded."""

    type: str
    parameters: dict[str, str]

    @property
    def filename(self) -> str | None:
        """The name of the file, where `filename*` has priority over `filename`."""
        return self.parameters.get("filename*", self.parameters.get("filename"))


def decoded(parts: tuple[str, str | None]) -> str:
    """The text of an RFC 8187 value, from its charset and what is percent-encoded."""
    charset, encoded = parts
    try:
        return unquote(encoded or "", encoding=charset, errors="strict")
    except UnicodeDecodeError as e:
        raise Invalid(f"not {charset}: {e.reason}") from None


charset = regex("(?i)utf-8|iso-8859-1").named("charset")
language = -maybe(regex("[A-Za-z0-9-]+").named("language"))
encoded = regex(r"(?:[A-Za-z0-9!#$&+\-.^_`|~]|%[0-9A-Fa-f]{2})+").named("encoded text")
ext_value = (charset + -lit("'") + language + -lit("'") + maybe(encoded)).validated(decoded)
# A token can end in `*` as well, so the extended form goes first.
ext_name = regex(rf"[{TCHAR}]+\*").named("extended name") >> str.lower
ext_parameter = (ext_name + -lit("=") + ext_value).starmap(lambda k, v: (k, v))

# `filename*` has no quoted form, so an extended name is not a parameter like the others.
not_extended = parameter.when(lambda p: not p[0].endswith("*"))
disposition = name + many(-semicolon + (ext_parameter | not_extended)) + ows >> (
    lambda r: Disposition(r[0], dict(r[1]))
)
# --8<-- [end:disposition]


# --8<-- [start:link]
@dataclass(frozen=True, slots=True)
class Link:
    """A link of a `Link` header: where it points to, and how that relates to this resource."""

    target: Uri
    parameters: dict[str, str]


# Between the brackets is a URI reference, so `,` and `;` there are text, and the grammar exists.
link = -lit("<") + reference + -lit(">") + parameters >> (lambda r: Link(r[0], r[1]))
# --8<-- [end:link]


# --8<-- [start:challenge]
@dataclass(frozen=True, slots=True)
class Challenge:
    """One way to authenticate that a `WWW-Authenticate` offers, with a `token68` or parameters."""

    scheme: str
    token68: str | None
    parameters: dict[str, str]


token68 = regex(r"[A-Za-z0-9\-._~+/]+=*").named("token68")
auth_parameter = (name + ows + -lit("=") + ows + value).starmap(lambda k, v: (k, v))
# A comma ends a parameter or a whole challenge. `separated()` goes on only if a parameter follows,
# and leaves the comma for the list of challenges when a new scheme does.
auth_parameters = separated(auth_parameter, oneplus(comma)).when(bool) >> dict


def challenge_of(parts: tuple[str, Any]) -> Challenge:
    """A challenge from its scheme and what follows it, if anything does."""
    scheme, data = parts
    if data is None:
        return Challenge(scheme, None, {})
    return Challenge(scheme, data, {}) if isinstance(data, str) else Challenge(scheme, None, data)


challenge = token + maybe(-regex(" +") + (auth_parameters | token68)) >> challenge_of
# --8<-- [end:challenge]

# --8<-- [start:parse]
content_type: CompiledParser[str, Media] = (media + ows + -finished).compile()
accept: CompiledParser[str, list[MediaRange]] = (comma_list(media_range) + -finished).compile()
content_disposition: CompiledParser[str, Disposition] = (disposition + -finished).compile()
links: CompiledParser[str, list[Link]] = (comma_list(link) + -finished).compile()
www_authenticate: CompiledParser[str, list[Challenge]] = (
    comma_list(challenge) + -finished
).compile()
# --8<-- [end:parse]
