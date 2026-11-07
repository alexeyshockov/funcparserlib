"""A JSON parser using funcparserlib.

The parser is based on [the JSON grammar][1].

  [1]: https://tools.ietf.org/html/rfc4627
"""

import re
import sys
from collections.abc import Callable, Sequence
from functools import cache
from pprint import pformat
from re import VERBOSE, Match
from typing import Any, TypeVar

from funcparserlib.lexer import LexerError, Token, Tokenizer, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    ForwardDeclParser,
    NoParseError,
    Parser,
    finished,
    forward_decl,
    many,
    maybe,
    tok,
)

ENCODING = "UTF-8"
# noinspection SpellCheckingInspection
regexes = {
    "escaped": r"""
        \\                                  # Escape
          ((?P<standard>["\\/bfnrt])        # Standard escapes
        | (u(?P<unicode>[0-9A-Fa-f]{4})))   # uXXXX
        """,
    "unescaped": r"""
        [^"\\]                              # Unescaped: avoid ["\\]
        """,
}
re_esc = re.compile(regexes["escaped"], VERBOSE)
T = TypeVar("T")
JsonValue = None | bool | dict | list | int | float | str
JsonMember = tuple[str, JsonValue]


def tokenize(s: str) -> list[Token]:
    return list(_tokenizer()(s))


# Built once: the specs and the grammar do not depend on the input.
@cache
def _tokenizer() -> Tokenizer:
    """The JSON tokenizer."""
    specs = [
        TokenSpec("space", r"[ \t\r\n]+"),
        TokenSpec("string", r'"({unescaped} | {escaped})*"'.format(**regexes), VERBOSE),
        TokenSpec(
            "number",
            r"""
            -?                  # Minus
            (0|([1-9][0-9]*))   # Int
            (\.[0-9]+)?         # Frac
            ([Ee][+-]?[0-9]+)?   # Exp
            """,
            VERBOSE,
        ),
        TokenSpec("op", r"[{}\[\]\-,:]"),
        TokenSpec("name", r"[A-Za-z_][A-Za-z_0-9]*"),
    ]
    return make_tokenizer(specs, skip={"space"})


def parse(tokens: Sequence[Token]) -> JsonValue:
    return _grammar().parse(tokens)


@cache
def _grammar() -> Parser[Token, JsonValue]:
    """The JSON grammar, up to the end of input."""

    def const(x: T) -> Callable[[Any], T]:
        return lambda _: x

    def op(s: str) -> Parser[Token, str]:
        return tok("op", s)

    def n(s: str) -> Parser[Token, str]:
        return tok("name", s)

    def make_array(values: tuple[JsonValue, list[JsonValue]] | None) -> list[JsonValue]:
        if values is None:
            return []
        else:
            return [values[0]] + values[1]

    def make_object(values: tuple[JsonMember, list[JsonMember]] | None) -> dict[str, Any]:
        if values is None:
            return {}
        else:
            first, rest = values
            k, v = first
            d = {k: v}
            d.update(rest)
            return d

    def make_number(s: str) -> int | float:
        try:
            return int(s)
        except ValueError:
            return float(s)

    def unescape(s: str) -> str:
        std = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }

        def sub(m: Match[str]) -> str:
            if m.group("standard") is not None:
                return std[m.group("standard")]
            else:
                return chr(int(m.group("unicode"), 16))

        # A `\uXXXX` pair encodes one character outside the BMP; a lone surrogate stays as is.
        return (
            re_esc.sub(sub, s).encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")
        )

    def make_string(s: str) -> str:
        return unescape(s[1:-1])

    def make_member(values: JsonMember) -> JsonMember:
        k, v = values
        return k, v

    null = n("null") >> const(None)
    true = n("true") >> const(True)
    false = n("false") >> const(False)
    number = tok("number") >> make_number
    string = tok("string") >> make_string
    value: ForwardDeclParser[Token, JsonValue] = forward_decl().named("json_value")
    member = string + -op(":") + value >> make_member
    json_object = (
        (-op("{") + maybe(member + many(-op(",") + member)) + -op("}")) >> make_object
    ).named("json_object")
    json_array = (
        (-op("[") + maybe(value + many(-op(",") + value)) + -op("]")) >> make_array
    ).named("json_array")
    value.define(null | true | false | json_object | json_array | number | string)
    return value + -finished


def loads(s: str) -> JsonValue:
    return parse(tokenize(s))


def main() -> None:
    try:
        text = sys.stdin.read()
        tree = loads(text)
        print(pformat(tree))  # noqa: T201 — the CLI's output
    except (NoParseError, LexerError) as e:
        print(f"syntax error: {e}", file=sys.stderr)  # noqa: T201 — the CLI's output
        sys.exit(1)


if __name__ == "__main__":
    main()
