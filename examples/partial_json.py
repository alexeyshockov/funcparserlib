"""JSON from a language model: a document cut off anywhere, and what models write in place of JSON.

Goal: give the value that a prefix of a JSON document stands for while it streams, and read the
mistakes a model makes when they have one meaning: a trailing comma, single quotes, `True` and
`None`, a Markdown code fence around the document. What has more than one meaning stays an error.

A prefix stands for what it has finished, and a value once given only grows: a string is given as
far as it goes, and a number, a literal or a key that the text may still change is left out until
something ends it. `parse()` of each prefix gives what a program can show while the rest comes. The
document is an object or an array, so every number in it is ended by what follows it.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final

from funcparserlib.lexer import Token, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    CompiledParser,
    ForwardDeclParser,
    IgnoredParser,
    Parser,
    finished,
    forward_decl,
    maybe,
    separated,
    skip,
    tok,
    tokenized,
)

# --8<-- [start:values]
type Json = bool | int | float | str | list[Json] | dict[str, Json] | None
type Document = dict[str, Json] | list[Json]


class Missing(Enum):
    """A value the text has not reached yet, or may still change: `tr` of `true`, `1.` of `1.5`."""

    MISSING = "missing"


MISSING: Final = Missing.MISSING
# --8<-- [end:values]

# --8<-- [start:tokens]
ESCAPE = r"""\\(?:["'\\/bfnrt]|u[0-9A-Fa-f]{4})"""
UNFINISHED_ESCAPE = r"\\(?:u[0-9A-Fa-f]{0,3})?"
BODY = {quote: rf"(?:[^{quote}\\]|{ESCAPE})*" for quote in "\"'"}
NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[Ee][+-]?[0-9]+)?"

tokenizer = make_tokenizer(
    [
        TokenSpec("space", r"\s+"),
        TokenSpec("fence", r"```[A-Za-z]*|`{1,2}\Z"),
        TokenSpec("string", "|".join(f"{q}{BODY[q]}{q}" for q in "\"'")),
        TokenSpec(
            "unfinished_string", "|".join(rf"{q}{BODY[q]}(?:{UNFINISHED_ESCAPE})?\Z" for q in "\"'")
        ),
        TokenSpec(
            "unfinished_number",
            rf"(?:-|{NUMBER}|-?(?:0|[1-9][0-9]*)(?:\.|(?:\.[0-9]+)?[Ee][+-]?))\Z",
        ),
        TokenSpec("number", NUMBER),
        TokenSpec("op", r"[{}\[\],:]"),
        TokenSpec("name", r"[A-Za-z_][A-Za-z_0-9]*"),
    ],
    skip={"space"},
)
# --8<-- [end:tokens]

# --8<-- [start:scalars]
ESCAPES = {
    '"': '"',
    "'": "'",
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
LITERALS: dict[str, Json] = {"true": True, "false": False, "null": None}
LITERALS.update({"True": True, "False": False, "None": None})


def unescape(body: str) -> str:
    """The text of a string's body, with a `\\uXXXX` pair as the one character it encodes."""
    text = re.sub(
        r"\\(?:u([0-9A-Fa-f]{4})|(.))",
        lambda m: chr(int(m[1], 16)) if m[1] else ESCAPES[m[2]],
        body,
    )
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")


def string_of(token: str) -> str:
    return unescape(token[1:-1])


def unfinished_string_of(token: str) -> str:
    """The text of a string the text ends inside, without an escape it ends inside.

    A high surrogate at the end is the first half of a `\\uXXXX` pair, and is left out with it.
    """
    body = re.match(BODY[token[0]], token[1:])
    assert body is not None  # the token is a quote and a body
    return re.sub(r"[\ud800-\udbff]\Z", "", unescape(body[0]))


def number_of(token: str) -> int | float:
    return float(token) if any(c in token for c in ".eE") else int(token)


def missing(_: object) -> Missing:
    return MISSING


def is_literal_prefix(name: str) -> bool:
    return any(literal.startswith(name) for literal in LITERALS)


string = (tok("string") >> string_of) | (tok("unfinished_string") >> unfinished_string_of)
number = tok("number") >> number_of
literal = tok("name").when(LITERALS.__contains__) >> LITERALS.__getitem__
unfinished = (tok("unfinished_number") | tok("name").when(is_literal_prefix) + -finished) >> missing
nothing = finished >> missing
# --8<-- [end:scalars]


# --8<-- [start:containers]
def op(s: str) -> Parser[Token, str]:
    return tok("op", s)


def close(s: str) -> IgnoredParser[Token]:
    """The closing bracket `s`, or the end of the text, where every open bracket closes."""
    return skip(op(s) | finished)


def make_object(members: list[tuple[str, Json | Missing]]) -> dict[str, Json]:
    return {k: v for k, v in members if v is not MISSING}


def as_value(v: Json | Missing) -> Json | Missing:
    return v


def make_array(items: list[Json | Missing]) -> list[Json]:
    return [v for v in items if v is not MISSING]


value: ForwardDeclParser[Token, Json | Missing] = forward_decl().named("value")
value_or_end = value | nothing
after_key = -op(":") + value_or_end | nothing
member = string + after_key
json_object = -op("{") + separated(member, op(","), trailing=True) + close("}") >> make_object
json_array = -op("[") + separated(value, op(","), trailing=True) + close("]") >> make_array
# `Parser` is invariant in its value: `define()` takes a parser of `Json | Missing` exactly.
value.define((json_object | json_array | string | number | literal | unfinished) >> as_value)
# --8<-- [end:containers]

# --8<-- [start:parse]
body = json_object | json_array | nothing
fence = -maybe(tok("fence"))
document = fence + body + fence + -finished
compiled: CompiledParser[str, Document | Missing] = tokenized(document, tokenizer).compile()


def parse(text: str) -> Document | Missing:
    """The object or array the start of a JSON document `text` stands for, or `MISSING` before it
    has one.

    Raises `NoParseError` or `LexerError` where `text` stops being the start of a document.
    """
    return compiled.parse_text(text)


# --8<-- [end:parse]

if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(parse(sys.stdin.read()))  # noqa: T203 — the program's output
