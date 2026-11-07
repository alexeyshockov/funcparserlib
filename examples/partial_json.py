"""JSON from a language model: a document cut off anywhere, and what models write in place of JSON.

Goal: give the value that a prefix of a JSON document stands for while it streams, and read the
mistakes a model makes when they have one meaning: a trailing comma, single quotes, `True` and
`None`, a Markdown code fence around the document. What has more than one meaning stays an error.

A prefix stands for what it has finished, and a value once given only grows: a string is given as
far as it goes, and a number, a literal or a key that the text may still change is left out until
something ends it. `parse()` of each prefix gives what a program can show while the rest comes. The
document is an object or an array, so every number in it is ended by what follows it.

A string, an array or an object the text ends inside is an `Open` subclass of its type, so a program
tells the finished parts from those still being written. `find()` gives the objects in a reply with
prose around them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from enum import Enum
from typing import Final

from funcparserlib.lexer import Token, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    CompiledParser,
    ForwardDeclParser,
    Located,
    Parser,
    finished,
    forward_decl,
    maybe,
    regex,
    separated,
    tok,
    tokenized,
)
from funcparserlib.util import pretty_tree

# --8<-- [start:values]
type Json = bool | int | float | str | list[Json] | dict[str, Json] | None
type Document = dict[str, Json] | list[Json]


class Missing(Enum):
    """A value the text has not reached yet, or may still change: `tr` of `true`, `1.` of `1.5`."""

    MISSING = "missing"


MISSING: Final = Missing.MISSING
# --8<-- [end:values]


# --8<-- [start:open]
class Open:
    """A string, an array or an object the text ends inside: the rest of the text may add to it."""

    __slots__ = ()


class OpenStr(Open, str):
    __slots__ = ()


class OpenList(Open, list[Json]):
    __slots__ = ()


class OpenDict(Open, dict[str, Json]):
    __slots__ = ()


# --8<-- [end:open]

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
    return OpenStr(re.sub(r"[\ud800-\udbff]\Z", "", unescape(body[0])))


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


def closed(_: object) -> bool:
    return True


def ended(_: object) -> bool:
    return False


def close(s: str) -> Parser[Token, bool]:
    """The closing bracket `s`, or the end of the text, where every open bracket closes: whether the
    bracket was there.
    """
    return (op(s) >> closed) | (finished >> ended)


def make_object(members: list[tuple[str, Json | Missing]], is_closed: bool) -> dict[str, Json]:
    d = {k: v for k, v in members if v is not MISSING}
    return d if is_closed else OpenDict(d)


def as_value(v: Json | Missing) -> Json | Missing:
    return v


def make_array(items: list[Json | Missing], is_closed: bool) -> list[Json]:
    a = [v for v in items if v is not MISSING]
    return a if is_closed else OpenList(a)


value: ForwardDeclParser[Token, Json | Missing] = forward_decl().named("value")
value_or_end = value | nothing
after_key = -op(":") + value_or_end | nothing
member = string + after_key
object_body = (separated(member, op(","), trailing=True) + close("}")).starmap(make_object)
json_object = -op("{") + object_body
json_array = -op("[") + (separated(value, op(","), trailing=True) + close("]")).starmap(make_array)
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

# --8<-- [start:find]
in_reply = (-regex(r"\{") + tokenized(object_body, tokenizer)).compile()


def find(reply: str) -> Iterator[Located[dict[str, Json]]]:
    """The objects in a model's reply, with where each is, the last one possibly unfinished."""
    return in_reply.scan_text(reply)


# --8<-- [end:find]

# --8<-- [start:tree]
type Labeled = tuple[str, object]


def kids(node: Labeled) -> list[Labeled]:
    _, v = node
    if isinstance(v, dict):
        return [(f"{k!r}: ", x) for k, x in v.items()]
    if isinstance(v, list):
        return [("", x) for x in v]
    return []


def show(node: Labeled) -> str:
    label, v = node
    text = "{}" if isinstance(v, dict) else "[]" if isinstance(v, list) else repr(v)
    return label + text + ("  (open)" if isinstance(v, Open) else "")


def tree(value: Document) -> str:
    """`value` drawn as a tree, with the parts the text ends inside marked as open."""
    root: Labeled = ("", value)
    return pretty_tree(root, kids, show)


# --8<-- [end:tree]

if __name__ == "__main__":
    import sys

    result = parse(sys.stdin.read())
    print(result if result is MISSING else tree(result))
