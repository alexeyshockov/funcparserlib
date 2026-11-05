"""A KDL 2.0 parser: a regexp tokenizer, and a grammar that follows the spec rule by rule.

The tokenizer keeps whitespace, because KDL needs it: a newline ends a node, and two arguments must
have whitespace between them. The grammar reads those tokens and builds a tree of `Node`. Strings
and numbers get their values in the `>>` functions of the grammar.

The parser covers the whole language but for the nesting of comments: a regexp cannot count, so a
`/* */` comment holds comments up to 8 levels deep. Text the tokens do not cover fails with
`LexerError`, text the grammar does not cover with `NoParseError`, and a string with a bad escape or
indent with `ParseErrors`, which holds every such string of the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from funcparserlib.lexer import Token, TokenSpec, make_tokenizer
from funcparserlib.parser import ForwardDeclParser, Parser, finished, forward_decl, many, maybe, tok

# --8<-- [start:tree]
type Scalar = str | int | Decimal | bool | None


@dataclass(frozen=True, slots=True)
class Value:
    """An argument or a property value, with its type annotation."""

    value: Scalar
    type: str | None = None


@dataclass(frozen=True, slots=True)
class Prop:
    """A property of a node."""

    key: str
    value: Value


@dataclass(frozen=True, slots=True)
class Node:
    """A node; of a property repeated in the document, `props` holds the rightmost."""

    name: str
    args: list[Value] = field(default_factory=list)
    props: dict[str, Value] = field(default_factory=dict)
    children: list[Node] = field(default_factory=list)
    type: str | None = None


# --8<-- [end:tree]


# --8<-- [start:characters]
SPACE = r"\t \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000"
NEWLINE = r"\n\r\x0b\x0c\x85\u2028\u2029"
DISALLOWED = r"\x00-\x08\x0e-\x1f\x7f\ud800-\udfff\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff"
NL = rf"(?:\r\n|[{NEWLINE}])"
IDENT_CHAR = rf"[^{SPACE}{NEWLINE}{DISALLOWED}\\/(){{}};\[\]\"#=]"
# A number or keyword ends where an identifier could not go on, so `1.0v2` is no token.
END = rf"(?!{IDENT_CHAR})"
# --8<-- [end:characters]


# --8<-- [start:comments]
def block_comment(levels: int) -> str:
    """The regexp of a `/* */` comment with up to `levels` levels of comments nested in it."""
    plain = r"[^*/]|\*(?!/)|/(?!\*)"
    body = plain
    for _ in range(levels):
        body = rf"{plain}|/\*(?:{body})*\*/"
    return rf"/\*(?:{body})*\*/"


COMMENT = block_comment(8)
LINE_COMMENT = rf"//[^{NEWLINE}]*(?:{NL}|\Z)"
ESCLINE = rf"\\(?:[{SPACE}]|{COMMENT})*(?:{LINE_COMMENT}|{NL}|\Z)"
# --8<-- [end:comments]

# --8<-- [start:tokenizer]
tokenize = make_tokenizer(
    [
        TokenSpec("ws", rf"(?:[{SPACE}]+|{COMMENT}|{ESCLINE})+"),
        TokenSpec("nl", rf"{NL}|{LINE_COMMENT}"),
        TokenSpec("slashdash", r"/-"),
        TokenSpec("multiline", rf'"""{NL}(?:[^\\{DISALLOWED}]|\\[^{DISALLOWED}])*?"""'),
        TokenSpec(
            "string", rf'"(?:[^"\\{NEWLINE}{DISALLOWED}]|\\(?:[{SPACE}{NEWLINE}]+|.))*"', re.DOTALL
        ),
        TokenSpec("raw_multiline", rf'(#+)"""{NL}[^{DISALLOWED}]*?"""\1'),
        TokenSpec("raw", rf'(#+)"(?!"")[^{NEWLINE}{DISALLOWED}]*?"\1'),
        TokenSpec("keyword", rf"#(?:true|false|null|inf|-inf|nan){END}"),
        TokenSpec(
            "number",
            r"[+-]?(?:0x[0-9a-fA-F][0-9a-fA-F_]*|0o[0-7][0-7_]*|0b[01][01_]*"
            rf"|[0-9][0-9_]*(?:\.[0-9][0-9_]*)?(?:[eE][+-]?[0-9][0-9_]*)?){END}",
        ),
        TokenSpec("ident", rf"(?![+-]?\.?[0-9]){IDENT_CHAR}+"),
        TokenSpec("op", r"[(){};=]"),
    ]
)
# --8<-- [end:tokenizer]


# --8<-- [start:strings]
ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "b": "\b", "f": "\f", "s": " "}
ESCAPE = re.compile(
    rf"\\(?:(?P<space>[{SPACE}{NEWLINE}]+)|u\{{(?P<code>[0-9a-fA-F]{{1,6}})\}}|(?P<char>.))",
    re.DOTALL,
)
LINE = re.compile(NL)
BLANK = re.compile(rf"[{SPACE}]*")


def is_scalar(code: str) -> bool:
    """Whether the hexadecimal `code` is a Unicode scalar value."""
    n = int(code, 16)
    return n <= 0x10FFFF and not 0xD800 <= n <= 0xDFFF


def valid_escapes(text: str) -> bool:
    """Whether every escape in `text` is one that KDL defines."""
    return all(
        m["space"] or (is_scalar(m["code"]) if m["code"] else m["char"] in ESCAPES)
        for m in ESCAPE.finditer(text)
    )


def unescape(text: str) -> str:
    """The string `text` stands for; an invalid escape, reported by a check, stands for nothing."""

    def resolve(m: re.Match[str]) -> str:
        if m["code"]:
            return chr(int(m["code"], 16)) if is_scalar(m["code"]) else ""
        return ESCAPES.get(m["char"], "") if m["char"] else ""

    return ESCAPE.sub(resolve, text)


def without_space_escapes(text: str) -> str:
    """`text` without its escaped whitespace, every other escape left as it is."""
    return ESCAPE.sub(lambda m: "" if m["space"] else m[0], text)


def valid_indent(text: str) -> bool:
    """Whether the last line of `text` is blank and starts every other line that is not blank."""
    _, *lines, last = LINE.split(text)
    return BLANK.fullmatch(last) is not None and all(
        line.startswith(last) or BLANK.fullmatch(line) for line in lines
    )


def dedent(text: str) -> str:
    """The lines between the first and the last of `text`, without the indent of the last."""
    _, *lines, last = LINE.split(text)
    return "\n".join("" if BLANK.fullmatch(line) else line.removeprefix(last) for line in lines)


# --8<-- [end:strings]


# --8<-- [start:numbers]
KEYWORDS: dict[str, Scalar] = {
    "#true": True,
    "#false": False,
    "#null": None,
    "#inf": Decimal("Infinity"),
    "#-inf": Decimal("-Infinity"),
    "#nan": Decimal("NaN"),
}
KEYWORD_NAMES = {keyword.removeprefix("#") for keyword in KEYWORDS}


def number(text: str) -> int | Decimal:
    """An `int`, or a `Decimal`, which holds what a `float` would round or overflow."""
    digits = text.replace("_", "")
    if digits.lstrip("+-")[:2] in ("0x", "0o", "0b"):
        return int(digits, 0)
    value = Decimal(digits)
    # `int()` of a string refuses more than 4300 digits, and of a `Decimal` does not.
    return int(value) if digits.lstrip("+-").isdigit() else value


# --8<-- [end:numbers]


# --8<-- [start:grammar]
def op(s: str) -> Parser[Token, str]:
    """The punctuation `s`."""
    return tok("op", s)


def dropped(p: Parser[Token, Any]) -> Parser[Token, None]:
    """`p` after a slashdash, which turns what `p` parsed into `None`."""
    return slashdash + -p >> (lambda _: None)


def separated[B](p: Parser[Token, B], end: Parser[Token, Any]) -> Parser[Token, list[B]]:
    """Any number of `p`, with an `end` between two of them and optionally after the last."""
    return maybe(p + many(-end + p) + -maybe(end)) >> (lambda r: [] if r is None else [r[0], *r[1]])


def make_node(parts: tuple[Any, ...]) -> Node:
    """A node from its type, name, entries and children."""
    type_, name, entries, nested = parts
    args = [e for e in entries if isinstance(e, Value)]
    props = {e.key: e.value for e in entries if isinstance(e, Prop)}
    return Node(name, args, props, nested or [], type_)


BAD_ESCAPE = "unknown escape in a string"
BAD_INDENT = "a line does not start with the indent of the closing quotes"

ws = tok("ws").named("whitespace")
newline = tok("nl").named("newline")
space = -maybe(ws)
line_space = -many(ws | newline)
slashdash = -tok("slashdash").named("'/-'") + line_space

quoted = tok("string").check(valid_escapes, BAD_ESCAPE) >> (lambda t: unescape(t[1:-1]))
multiline = (
    (tok("multiline") >> (lambda t: without_space_escapes(t[3:-3])))
    .check(valid_escapes, BAD_ESCAPE)
    .check(valid_indent, BAD_INDENT)
) >> (lambda t: unescape(dedent(t)))
raw = tok("raw") >> (lambda t: t.strip("#")[1:-1])
raw_multiline = (tok("raw_multiline") >> (lambda t: t.strip("#")[3:-3])).check(
    valid_indent, BAD_INDENT
) >> dedent
bare = tok("ident").when(lambda s: s not in KEYWORD_NAMES).named("identifier")
string = (bare | quoted | multiline | raw | raw_multiline).named("string")

annotation = -op("(") + space + string + space + -op(")")
literal = string | tok("number") >> number | tok("keyword") >> KEYWORDS.__getitem__
value = maybe(annotation) + space + literal >> (lambda r: Value(r[1], r[0]))
prop_value = space + -op("=") + space + value
string_entry = string + maybe(prop_value) >> (
    lambda r: Value(r[0]) if r[1] is None else Prop(r[0], r[1])
)
entry = string_entry | value

nodes: ForwardDeclParser[Token, list[Node]] = forward_decl().named("nodes")
children = -op("{") + nodes + -op("}")
entries = many(-ws + entry | space + dropped(entry))
dropped_children = -many(space + dropped(children))

body = (
    maybe(annotation + space)
    + string
    + entries
    + dropped_children
    + maybe(space + children)
    + dropped_children
    + space
) >> make_node
node = dropped(body) | body
terminator = (newline | op(";")) + line_space
nodes.define(
    line_space + separated(node, terminator) >> (lambda ns: [n for n in ns if n is not None])
)
document = nodes + -finished
# --8<-- [end:grammar]


# --8<-- [start:parse]
def parse(text: str) -> list[Node]:
    """The top-level nodes of the KDL document `text`.

    Raises `LexerError` for text that is no token, `NoParseError` for tokens out of place,
    `ParseErrors` for strings with a bad escape or indent, `RecursionError` for a document nested
    too deeply and `decimal.InvalidOperation` for an exponent a `Decimal` cannot hold.
    """
    return document.parse(list(tokenize(text.removeprefix("\ufeff"))))


# --8<-- [end:parse]

if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(parse(sys.stdin.read()))  # noqa: T203 — the program's output
