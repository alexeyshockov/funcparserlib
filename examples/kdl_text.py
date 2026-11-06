"""A KDL 2.0 parser that reads the text itself, without a tokenizer.

The grammar is `examples/kdl.py` with its lexical part written as parsers of the text: whitespace,
comments and strings are rules, and what is not lexical (the tree, the values of strings and
numbers) is imported from there. A `/* */` comment is a recursive rule, so it nests as deep as the
stack allows, and a rule asks only for the tokens that make sense where it is, so the alternatives
of a string need no order. Every failure is a `NoParseError` or a `ParseErrors` with the line and
column of the text, including text a tokenizer would not read.
"""

from __future__ import annotations

import re
from typing import Any

from examples.kdl import (
    DISALLOWED,
    END,
    IDENT_CHAR,
    KEYWORD_NAMES,
    KEYWORDS,
    LINE_COMMENT,
    NEWLINE,
    NL,
    SPACE,
    Node,
    Prop,
    Value,
    dedent,
    make_node,
    number,
    unescape,
    without_space_escapes,
)
from funcparserlib.parser import (
    CompiledParser,
    ForwardDeclParser,
    Parser,
    finished,
    forward_decl,
    lit,
    longest,
    many,
    maybe,
    oneplus,
    regex,
)

# --8<-- [start:whitespace]
# KDL keeps its whitespace significant, so these parsers skip nothing.
spaces = regex(rf"[{SPACE}]+")
line_comment = regex(LINE_COMMENT)
newline_char = regex(NL)

# A comment holds comments, which a regex cannot count and a rule can.
block_comment: ForwardDeclParser[str, Any] = forward_decl().named("comment")
block_comment.define(
    lit("/*")
    + many(regex(r"(?:[^*/]|\*(?!/)|/(?!\*))+").named("comment text") | block_comment)
    + lit("*/")
)

# A backslash continues the line, and so does a comment before the newline.
esc_line = lit("\\") + many(spaces | block_comment) + (line_comment | newline_char | finished)

ws = oneplus(spaces | block_comment | esc_line).named("whitespace")
newline = (newline_char | line_comment).named("newline")
# --8<-- [end:whitespace]

# --8<-- [start:strings]
plain = regex(rf'[^"\\{NEWLINE}{DISALLOWED}]+').named("text")
escape = regex(rf"\\(?:[{SPACE}{NEWLINE}]+|.)", re.DOTALL).named("escape")
# Between the quotes come text and escapes in any order; a missing quote fails there.
quoted_body = (-lit('"') + many(plain | escape) + -lit('"')).context("string")

quoted = quoted_body >> "".join
multiline = regex(rf'"""{NL}(?:[^\\{DISALLOWED}]|\\[^{DISALLOWED}])*?"""')
raw = regex(rf'(#+)"(?!"")[^{NEWLINE}{DISALLOWED}]*?"\1')
raw_multiline = regex(rf'(#+)"""{NL}[^{DISALLOWED}]*?"""\1')
bare = regex(rf"(?![+-]?\.?[0-9]){IDENT_CHAR}+").when(lambda s: s not in KEYWORD_NAMES)

# `quoted` reads the `""` that opens a `"""` string, and `longest()` takes the longer match.
string = longest(
    bare.named("identifier"),
    quoted.validated(unescape),
    (multiline >> (lambda s: without_space_escapes(s[3:-3]))).validated(dedent).validated(unescape),
    raw >> (lambda s: s.strip("#")[1:-1]),
    (raw_multiline >> (lambda s: s.strip("#")[3:-3])).validated(dedent),
).named("string")
# --8<-- [end:strings]

# --8<-- [start:grammar]
numeric = regex(
    r"[+-]?(?:0x[0-9a-fA-F][0-9a-fA-F_]*|0o[0-7][0-7_]*|0b[01][01_]*"
    rf"|[0-9][0-9_]*(?:\.[0-9][0-9_]*)?(?:[eE][+-]?[0-9][0-9_]*)?){END}"
).named("number")
keyword = regex(rf"#(?:true|false|null|inf|-inf|nan){END}").named("keyword")

space = -maybe(ws)
line_space = -many(ws | newline)
slashdash = -lit("/-") + line_space


def dropped(p: Parser[str, Any]) -> Parser[str, None]:
    """`p` after a slashdash, which turns what `p` parsed into `None`."""
    return slashdash + -p >> (lambda _: None)


def separated[B](p: Parser[str, B], end: Parser[str, Any]) -> Parser[str, list[B]]:
    """Any number of `p`, with an `end` between two of them and optionally after the last."""
    return maybe(p + many(-end + p) + -maybe(end)) >> (lambda r: [] if r is None else [r[0], *r[1]])


annotation = -lit("(") + space + string + space + -lit(")")
literal = string | numeric >> number | keyword >> KEYWORDS.__getitem__
value = maybe(annotation) + space + literal >> (lambda r: Value(r[1], r[0]))
prop_value = space + -lit("=") + space + value
string_entry = string + maybe(prop_value) >> (
    lambda r: Value(r[0]) if r[1] is None else Prop(r[0], r[1])
)
entry = string_entry | value

nodes: ForwardDeclParser[str, list[Node]] = forward_decl().named("nodes")
children = -lit("{") + nodes + -lit("}")
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
terminator = (newline | lit(";")) + line_space
nodes.define(
    line_space + separated(node, terminator) >> (lambda ns: [n for n in ns if n is not None])
)
document: Parser[str, list[Node]] = nodes + -finished
# --8<-- [end:grammar]

# --8<-- [start:parse]
compiled: CompiledParser[str, list[Node]] = document.compile()


def parse(source: str) -> list[Node]:
    """The top-level nodes of the KDL document `source`.

    Raises `NoParseError` for text the grammar does not cover, with its line and column,
    `ParseErrors` for strings with a bad escape or indent, `RecursionError` for a document nested
    too deeply and `decimal.InvalidOperation` for an exponent a `Decimal` cannot hold.
    """
    return compiled.parse_text(source.removeprefix("\ufeff"))


# --8<-- [end:parse]

if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(parse(sys.stdin.read()))  # noqa: T203 — the program's output
