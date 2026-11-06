"""A Markdown parser: markdown-it-py tokenizes, funcparserlib builds a typed syntax tree.

markdown-it-py turns Markdown into a flat stream of tokens, where a block is an `*_open` token, its
contents and a matching `*_close` token, and each `inline` token carries the stream of its text as
`children`. Those two streams are what the two grammars below read: the block grammar parses the
document's stream, and the inline grammar parses each `inline` token's children.

The tree covers headings, paragraphs, bullet and ordered lists, block quotes, code blocks and
thematic breaks; inside them, text, code spans, emphasis, strong emphasis, links and line breaks.
Anything else (an image, an HTML block) fails with `NoParseError`, naming the token and its line.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.token import Token

from funcparserlib.parser import (
    ForwardDeclParser,
    Parser,
    TokenDescription,
    finished,
    forward_decl,
    many,
    some,
)


# --8<-- [start:tree]
@dataclass(frozen=True, slots=True)
class Text:
    """Plain text."""

    text: str


@dataclass(frozen=True, slots=True)
class Code:
    """A code span."""

    text: str


@dataclass(frozen=True, slots=True)
class Emphasis:
    """Emphasised inline content: `*text*`."""

    children: Sequence[Inline]


@dataclass(frozen=True, slots=True)
class Strong:
    """Strongly emphasised inline content: `**text**`."""

    children: Sequence[Inline]


@dataclass(frozen=True, slots=True)
class Link:
    """A link to `href` with its text."""

    href: str
    children: Sequence[Inline]


@dataclass(frozen=True, slots=True)
class Break:
    """A line break: a soft one is a newline in the paragraph, a hard one ends the line."""

    hard: bool


type Inline = Text | Code | Emphasis | Strong | Link | Break


@dataclass(frozen=True, slots=True)
class Heading:
    """A heading of `level` 1 to 6."""

    level: int
    children: Sequence[Inline]


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A paragraph."""

    children: Sequence[Inline]


@dataclass(frozen=True, slots=True)
class BulletList:
    """A bullet list; each item is a list of blocks."""

    items: Sequence[Sequence[Block]]


@dataclass(frozen=True, slots=True)
class OrderedList:
    """An ordered list numbered from `start`; each item is a list of blocks."""

    start: int
    items: Sequence[Sequence[Block]]


@dataclass(frozen=True, slots=True)
class Blockquote:
    """A block quote."""

    children: Sequence[Block]


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A fenced or indented code block; `info` is the fence's language, empty if none."""

    info: str
    text: str


@dataclass(frozen=True, slots=True)
class ThematicBreak:
    """A thematic break: `---`."""


type Block = Heading | Paragraph | BulletList | OrderedList | Blockquote | CodeBlock | ThematicBreak
# --8<-- [end:tree]


# --8<-- [start:token]
def token(type_: str) -> Parser[Token, Token]:
    """Any one markdown-it token of the type `type_`."""

    def is_type(t: Token) -> bool:
        return t.type == type_

    return some(is_type).named(type_)


# --8<-- [end:token]


# --8<-- [start:describe]
def describe(t: Token, line: int | None = None) -> TokenDescription:
    """How an error shows a markdown-it token: its type and the line it is on.

    A block token knows its lines (`map`); an inline token does not, and takes `line` from the block
    around it. markdown-it-py records no columns, so the description has no position.
    """
    if t.map:
        line = t.map[0] + 1
    return TokenDescription(t.type if line is None else f"{t.type} on line {line}")


# --8<-- [end:describe]


# --8<-- [start:inline]
def make_link(opening_and_children: tuple[Token, list[Inline]]) -> Link:
    """A link from its `link_open` token, which holds the target, and its text."""
    opening, children = opening_and_children
    return Link(str(opening.attrs["href"]), children)


inline: ForwardDeclParser[Token, Inline] = forward_decl().named("inline content")
text = token("text") >> (lambda t: Text(t.content))
code = token("code_inline") >> (lambda t: Code(t.content))
softbreak = token("softbreak") >> (lambda _: Break(hard=False))
hardbreak = token("hardbreak") >> (lambda _: Break(hard=True))
emphasis = -token("em_open") + many(inline) + -token("em_close") >> Emphasis
strong = -token("strong_open") + many(inline) + -token("strong_close") >> Strong
link = token("link_open") + many(inline) + -token("link_close") >> make_link
inline.define(text | code | softbreak | hardbreak | emphasis | strong | link)
inlines = many(inline) + -finished
# --8<-- [end:inline]


# --8<-- [start:block]
def inline_content(t: Token) -> list[Inline]:
    """The inline content of an `inline` token, parsed from its children, which take its line."""
    line = t.map[0] + 1 if t.map else None
    return inlines.parse(t.children or [], describe=lambda child: describe(child, line))


def make_heading(opening_and_children: tuple[Token, list[Inline]]) -> Heading:
    """A heading from its `heading_open` token, whose tag (`h1` to `h6`) holds the level."""
    opening, children = opening_and_children
    return Heading(int(opening.tag[1:]), children)


def make_ordered_list(opening_and_items: tuple[Token, list[list[Block]]]) -> OrderedList:
    """An ordered list from its `ordered_list_open` token, which holds the first number."""
    opening, items = opening_and_items
    return OrderedList(int(opening.attrs.get("start", 1)), items)


block: ForwardDeclParser[Token, Block] = forward_decl().named("block")
inline_ = token("inline") >> inline_content
heading = token("heading_open") + inline_ + -token("heading_close") >> make_heading
paragraph = -token("paragraph_open") + inline_ + -token("paragraph_close") >> Paragraph
item = -token("list_item_open") + many(block) + -token("list_item_close")
bullet_list = -token("bullet_list_open") + many(item) + -token("bullet_list_close") >> BulletList
ordered_list = (
    token("ordered_list_open") + many(item) + -token("ordered_list_close") >> make_ordered_list
)
blockquote = -token("blockquote_open") + many(block) + -token("blockquote_close") >> Blockquote
fence = token("fence") >> (lambda t: CodeBlock(t.info, t.content))
indented_code = token("code_block") >> (lambda t: CodeBlock("", t.content))
thematic_break = token("hr") >> (lambda _: ThematicBreak())
block.define(
    heading
    | paragraph
    | bullet_list
    | ordered_list
    | blockquote
    | fence
    | indented_code
    | thematic_break
)
document = many(block) + -finished
# --8<-- [end:block]

# --8<-- [start:parse]
markdown = MarkdownIt("commonmark")


def parse(text: str) -> list[Block]:
    """The blocks of the Markdown `text`; raises `NoParseError` for what the tree does not cover."""
    return document.parse(markdown.parse(text), describe=describe)


# --8<-- [end:parse]

if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(parse(sys.stdin.read()))  # noqa: T203 — the program's output
