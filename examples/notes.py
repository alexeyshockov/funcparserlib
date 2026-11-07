"""Migrating notes in Markdown to GitHub alerts: markdown-it reads the file, a grammar finds the
notes.

Goal: a whole-file rewrite where a construct is a sequence of markdown-it blocks, found anywhere in
a file, list items and block quotes included, and the text around the constructs is kept to the
character.

A note is written one of three ways: a paragraph that starts with `Note:` or `**Note:**`, a block
quote whose first paragraph is one, or a `Note` heading followed by paragraphs. Each one becomes a
GitHub alert, `> [!NOTE]`, with its paragraphs. A `Note` heading with no paragraph after it is a
broken note, and stops the migration at its line.

`Parser.scan()` finds the notes in markdown-it's tokens, and its `span` function turns the tokens of
a note into the lines of the file they came from. markdown-it gives the text of a block without the
prefix each of its lines has in a container, `- ` or `> `, so the new text gets the prefix back.

The line endings of a note are kept, and so is the end of a file without one.

Run it on a file: `python -m examples.notes README.md`.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from funcparserlib.parser import Parser, TokenDescription, commit, many, oneplus, some


# --8<-- [start:grammar]
@dataclass(frozen=True)
class Note:
    """A note: its paragraphs, each a text of lines, and whether it was a block quote."""

    paragraphs: Sequence[str]
    quoted: bool = False


NOTE = re.compile(r"(?:\*\*Note:\*\*|Note:)[ \t]*(.+)", re.DOTALL)


def block(type_: str) -> Parser[Token, Token]:
    """Any one markdown-it token of the type `type_`."""

    def is_type(t: Token) -> bool:
        return t.type == type_

    return some(is_type).named(type_)


def inline(test: re.Pattern[str], label: str) -> Parser[Token, str]:
    """An `inline` token whose text `test` matches, giving its first group."""

    def matches(t: Token) -> bool:
        return t.type == "inline" and test.fullmatch(t.content) is not None

    def read(t: Token) -> str:
        found = test.fullmatch(t.content)
        return found[1] if found else ""

    return some(matches).named(label) >> read


ANY = re.compile(r"((?s:.+))")
paragraph = -block("paragraph_open") + inline(ANY, "text") + -block("paragraph_close")
noted = -block("paragraph_open") + inline(NOTE, "a note") + -block("paragraph_close")
titled = -block("heading_open") + -inline(re.compile(r"(Note)"), "Note")
# Once a heading says `Note`, what follows has to be a note.
under = commit(-block("heading_close") + oneplus(paragraph))

quote = -block("blockquote_open") + noted + many(paragraph) + -block("blockquote_close")

note = (
    (quote >> (lambda p: Note([p[0], *p[1]], quoted=True)))
    | (noted >> (lambda text: Note([text])))
    | (titled + under >> Note)
)
# --8<-- [end:grammar]


# --8<-- [start:span]
BREAK = re.compile(r"\r\n|\r|\n")
LAST_BREAK = re.compile(r"(?:\r\n|\r|\n)\Z")


def line_starts(text: str) -> list[int]:
    """The offset each line of `text` starts at, and its length after the last.

    Lines end as markdown-it ends them, at `\\r\\n`, `\\r` or `\\n`.
    """
    return [0, *(m.end() for m in BREAK.finditer(text)), len(text)]


def lines_of(tokens: Sequence[Token], start: int, end: int) -> tuple[int, int]:
    """The first line of the tokens from `start` to `end`, and the line after their last."""
    maps = [t.map for t in tokens[start:end] if t.map is not None]
    if not maps:
        msg = f"no token from {start} to {end} has the lines it came from"
        raise ValueError(msg)
    return maps[0][0], max(m[1] for m in maps)


# --8<-- [end:span]


# --8<-- [start:prefix]
CONTAINER = re.compile(r"(?:[ \t]*(?:>[ \t]?|[-*+][ \t]+|\d+[.)][ \t]+))*[ \t]*")


def prefixes(line: str, *, quoted: bool) -> tuple[str, str]:
    """The prefix of the first line of a block in a container, and of the lines after it.

    A list marker stands on the first line only: the lines after it are indented as far. The quote
    of a `quoted` note is the note's own, and no part of its prefix.
    """
    first = CONTAINER.match(line)
    head = first[0] if first else ""
    if quoted:
        head = re.sub(r">[ \t]?(?=[^>]*$)", "", head, count=1)
    rest = re.sub(r"[-*+]|\d+[.)]", lambda m: " " * len(m[0]), head)
    return head, rest


def alert(found: Note) -> list[str]:
    """The lines of a GitHub alert of the note, without the prefix of a container."""
    lines = ["> [!NOTE]"]
    for i, text in enumerate(found.paragraphs):
        if i:
            lines.append(">")
        lines += [f"> {line}" for line in text.splitlines()]
    return lines


# --8<-- [end:prefix]


# --8<-- [start:migrate]
def describe(t: Token) -> TokenDescription:
    """A markdown-it token in an error: its type, at its first line."""
    place = (t.map[0] + 1, 1) if t.map else None
    return TokenDescription(t.type, place, place)


def migrate(text: str) -> str:
    """The text with each note written as a GitHub alert, and the rest as it was."""
    tokens = MarkdownIt("commonmark").parse(text)
    starts = line_starts(text)
    parts: list[str] = []
    done = 0
    for m in note.scan(tokens, describe=describe, span=lines_of):
        start, end = starts[m.start], starts[m.end]
        region = text[start:end]
        head, rest = prefixes(region, quoted=m.value.quoted)
        lines = alert(m.value)
        new = [head + lines[0], *(rest + line for line in lines[1:])]
        # The note keeps its own line ending, and has none after its last line where it had none.
        first, last = BREAK.search(region), LAST_BREAK.search(region)
        eol = first[0] if first else "\n"
        parts += [text[done:start], eol.join(new), last[0] if last else ""]
        done = end
    parts.append(text[done:])
    return "".join(parts)


# --8<-- [end:migrate]


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = Path(name)
        path.write_text(migrate(path.read_text(encoding="utf-8")), encoding="utf-8")
