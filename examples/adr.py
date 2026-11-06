"""A parser of architecture decision records in Nygard's format, and the rules across them.

markdown-it-py tokenizes the Markdown, and a grammar reads its tokens. Nygard's post names the parts
of a record and leaves their Markdown open, so the grammar reads the ways records write them: a
title as `0001 — Title`, `1. Title` or `ADR 1: Title`, and the status and the date as a list or as
adr-tools writes them. Each of those has a text grammar of its own: a status is a closed set, and
one that supersedes gives the number it names. What is wrong is rejected and parsing goes on, so one
run reports every field. The rules that span records, such as a gap in the numbers, are no part of
any one parse: `check()` is a plain function over the parsed records.

Run it on a directory: `python -m examples.adr docs/adr`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from funcparserlib import text
from funcparserlib.parser import (
    Invalid,
    NoParseError,
    ParseErrors,
    Parser,
    TokenDescription,
    finished,
    many,
    some,
)


# --8<-- [start:tree]
@dataclass(frozen=True, slots=True)
class Status:
    """The status of a record; `superseded_by` is the number of the record that replaces it."""

    name: str
    superseded_by: int | None = None


@dataclass(frozen=True, slots=True)
class Title:
    """The title of a record, with its number."""

    number: int
    title: str


@dataclass(frozen=True, slots=True)
class Details:
    """The status and the date of a record."""

    status: Status
    date: date


@dataclass(frozen=True, slots=True)
class Record:
    """A decision record."""

    number: int
    title: str
    status: Status
    date: date
    context: str
    decision: str
    consequences: str


# --8<-- [end:tree]


# --8<-- [start:parts]
def read_day(value: str) -> date:
    """The day `value` names; raises `Invalid` for a day the calendar does not have."""
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise Invalid(f"no such day: {value}") from e


number = text.regex(r"\d+").named("number") >> int
day = text.regex(r"\d{4}-\d{2}-\d{2}").named("a date").validated(read_day)

plain_status = text.lit("Proposed") | text.lit("Accepted") | text.lit("Deprecated")
written = -text.lit("Superseded by ADR ") + number
linked = -text.lit("Superseded by [") + number + -text.regex(r"[^\]]*\]\([^)]*\)")
superseded = written | linked
status = (plain_status >> Status) | (superseded >> (lambda n: Status("Superseded", n)))

title_number = (
    number + -text.lit(" — ")
    | number + -text.lit(". ")
    | -text.lit("ADR ") + number + -text.lit(": ")
)
title = title_number + text.regex(r".+") + -finished >> (lambda p: Title(*p))

listed_status = -text.lit("**Status:** ") + status + -finished
listed_date = -text.lit("**Date:** ") + day + -finished
plain_date = -text.lit("Date: ") + day + -finished
# --8<-- [end:parts]


# --8<-- [start:tokens]
def describe(t: Token) -> TokenDescription:
    """How an error shows a token: its text or its type, on the line it starts."""
    shown = repr(t.content) if t.type == "inline" else t.type
    if t.map is None:
        return TokenDescription(shown)
    place = (t.map[0] + 1, 1)
    return TokenDescription(shown, place, place)


def token(type_: str) -> Parser[Token, Token]:
    """Any one markdown-it token of the type `type_`."""

    def is_type(t: Token) -> bool:
        return t.type == type_

    return some(is_type).named(type_)


def reading[T](grammar: Parser[str, T], label: str) -> Parser[Token, T]:
    """The value `grammar` reads from an `inline` token, which is rejected when `grammar` fails.

    The error is at the token, on its line in the file, and `label` says what was expected.
    """
    return token("inline").validated(lambda t: grammar.parse_text(t.content), label)


def opening(tag: str) -> Parser[Token, Token]:
    """The token that opens a heading: `h1` for `#`."""

    def is_opening(t: Token) -> bool:
        return t.type == "heading_open" and t.tag == tag

    return some(is_opening).named(f"{tag} heading")


def titled(name: str) -> Parser[Token, Token]:
    """The heading `## name`. Any other heading is another part of the record, so this fails."""

    def is_named(t: Token) -> bool:
        return t.content == name

    inline = token("inline").when(is_named).named(f"## {name}")
    return -opening("h2") + inline + -token("heading_close")


def paragraph[T](grammar: Parser[str, T], label: str) -> Parser[Token, T]:
    """The value `grammar` reads from a paragraph."""
    return -token("paragraph_open") + reading(grammar, label) + -token("paragraph_close")


def item[T](grammar: Parser[str, T], label: str) -> Parser[Token, T]:
    """The value `grammar` reads from a list item."""
    return -token("list_item_open") + paragraph(grammar, label) + -token("list_item_close")


def is_inside_section(t: Token) -> bool:
    """Whether a token belongs to the section: anything up to the next `##`."""
    return t.type != "heading_open" or t.tag not in {"h1", "h2"}


def text_of(tokens: list[Token]) -> str:
    """The text of the tokens that have any."""
    return "\n".join(t.content.strip() for t in tokens if t.content)


# --8<-- [end:tokens]


# --8<-- [start:grammar]
STATUSES = "Proposed, Accepted, Deprecated or Superseded by ADR NNNN"

listed = (
    -token("bullet_list_open")
    + item(listed_status, f"**Status:** {STATUSES}")
    + item(listed_date, "**Date:** and a day of the calendar")
    + -token("bullet_list_close")
) >> (lambda p: Details(*p))
adr_tools = (
    paragraph(plain_date, "Date: and a day of the calendar")
    + -titled("Status")
    + paragraph(status + -finished, STATUSES)
) >> (lambda p: Details(p[1], p[0]))


def section(name: str) -> Parser[Token, str]:
    """The text under the heading `## name`."""
    body = many(some(is_inside_section).named("text")) >> text_of
    return -titled(name) + body


def make_record(parts: tuple[Title, Details, str, str, str]) -> Record:
    """A record from the title, the details and the three sections."""
    head, details, context, decision, consequences = parts
    return Record(
        head.number, head.title, details.status, details.date, context, decision, consequences
    )


record = (
    -opening("h1")
    + reading(title, "`NNNN — Title`, `N. Title` or `ADR N: Title`")
    + -token("heading_close")
    + (listed | adr_tools)
    + section("Context")
    + section("Decision")
    + section("Consequences")
    + -finished
) >> make_record
# --8<-- [end:grammar]


# --8<-- [start:parse]
markdown = MarkdownIt("commonmark")


def parse(source: str) -> Record:
    """The record in `source`; raises `ParseErrors` or `NoParseError` for what is not one."""
    return record.parse(markdown.parse(source), describe=describe)


# --8<-- [end:parse]


# --8<-- [start:across]
def check_numbers(records: Mapping[str, Record]) -> Iterable[str]:
    """What is wrong across `records`, which are keyed by file name."""
    numbers = sorted(r.number for r in records.values())
    if numbers != list(range(1, len(numbers) + 1)):
        yield f"the numbers are not 1 to {len(numbers)} without gaps: {numbers}"
    for name, r in records.items():
        if not name.startswith(f"{r.number:04}-"):
            yield f"{name}: the title says {r.number:04}"
        target = r.status.superseded_by
        if target == r.number:
            yield f"{name}: superseded by itself"
        elif target is not None and target not in numbers:
            yield f"{name}: superseded by ADR {target:04}, which does not exist"


def check(directory: Path) -> list[str]:
    """What is wrong in the records `NNNN-*.md` of `directory`, across them once each one parses."""
    problems: list[str] = []
    records: dict[str, Record] = {}
    for path in sorted(directory.glob("[0-9][0-9][0-9][0-9]-*.md")):
        source = path.read_text(encoding="utf-8")
        try:
            records[path.name] = parse(source)
        except (NoParseError, ParseErrors) as e:
            problems.append(f"{path.name}:\n{e.render(source)}")
    if not problems and not records:
        return [f"no records in {directory}"]
    return problems if problems else list(check_numbers(records))


# --8<-- [end:across]

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m examples.adr DIRECTORY")
    if found := check(Path(sys.argv[1])):
        sys.exit("\n\n".join(found))
