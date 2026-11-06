"""A Keep a Changelog parser that enforces the format's rules.

markdown-it-py tokenizes the Markdown, and a grammar reads its tokens: which headings there are, and
in which order. What a heading says has a grammar of its own, a text grammar of a version, a date
and a kind of change. What sits under a heading is kept as text. `Parser.validated()` rejects what
is wrong and parsing goes on, so one run reports all of it, and `Parser.recover()` leaves a broken
release out of the result.

Versions follow PEP 440, such as `1.0`, `2.0.0b1` and `1.2.post3.dev4`.

Run it on a file: `python -m examples.changelog CHANGELOG.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

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
    maybe,
    some,
)


# --8<-- [start:tree]
@dataclass(frozen=True, slots=True)
class Version:
    """A PEP 440 version: `1.2rc1.post3.dev4` has each of the parts."""

    release: tuple[int, ...]
    pre: tuple[str, int] | None
    post: int | None
    dev: int | None

    def __str__(self) -> str:
        """The version as it is written, for an error."""
        pre = "".join(map(str, self.pre or ()))
        post = "" if self.post is None else f".post{self.post}"
        dev = "" if self.dev is None else f".dev{self.dev}"
        return ".".join(map(str, self.release)) + pre + post + dev

    @property
    def order(self) -> tuple[list[int], tuple[str, int], int, bool, int]:
        """What versions sort by: `1.0.dev1 < 1.0a1 < 1.0rc1 < 1.0 == 1.0.0 < 1.0.post1`."""
        release = list(self.release)
        while release and release[-1] == 0:
            release.pop()
        only_dev = self.dev is not None and self.pre is None and self.post is None
        pre = self.pre or (("", 0) if only_dev else ("z", 0))
        post = -1 if self.post is None else self.post
        return (release, pre, post, self.dev is None, self.dev or 0)


@dataclass(frozen=True, slots=True)
class Title:
    """A release heading; `Unreleased` has no `version` and no `date`."""

    version: Version | None
    date: date | None
    yanked: bool


@dataclass(frozen=True, slots=True)
class Changes:
    """The changes of one `kind`, such as `Added`."""

    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class Release:
    """A release, or `Unreleased`: its title, the text before its changes, and its changes."""

    title: Title
    notes: str
    changes: Sequence[Changes]


# --8<-- [end:tree]


# --8<-- [start:version]
number = text.regex(r"\d+").named("number") >> int
release_segment = number + many(-text.lit(".") + number) >> (lambda p: (p[0], *p[1]))
pre_release = (text.lit("a") | text.lit("b") | text.lit("rc")) + number
post_release = -text.lit(".post") + number
dev_release = -text.lit(".dev") + number
version = (
    release_segment + maybe(pre_release) + maybe(post_release) + maybe(dev_release)
    >> (lambda p: Version(*p))
).named("version")
# --8<-- [end:version]


# --8<-- [start:date]
def read_day(value: str) -> date:
    """The day `value` names; raises `Invalid` for a day the calendar does not have."""
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise Invalid(f"no such day: {value}") from e


day = text.regex(r"\d{4}-\d{2}-\d{2}").named("a date").validated(read_day)
# --8<-- [end:date]


# --8<-- [start:headings]
unreleased = text.lit("[Unreleased]") >> (lambda _: Title(None, None, yanked=False))
separator = -(text.lit(" - ") | text.lit(" — "))
yanked = maybe(text.lit(" [YANKED]")) >> (lambda y: y is not None)
numbered = -text.lit("[") + version + -text.lit("]") + separator + day + yanked >> (
    lambda p: Title(*p)
)
release_title = (unreleased | numbered) + -finished

change_kind = (
    text.lit("Added")
    | text.lit("Changed")
    | text.lit("Deprecated")
    | text.lit("Removed")
    | text.lit("Fixed")
    | text.lit("Security")
) + -finished
# --8<-- [end:headings]


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
    """The token that opens a heading: `h2` for `##`."""

    def is_opening(t: Token) -> bool:
        return t.type == "heading_open" and t.tag == tag

    return some(is_opening).named(f"{tag} heading")


def heading[T](tag: str, grammar: Parser[str, T], label: str) -> Parser[Token, T]:
    """The value `grammar` reads from a heading's source text, a link as it is written."""
    return -opening(tag) + reading(grammar, label) + -token("heading_close")


def text_under(level: int) -> Parser[Token, str]:
    """The text of every token up to a heading of `level` or a higher one."""

    def is_inside(t: Token) -> bool:
        return t.type != "heading_open" or int(t.tag[1:]) > level

    def text_of(tokens: list[Token]) -> str:
        return "\n".join(t.content.strip() for t in tokens if t.content)

    return many(some(is_inside).named("text")) >> text_of


# --8<-- [end:tokens]


# --8<-- [start:grammar]
def with_unique_kinds(release: Release) -> Release:
    """The release, with an error for a kind of change that has two sections."""
    kinds = [changes.kind for changes in release.changes]
    for kind in kinds:
        if kinds.count(kind) > 1:
            raise Invalid(f"{kind} appears twice in a release", keep=True)
    return release


def newest_first(releases: list[Release]) -> list[Release]:
    """The releases, with an error where `Unreleased`, a version or a date is out of order."""
    titles = [r.title for r in releases]
    if any(t.version is None for t in titles[1:]):
        raise Invalid("Unreleased is not the first release", keep=True)
    numbered = [(t.version, t) for t in titles if t.version]
    for (older, first), (newer, second) in pairwise(numbered):
        if older.order <= newer.order:
            raise Invalid(f"{newer} comes after {older}", keep=True)
        if first.date and second.date and first.date < second.date:
            raise Invalid(f"{newer} has a later date than {older}", keep=True)
    return releases


title = heading("h2", release_title, "`[version] - date` or `[Unreleased]`")
kind = heading("h3", change_kind, "Added, Changed, Deprecated, Removed, Fixed or Security")
changes = kind + text_under(3) >> (lambda p: Changes(*p))
release = (title + text_under(3) + many(changes) >> (lambda p: Release(*p))).validated(
    with_unique_kinds
)
releases = (many(release.recover()) >> (lambda found: [r for r in found if r])).validated(
    newest_first
)
first_heading = -opening("h1") + -token("inline") + -token("heading_close")
changelog = first_heading + -text_under(2) + releases + -finished
# --8<-- [end:grammar]


# --8<-- [start:parse]
markdown = MarkdownIt("commonmark")


def parse(source: str) -> list[Release]:
    """The releases of a changelog; raises `ParseErrors` or `NoParseError` for a broken rule."""
    return changelog.parse(markdown.parse(source), describe=describe)


# --8<-- [end:parse]

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        sys.exit("usage: python -m examples.changelog FILE")
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    try:
        parse(source)
    except (NoParseError, ParseErrors) as e:
        sys.exit(e.render(source))
