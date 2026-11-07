"""Migrating a file format: what changed between two versions is rewritten, and the rest stays.

Goal: a whole-file rewrite with `Parser.replace_text()`, where text around the old constructs is
kept to the character, a code block is left alone, and a broken construct stops the migration.
"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from funcparserlib.parser import Invalid, Located, commit, lit, regex


# --8<-- [start:constructs]
@dataclass(frozen=True)
class Link:
    """`[[target|title]]` in the old format."""

    target: str
    title: str


@dataclass(frozen=True)
class Due:
    """`@due(2026-11-07)` in the old format."""

    day: date


def day(text: str) -> date:
    """The date of `text`, rejected when the calendar does not have it."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise Invalid("a date") from None


name = regex(r"[^|\]\n]+").named("name")
link = commit((-lit("[[") + name + -lit("|") + name + -lit("]]")).starmap(Link))
due = commit(-lit("@due(") + regex(r"\d{4}-\d\d-\d\d").validated(day) + -lit(")")) >> Due
# --8<-- [end:constructs]

# --8<-- [start:fence]
fence = regex(r"^```.*?^```$", re.MULTILINE | re.DOTALL)
construct = fence | link | due
# --8<-- [end:fence]


# --8<-- [start:migrate]
def render(m: Located[str | Link | Due]) -> str:
    """The new format of a construct; a code block stays as it was read."""
    match m.value:
        case Link(target, title):
            return f"[{title}]({target})"
        case Due(when):
            return f"*due {when.isoformat()}*"
        case text:
            return text


def migrate(text: str) -> str:
    """The `text` in the new format."""
    return construct.replace_text(text, render)


# --8<-- [end:migrate]


# --8<-- [start:file]
def migrate_file(path: Path) -> None:
    """Rewrite the file at `path`, which stays as it is when a construct in it is broken."""
    path.write_text(migrate(path.read_text()))


# --8<-- [end:file]
