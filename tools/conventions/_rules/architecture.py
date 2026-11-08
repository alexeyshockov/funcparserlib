"""CON31x: `ARCHITECTURE.md`, the repository's architecture guide, held to the code."""

import re
from collections.abc import Iterator
from pathlib import PurePosixPath

from .._engine import Finding, rule
from .._project import Project
from ._ast import line_at

GUIDE = "ARCHITECTURE.md"
FENCE = re.compile(r"^(`{3,}|~{3,})[ \t]*([\w-]*)[^\n]*\n(.*?)^\1[ \t]*$", re.M | re.S)
CODE_SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
SUFFIXES = r"py|pyi|rs|md|toml|ya?ml|json|ini|cfg|just|html|txt|lock"
# A path a guide names: a directory ending in `/`, or a file with one of `SUFFIXES`, either below
# directories.
PATH = re.compile(rf"(?:[\w.-]+/)*[\w.-]+\.(?:{SUFFIXES})|(?:[\w.-]+/)+")
# A path in a diagram label, as above. A plain word is not read, nor `HTTP/1.1`, which ends in
# neither `/` nor a suffix, nor a `<br/>`.
DIAGRAM_PATH = re.compile(rf"(?<![\w./<-])((?:[\w-]+/)*[\w-]+\.(?:{SUFFIXES})|(?:[\w-]+/)+(?!\w))")
GUIDANCE = re.compile(r"<!--\s*architecture:")
GUIDANCE_BLOCK = re.compile(r"<!--\s*architecture:.*?-->", re.S)


def _blank(match: re.Match[str]) -> str:
    return re.sub(r"[^\n]", " ", match.group())


def known_paths(project: Project) -> set[tuple[str, ...]]:
    """Every listed file and every directory holding one, as path components."""

    def compute() -> set[tuple[str, ...]]:
        paths: set[tuple[str, ...]] = set()
        for path in project.files:
            parts = path.relative_to(project.root).parts
            paths.update(parts[:end] for end in range(1, len(parts) + 1))
        return paths

    return project.memo("architecture.paths", compute)


def path_resolves(project: Project, written: str) -> bool:
    """A path resolves where some listed path ends with its components."""
    parts = PurePosixPath(written.removeprefix("./")).parts
    if not parts:
        return True
    return any(path[-len(parts) :] == parts for path in known_paths(project))


@rule("CON310", "CON311", "CON312", "CON313")
def architecture(project: Project) -> Iterator[Finding]:
    """The repository has an architecture guide, and the files and directories it names exist.

    CON310: The repository root holds `ARCHITECTURE.md`.

    Reason: it is the map a contributor, human or agent, reads before changing the code: what the
    system is, where each part lives, the main path through it and the invariants a change keeps.
    Without it the map is rebuilt from the source in every session.

    Fix: add `ARCHITECTURE.md`; `CONVENTIONS.md` ("Architecture") says what it holds.

    CON311: Every path a code span in `ARCHITECTURE.md` names exists.

    Reason: the guide is read instead of the code, so a codemap entry left behind by a move misleads
    exactly the reader who relies on it. A span is a path when it ends in `/` or in a file suffix
    (`net/`, `_executor.py`), and it resolves where a listed file or directory ends with it. A name
    in prose is not read: backticks there also mark environment variables, wire messages and other
    projects' symbols, which no index of this repository can tell from a stale name.

    Fix: correct or drop the path. A path outside the repository, such as another project's file,
    takes `conventions: ignore[CON311]` on its line.

    CON312: Every path a diagram in `ARCHITECTURE.md` names exists.

    Reason: a diagram goes stale the way a codemap does, and is the part a reader trusts at a
    glance. A label names code by path (`net/`, `_executor.py`), read as CON311 reads a span; a
    plain word in a label is not read.

    Fix: update the diagram with the code it draws.

    CON313: `ARCHITECTURE.md` holds no unwritten section.

    Reason: the rendered guide carries an `<!-- architecture: … -->` comment where each section is
    to be written; one left in place is a section nobody wrote.

    Fix: write the section and remove the comment, or remove both where the project has no such
    material.
    """
    path = project.root / GUIDE
    if not path.is_file():
        yield Finding(GUIDE, None, "CON310", "the repository has no ARCHITECTURE.md")
        return
    text = project.text(path) or ""
    for match in GUIDANCE.finditer(FENCE.sub(_blank, text)):
        yield Finding(
            GUIDE, line_at(text, match.start()), "CON313", "a section is still to be written"
        )
    written = GUIDANCE_BLOCK.sub(_blank, text)
    for match in FENCE.finditer(written):
        if match.group(2) != "mermaid":
            continue
        body_start = match.start(3)
        for token in DIAGRAM_PATH.finditer(match.group(3)):
            if not path_resolves(project, token.group(1)):
                yield Finding(
                    GUIDE,
                    line_at(text, body_start + token.start(1)),
                    "CON312",
                    f"the diagram names `{token.group(1)}`, which is nowhere in the repository",
                )
    for match in CODE_SPAN.finditer(FENCE.sub(_blank, written)):
        span = match.group(1).strip()
        if PATH.fullmatch(span) and not path_resolves(project, span):
            yield Finding(
                GUIDE,
                line_at(text, match.start()),
                "CON311",
                f"`{span}` is nowhere in the repository",
            )
