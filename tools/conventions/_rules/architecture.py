"""CON31x: `ARCHITECTURE.md`, the repository's architecture guide, held to the code."""

import ast
import builtins
import importlib
import keyword
import re
import sys
import tomllib
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
# A name in code: dotted, optionally called.
NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?")
# A plain lowercase word, such as a contract's name or a value: not read as code.
WORD = re.compile(r"[a-z][a-z0-9]*")
# The standard library modules whose members a guide cites bare: `Protocol`, `Mapping`.
STDLIB_MEMBERS = (
    "abc",
    "asyncio",
    "collections",
    "collections.abc",
    "concurrent.futures",
    "contextlib",
    "dataclasses",
    "enum",
    "functools",
    "pathlib",
    "socket",
    "ssl",
    "threading",
    "typing",
)
# What a diagram label names in code: a path as above, or a call, `name()` or `Type.name()`. A
# plain word is not read, nor `HTTP/1.1`, which ends in neither `/` nor a suffix, nor a `<br/>`.
DIAGRAM_TOKEN = re.compile(
    rf"(?<![\w./<-])("
    rf"(?:[\w-]+/)*[\w-]+\.(?:{SUFFIXES})"
    r"|(?:[\w-]+/)+(?!\w)"
    r"|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\(\)"
    r")"
)
RUST_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
GUIDANCE = re.compile(r"<!--\s*architecture:")
GUIDANCE_BLOCK = re.compile(r"<!--\s*architecture:.*?-->", re.S)


def _blank(match: re.Match[str]) -> str:
    return re.sub(r"[^\n]", " ", match.group())


def _python_names(tree: ast.Module) -> Iterator[str]:
    """Every name the module defines or uses: definitions, references, attributes, imports, keyword
    arguments, and string constants shaped like a name, as an environment variable read by name is.
    """
    for node in ast.walk(tree):
        match node:
            case ast.ClassDef() | ast.FunctionDef() | ast.AsyncFunctionDef():
                yield node.name
            case ast.Name():
                yield node.id
            case ast.Attribute():
                yield node.attr
            case ast.arg():
                yield node.arg
            case ast.keyword(arg=str() as name):
                yield name
            case ast.alias():
                yield from node.name.split(".")
                if node.asname:
                    yield node.asname
            case ast.ImportFrom(module=str() as module):
                yield from module.split(".")
            case ast.Constant(value=str() as value) if IDENTIFIER.fullmatch(value):
                yield value
            case _:
                pass


def _requirement_names(project: Project, *, own: bool = True) -> Iterator[str]:
    """Every requirement each `pyproject.toml` of the repository names, and with `own` the
    distributions they declare.
    """
    for path in project.files:
        if path.name != "pyproject.toml":
            continue
        try:
            pyproject = tomllib.loads(project.text(path) or "")
        except tomllib.TOMLDecodeError:
            continue
        table = pyproject.get("project", {})
        groups = (table.get("optional-dependencies", {}), pyproject.get("dependency-groups", {}))
        requirements = [
            *([table.get("name", "")] if own else []),
            *table.get("dependencies", []),
            *(r for group in groups for extra in group.values() for r in extra),
        ]
        for requirement in requirements:
            if isinstance(requirement, str) and (match := re.match(r"[\w.-]+", requirement)):
                yield match.group()
                yield match.group().replace("-", "_")


def known_names(project: Project) -> set[str]:
    """Every name a guide may cite: what the repository's Python and Rust define or use, its modules
    and distributions, what they depend on, the keywords, the builtins and the standard library.
    """

    def compute() -> set[str]:
        names = set(dir(builtins)) | set(sys.stdlib_module_names) | set(keyword.kwlist)
        for module in STDLIB_MEMBERS:
            names.update(dir(importlib.import_module(module)))
        names.update(_requirement_names(project))
        for path in project.files:
            if path.suffix in {".py", ".pyi"}:
                names.add(path.stem)
                names.update(path.relative_to(project.root).parent.parts)
                if (tree := project.tree(path)) is not None:
                    names.update(_python_names(tree))
            elif path.suffix == ".rs":
                names.add(path.stem)
                code = RUST_COMMENT.sub("", project.text(path) or "")
                names.update(IDENTIFIER.findall(code))
        return names

    return project.memo("architecture.names", compute)


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


def foreign_heads(project: Project) -> set[str]:
    """The top-level modules a dotted name may start with that this repository does not define:
    the standard library and its dependencies, less its own packages.
    """

    def compute() -> set[str]:
        own = {
            path.relative_to(project.root).parent.name
            for path in project.files
            if path.name == "__init__.py"
        }
        heads = set(sys.stdlib_module_names) | set(_requirement_names(project, own=False))
        return heads - own

    return project.memo("architecture.foreign", compute)


def name_resolves(project: Project, written: str) -> bool:
    """A dotted name under a module this repository does not define resolves unread; any other name
    by its last part.
    """
    parts = written.removesuffix("()").split(".")
    if len(parts) > 1 and parts[0] in foreign_heads(project):
        return True
    return parts[-1] in known_names(project)


def _resolves(project: Project, token: str) -> bool:
    if PATH.fullmatch(token):
        return path_resolves(project, token)
    return name_resolves(project, token)


@rule("CON310", "CON311", "CON312", "CON313")
def architecture(project: Project) -> Iterator[Finding]:
    """The repository has an architecture guide, and the code it names exists.

    CON310: The repository root holds `ARCHITECTURE.md`.

    Reason: it is the map a contributor, human or agent, reads before changing the code: what the
    system is, where each part lives, the main path through it and the invariants a change keeps.
    Without it the map is rebuilt from the source in every session.

    Fix: add `ARCHITECTURE.md`; `CONVENTIONS.md` ("Architecture") says what it holds.

    CON311: Every code span in `ARCHITECTURE.md` that names a path or a name resolves.

    Reason: the guide is read instead of the code, so a codemap entry left behind by a rename or a
    move misleads exactly the reader who relies on it. A path resolves where a listed file or
    directory ends with it; a name, by its last dotted part, where the repository's Python or Rust
    defines or uses it outside a comment, or it is a module, a distribution or dependency any
    `pyproject.toml` names, a keyword, a builtin or in the standard library. A span that is neither
    shape, such as an operator or a command, is not read.

    Fix: rename or drop the entry. A name the code never spells, such as one the guide says does not
    exist, takes `conventions: ignore[CON311]` on its line.

    CON312: Every path and call a diagram in `ARCHITECTURE.md` names resolves.

    Reason: a diagram goes stale the way a codemap does, and is the part a reader trusts at a
    glance. A diagram names code by path (`net/`, `_executor.py`) or by call (`run_sync()`); those
    are read as CON311 reads a span, and a plain word in a label is not.

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
        for token in DIAGRAM_TOKEN.finditer(match.group(3)):
            if not _resolves(project, token.group(1)):
                yield Finding(
                    GUIDE,
                    line_at(text, body_start + token.start(1)),
                    "CON312",
                    f"the diagram names `{token.group(1)}`, which is nowhere in the repository",
                )
    for match in CODE_SPAN.finditer(FENCE.sub(_blank, written)):
        span = match.group(1).strip()
        if WORD.fullmatch(span) or not (PATH.fullmatch(span) or NAME.fullmatch(span)):
            continue
        if not _resolves(project, span):
            yield Finding(
                GUIDE,
                line_at(text, match.start()),
                "CON311",
                f"`{span}` is nowhere in the repository",
            )
