"""Findings, the rule registry, suppressions and the run."""

import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ._project import Project


@dataclass(frozen=True, slots=True, order=True)
class Finding:
    path: str
    line: int | None
    code: str
    message: str


type Check = Callable[[Project], Iterator[Finding]]


@dataclass(frozen=True, slots=True)
class Rule:
    code: str
    check: Check | None
    doc: str

    @property
    def summary(self) -> str:
        return self.doc.strip().splitlines()[0] if self.doc.strip() else ""


RULES: dict[str, Rule] = {}


@dataclass(frozen=True, slots=True)
class Fixes:
    """What a fixer would change: each file's new text, and what it left for a person."""

    rewrites: Mapping[Path, str]
    refused: Sequence[str] = ()


type Fix = Callable[[Project], Fixes]

# Each fixer under every code it resolves; one fixer can resolve several codes.
FIXERS: dict[str, Fix] = {}


def fixer(*codes: str) -> Callable[[Fix], Fix]:
    """Register a fixer for codes some rule already reports."""

    def register(fix: Fix) -> Fix:
        for code in codes:
            if code in FIXERS:
                raise ValueError(f"{code} has two fixers")
            FIXERS[code] = fix
        return fix

    return register


def rule(*codes: str) -> Callable[[Check], Check]:
    """Register a check under the codes it reports; the first code holds the check.

    A check reporting several codes is registered once, under its first, and each other code gets
    its own documentation from the matching `code:` paragraph of the docstring.
    """

    def register(check: Check) -> Check:
        docs = _code_docs(check.__doc__ or "", codes)
        for index, code in enumerate(codes):
            if code in RULES:
                raise ValueError(f"{code} is registered twice")
            RULES[code] = Rule(code, check if index == 0 else None, docs[code])
        return check

    return register


def _code_docs(doc: str, codes: Sequence[str]) -> Mapping[str, str]:
    """Split a docstring into each code's text: `CODE: summary` paragraphs, or the whole for one."""
    if len(codes) == 1:
        return {codes[0]: _dedent(doc)}
    parts = re.split(rf"^\s*({'|'.join(codes)}):\s*", doc, flags=re.MULTILINE)
    found = {code: _dedent(text) for code, text in zip(parts[1::2], parts[2::2], strict=True)}
    if missing := sorted(set(codes) - set(found)):
        raise ValueError(f"no documentation paragraph for {missing}")
    return found


def _dedent(doc: str) -> str:
    lines = doc.strip().splitlines()
    rest = [line for line in lines[1:] if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in rest), default=0)
    return "\n".join([lines[0], *(line[indent:] for line in lines[1:])]).strip() if lines else ""


# Runner codes: they report on the run itself and are never suppressed.
UNPARSABLE = "CON000"
STALE = "CON001"
UNKNOWN = "CON002"

RULES[UNPARSABLE] = Rule(
    UNPARSABLE,
    None,
    """A Python file the rules read does not parse.

Reason: every rule reads the AST; a file that does not parse is checked by none of them.

Fix: correct the syntax. The checks run on the interpreter in `.venv-tools`.""",
)
RULES[STALE] = Rule(
    STALE,
    None,
    """A suppression names a code that does not fire on its line.

Reason: a suppression outlives the finding it excused and silently excuses the next one.

Fix: remove the code from the marker, or the marker.""",
)
RULES[UNKNOWN] = Rule(
    UNKNOWN,
    None,
    """A suppression names a code no rule reports.

Reason: a misspelt code suppresses nothing, and the finding it meant still fires.

Fix: correct the code; `conventions rules` lists them.""",
)

# `conventions: ignore[CON100]` or `conventions: ignore[CON100, CON200]`, in any comment syntax.
MARKER = re.compile(r"conventions:\s*ignore\[(CON\d{3}(?:\s*,\s*CON\d{3})*)\]")

# Files whose comments can carry a marker.
TEXT_SUFFIXES = frozenset(
    {".py", ".pyi", ".toml", ".just", ".yaml", ".yml", ".md", ".ini", ".cfg", ".txt"}
)


def _markers(project: Project) -> Mapping[tuple[str, int], set[str]]:
    found: dict[tuple[str, int], set[str]] = defaultdict(set)
    for path in project.files:
        if path.suffix not in TEXT_SUFFIXES and path.name != "justfile":
            continue
        for number, line in enumerate((project.text(path) or "").splitlines(), 1):
            for match in MARKER.finditer(line):
                codes = {code.strip() for code in match.group(1).split(",")}
                found[project.relative(path), number] |= codes
    return found


def run(project: Project) -> list[Finding]:
    """Every rule over `project`, suppressions applied, sorted by location."""
    raw = [finding for r in RULES.values() if r.check for finding in r.check(project)]
    raw += [
        Finding(project.relative(path), error.lineno, UNPARSABLE, f"does not parse: {error.msg}")
        for path, error in project.unparsable.items()
    ]
    markers = _markers(project)
    fired: dict[tuple[str, int], set[str]] = defaultdict(set)
    for f in raw:
        if f.line is not None:
            fired[f.path, f.line].add(f.code)
    kept = [
        f
        for f in raw
        if f.code in (UNPARSABLE, STALE, UNKNOWN)
        or f.line is None
        or f.code not in markers.get((f.path, f.line), ())
    ]
    for (path, line), codes in markers.items():
        for code in sorted(codes):
            if code not in RULES:
                kept.append(Finding(path, line, UNKNOWN, f"no rule reports {code}"))
            elif code not in fired[path, line]:
                kept.append(Finding(path, line, STALE, f"{code} does not fire on this line"))
    return sorted(kept, key=lambda f: (f.path, f.line or 0, f.code, f.message))
