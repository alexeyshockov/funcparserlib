"""CON9xx: the documentation site, held to itself.

The rules apply when the repository root holds `zensical.toml`, the site's configuration; a `docs/`
directory alone is not a site (it may hold only `docs/adr/`). Pages are the Markdown files under
`docs/` as git lists them. Include paths resolve against the repository root, as `pymdownx.snippets`
is configured to.
"""

import ast
import re
import textwrap
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import griffe

from .._engine import Finding, Fixes, fixer, rule
from .._project import Project, RunError

# A snippet marker line, e.g. `# --8<-- [start:task1]`; pymdownx strips the whole line, whatever
# its indent and comment character.
MARKER = re.compile(r"-{2}8<-{2}\s*\[(start|end):([\w.\-]+)\]")
# A one-line include body: `--8<-- "examples/web/app.py:tasks"`.
INCLUDE = re.compile(r'^\s*-{2}8<-{2}\s*"([^":]+):([\w.\-]+)"\s*$')
# A whole-file include: `--8<-- "benchmarks/FREETHREADING.md"`.
INCLUDE_FILE = re.compile(r'^\s*-{2}8<-{2}\s*"([^":]+)"\s*$')
# Anything meant to be an include: two dashes, an 8 and a chevron, however spaced.
NEAR_INCLUDE = re.compile(r'^\s*-\s*-\s*8\s*<\s*-\s*-\s*"')
# The steering comment above a highlighted include: `<!-- hl: setup call -->`, one or more names.
HL_COMMENT = re.compile(r"^\s*<!--\s*hl:\s*([\w.\-]+(?:\s+[\w.\-]+)*)\s*-->\s*$")
# A fenced code block opener with an info string: ```python hl_lines="7-11"
FENCE = re.compile(r"^\s*```+\s*([\w-]*)\s*(.*)$")
HL_ATTR = re.compile(r'hl_lines="([^"]*)"')
# A fence opener up to its language, or up to its closing brace: the place a missing `hl_lines`
# goes.
FENCE_LANG = re.compile(r"^\s*```+\s*(?:\{[^}]*(?=\})|[\w-]*)")
# An inline code span, stripped before a line is read for links.
CODE_SPAN = re.compile(r"`+[^`]*`+")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE_EDGE = re.compile(r"^\s*(`{3,}|~{3,})")
# A python fence, however its info string grows: `python`, `{.python}`, `python title="x.py"`.
PY_FENCE = re.compile(r"^\s*```+\s*\{?\.?(?:python|py)\b")
EXTERNAL = ("http://", "https://", "mailto:", "#")
ADR_HUB = "adr/index.md"


def _site(project: Project) -> bool:
    return (project.root / "zensical.toml").is_file()


def _docs(project: Project) -> Path:
    return project.root / "docs"


def _pages(project: Project) -> list[Path]:
    """The site's Markdown pages, as git lists them."""
    return sorted(project.under([_docs(project)], frozenset({".md"})))


def _lines(project: Project, path: Path) -> Sequence[str]:
    return (project.text(path) or "").splitlines()


def _prose(lines: Sequence[str]) -> Iterator[tuple[int, str]]:
    """The lines outside fenced code blocks, numbered from 1."""
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if FENCE_EDGE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield number, line


def _links(line: str) -> list[str]:
    return LINK.findall(CODE_SPAN.sub("", line))


# --- hl_lines ---


def _find_marker(lines: Sequence[str], kind: str, name: str) -> int | None:
    for i, line in enumerate(lines):
        m = MARKER.search(line)
        if m and m.group(1) == kind and m.group(2) == name:
            return i
    return None


def expected_hl(lines: Sequence[str], section: str | None, sub: str) -> set[int]:
    """1-indexed line positions of `sub` within the rendered `section`, the whole file for `None`.

    Raises:
        LookupError: `section` or `sub` is not delimited by its markers, or `sub` renders empty.
    """
    start: int | None = -1
    end: int | None = len(lines)
    if section is not None:
        start, end = _find_marker(lines, "start", section), _find_marker(lines, "end", section)
    if start is None or end is None:
        raise LookupError(f"section {section!r} not delimited by [start/end:{section}] markers")
    rendered = [i for i in range(start + 1, end) if not MARKER.search(lines[i])]
    sub_start = _find_marker(lines, "start", sub)
    sub_end = _find_marker(lines, "end", sub)
    if sub_start is None or sub_end is None:
        raise LookupError(f"sub-section {sub!r} not delimited by [start/end:{sub}] markers")
    positions = {pos for pos, orig in enumerate(rendered, start=1) if sub_start < orig < sub_end}
    if not positions:
        where = "the file" if section is None else f"section {section!r}"
        raise LookupError(f"sub-section {sub!r} maps to no rendered lines of {where}")
    return positions


def parse_hl(raw: str) -> set[int]:
    out: set[int] = set()
    for token in raw.split():
        if "-" in token:
            lo, hi = token.split("-", 1)
            out |= set(range(int(lo), int(hi) + 1))
        elif token:
            out.add(int(token))
    return out


def fmt_hl(numbers: set[int]) -> str:
    runs: list[str] = []
    ordered = sorted(numbers)
    lo = prev = ordered[0]
    for n in [*ordered[1:], None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        runs.append(f"{lo}-{prev}" if prev > lo else f"{lo}")
        if n is not None:
            lo = prev = n
    return " ".join(runs)


def expected_hls(lines: Sequence[str], section: str | None, subs: Sequence[str]) -> set[int]:
    """The union of `expected_hl` over `subs`.

    Raises:
        LookupError: as `expected_hl`, for the first sub-section that fails.
    """
    return set().union(*(expected_hl(lines, section, sub) for sub in subs))


@dataclass(frozen=True, slots=True)
class HlFence:
    line: int
    # `None` where the fence has no `hl_lines` attribute.
    hl_raw: str | None
    include: str
    # `None` for a whole-file include.
    section: str | None
    # The names in the steering comment; empty without one.
    subs: Sequence[str]


def where(fence: HlFence) -> str:
    return fence.include if fence.section is None else f"section {fence.section!r}"


def hl_fences(lines: Sequence[str]) -> list[HlFence]:
    """Every single-include fence under a steering comment or with `hl_lines`."""
    found: list[HlFence] = []
    i = 0
    while i < len(lines):
        fence = FENCE.match(lines[i])
        if not fence or not lines[i].lstrip().startswith("```"):
            i += 1
            continue
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not lines[j].lstrip().startswith("```"):
            body.append(lines[j])
            j += 1
        hl = HL_ATTR.search(fence.group(2))
        include = (
            (INCLUDE.match(body[0]) or INCLUDE_FILE.match(body[0])) if len(body) == 1 else None
        )
        k = i - 1
        while k >= 0 and not lines[k].strip():
            k -= 1
        comment = HL_COMMENT.match(lines[k]) if k >= 0 else None
        subs = tuple(comment.group(1).split()) if comment else ()
        if (hl or subs) and include:
            raw = hl.group(1) if hl else None
            section = include.group(2) if include.re is INCLUDE else None
            found.append(HlFence(i + 1, raw, include.group(1), section, subs))
        i = j + 1
    return found


def stray_comments(lines: Sequence[str]) -> list[int]:
    """The prose lines holding a steering comment that no single-include fence follows."""
    steered = {fence.line for fence in hl_fences(lines) if fence.subs}
    stray: list[int] = []
    for number, line in _prose(lines):
        if not HL_COMMENT.match(line):
            continue
        k = number
        while k < len(lines) and not lines[k].strip():
            k += 1
        if k + 1 not in steered:
            stray.append(number)
    return stray


@rule("CON900", "CON901", "CON902")
def highlights(project: Project) -> Iterator[Finding]:
    """`hl_lines` on an included example section is derived from the example's markers.

    CON900: A highlighted include names its sub-sections in a `<!-- hl: NAME ... -->` comment, and
    every such comment sits above a fence holding one include.

    Reason: the comment is what the derivation reads; without it the numbers are hand-counted and
    drift the moment the example changes. A comment anywhere else highlights nothing.

    Fix: put `<!-- hl: NAME -->` on the line above the fence, and delimit `NAME` in the example with
    `[start:NAME]` and `[end:NAME]` markers inside the included section. Several names, separated by
    spaces, highlight each one's lines. A fence holding several includes becomes one fence per
    include.

    CON901: The section and sub-section a highlight names are delimited in the example.

    Reason: the derivation counts the lines between the markers; a missing marker leaves nothing to
    count.

    Fix: add the `[start:NAME]` / `[end:NAME]` markers, or correct the name.

    CON902: A fence under a steering comment carries `hl_lines`, the ranges its sub-sections'
    markers give.

    Reason: `pymdownx` strips every marker line, so the rendered lines between a sub-section's
    markers are the highlight; any other number highlights the wrong code, and a missing `hl_lines`
    highlights nothing. The build notices neither.

    Fix: write the ranges in the message, or run the fixer.
    """
    if not _site(project):
        return
    for page in _pages(project):
        relative = project.relative(page)
        lines = _lines(project, page)
        for number in stray_comments(lines):
            yield Finding(
                relative,
                number,
                "CON900",
                "`<!-- hl: ... -->` highlights nothing: no fence holding one include follows it",
            )
        for fence in hl_fences(lines):
            if not fence.subs:
                yield Finding(
                    relative,
                    fence.line,
                    "CON900",
                    f"`hl_lines` on an include of {where(fence)} needs a preceding "
                    "`<!-- hl: <sub-section> -->` comment",
                )
                continue
            source = project.root / fence.include
            try:
                want = expected_hls(
                    (project.text(source) or "").splitlines() if source.is_file() else [],
                    fence.section,
                    fence.subs,
                )
            except LookupError as e:
                yield Finding(relative, fence.line, "CON901", f"{fence.include}: {e}")
                continue
            names = " ".join(fence.subs)
            if fence.hl_raw is None:
                yield Finding(
                    relative,
                    fence.line,
                    "CON902",
                    f'hl:{names} has no hl_lines, should be "{fmt_hl(want)}"',
                )
            elif parse_hl(fence.hl_raw) != want:
                yield Finding(
                    relative,
                    fence.line,
                    "CON902",
                    f'hl:{names} has hl_lines="{fence.hl_raw}", should be "{fmt_hl(want)}"',
                )


# --- Includes ---


@rule("CON903", "CON904", "CON905")
def includes(project: Project) -> Iterator[Finding]:
    """An include is spelled the way `pymdownx.snippets` reads it, and names what is there.

    CON903: An include is spelled `--8<-- "path:section"` or `--8<-- "path"`.

    Reason: the extension matches the exact spelling. A near miss is not an error and not a warning:
    the page renders a code block containing the marker text.

    Fix: write two dashes, `8<`, two dashes, one space, then the quoted target.

    CON904: An included file exists.

    Reason: the build does not fail on a missing include target.

    Fix: correct the path, relative to the repository root.

    CON905: An included section is delimited in its file.

    Reason: a section name with no `[start:NAME]` marker includes nothing.

    Fix: correct the name, or add the markers to the file.
    """
    if not _site(project):
        return
    for page in _pages(project):
        relative = project.relative(page)
        for number, line in enumerate(_lines(project, page), 1):
            if NEAR_INCLUDE.match(line) and not (INCLUDE.match(line) or INCLUDE_FILE.match(line)):
                yield Finding(relative, number, "CON903", f"malformed include: {line.strip()}")
                continue
            if not (found := INCLUDE.match(line)):
                continue
            path, section = found.group(1), found.group(2)
            source = project.root / path
            if not source.exists():
                yield Finding(relative, number, "CON904", f"no such file {path}")
            elif f"[start:{section}]" not in (project.text(source) or ""):
                yield Finding(relative, number, "CON905", f"{path} has no section {section!r}")


# --- Structure ---


def nav_pages(project: Project) -> list[str]:
    """Every page `nav` names, docs-relative, in menu order; empty when there is no `nav`."""
    config = tomllib.loads(project.text(project.root / "zensical.toml") or "")
    pages: list[str] = []

    def walk(entry: object) -> None:
        match entry:
            case list():
                for item in entry:
                    walk(item)
            case dict():
                for value in entry.values():
                    walk(value)
            case str():
                pages.append(entry)
            case _:
                pass

    walk(config.get("project", {}).get("nav", []))
    return pages


def reachable_pages(project: Project) -> set[str]:
    """Every page a reader can arrive at from `nav`, following links transitively."""
    docs = _docs(project)
    seen: set[str] = set()
    queue = [page for page in nav_pages(project) if (docs / page).is_file()]
    while queue:
        page = queue.pop()
        if page in seen:
            continue
        seen.add(page)
        here = docs / page
        for line in _lines(project, here):
            for target in _links(line):
                if target.startswith(EXTERNAL):
                    continue
                path = target.split("#", 1)[0]
                if not path.endswith(".md"):
                    continue
                landed = (here.parent / path).resolve()
                if landed.is_file() and landed.is_relative_to(docs):
                    queue.append(landed.relative_to(docs).as_posix())
    return seen


@rule("CON906", "CON907", "CON908", "CON909", "CON914")
def structure(project: Project) -> Iterator[Finding]:
    """Every page is in the menu or reachable from it.

    CON906: Every `nav` entry is a page that exists.

    Reason: a menu entry with no page is a dead link in every page's navigation.

    Fix: correct the entry in `zensical.toml`, or add the page.

    CON907: Every page is reachable from the menu, directly or through links.

    Reason: a page no route enters is published and findable only by search. Reachability is
    transitive, so a hub listing its children makes them all reachable.

    Fix: link the page from a reachable page, or add it to `nav`.

    CON908: The ADR hub, `docs/adr/index.md`, is in the menu or linked from a page outside
    `docs/adr/`.

    Reason: each route alone can be dropped in good faith; both going at once hides every decision.

    Fix: add the hub to `nav`, or link to it from a page.

    CON909: Every numbered ADR is linked from the ADR index.

    Reason: the menu names the hub, not the decisions, so the index is the only route to each.

    Fix: add a link to the ADR on `docs/adr/index.md`.

    CON914: No page under `docs/adr/` is in the menu but the hub.

    Reason: the hub lists every decision (CON909); a menu listing them too is a second list to keep
    in step with it, and the one that drifts.

    Fix: drop the entry from `nav`; the hub links the ADR.
    """
    if not _site(project):
        return
    docs = _docs(project)
    config = project.root / "zensical.toml"
    nav = nav_pages(project)
    for page in nav:
        if not (docs / page).is_file():
            yield Finding(
                "zensical.toml",
                project.line_of(config, f'"{page}"'),
                "CON906",
                f"nav names {page}, which does not exist",
            )
    reachable = reachable_pages(project)
    for page in _pages(project):
        if page.relative_to(docs).as_posix() not in reachable:
            yield Finding(project.relative(page), None, "CON907", "no route from the menu")
    adr = docs / "adr"
    hub = adr / "index.md"
    if ADR_HUB not in nav:
        linkers = [
            page
            for page in _pages(project)
            if not page.is_relative_to(adr)
            for target in LINK.findall(project.text(page) or "")
            if target.split("#")[0].endswith(ADR_HUB)
        ]
        if not linkers:
            yield Finding(
                f"docs/{ADR_HUB}",
                None,
                "CON908",
                "the ADR hub is in no menu and no page links to it",
            )
    for page in nav:
        if page.startswith("adr/") and page != ADR_HUB:
            yield Finding(
                "zensical.toml",
                project.line_of(config, f'"{page}"'),
                "CON914",
                f"nav lists {page}; the ADR hub is its only route",
            )
    numbered = sorted(p.name for p in adr.glob("[0-9]*.md"))
    index = project.text(hub) if hub.is_file() else None
    for name in numbered:
        if index is None or f"({name})" not in index:
            yield Finding(f"docs/{ADR_HUB}", None, "CON909", f"the ADR index does not link {name}")


@rule("CON915")
def llms_txt(project: Project) -> Iterator[Finding]:
    """The site publishes `llms.txt` and `llms-full.txt`.

    Reason: `llms.txt` is the index a language model reads a site through, and `llms-full.txt` hands
    it every page at once; without them it scrapes the rendered HTML.

    Fix: add `[project.plugins.llmstxt]` with `full_output = "llms-full.txt"` to `zensical.toml`,
    and a section per distribution under `[project.plugins.llmstxt.sections]`.
    """
    if not _site(project):
        return
    config = tomllib.loads(project.text(project.root / "zensical.toml") or "")
    llmstxt = config.get("project", {}).get("plugins", {}).get("llmstxt")
    if not isinstance(llmstxt, dict) or llmstxt.get("enabled", True) is False:
        yield Finding("zensical.toml", None, "CON915", "no llms.txt: `llmstxt` is not configured")
    elif not llmstxt.get("full_output"):
        yield Finding("zensical.toml", None, "CON915", "no llms-full.txt: `full_output` is unset")


# --- The API reference ---

# A mkdocstrings block: `::: package.module` alone on its line.
DIRECTIVE = re.compile(r"^:::\s+([\w.]+)\s*$")
# mkdocstrings-python's default `filters`: no single-underscore name.
DEFAULT_FILTERS = ("!^_[^_]",)


def _python_handler(project: Project) -> dict[str, object]:
    config = tomllib.loads(project.text(project.root / "zensical.toml") or "")
    plugins = config.get("project", {}).get("plugins", {})
    return plugins.get("mkdocstrings", {}).get("handlers", {}).get("python", {})


def _kept(name: str, filters: Sequence[tuple[re.Pattern[str], bool]]) -> bool:
    """Whether mkdocstrings-python's `filters` render `name`: the last matching filter decides; with
    no match, a name is kept unless every filter is an inclusion.
    """
    keep: bool | None = None
    for pattern, exclude in filters:
        if pattern.search(name):
            keep = not exclude
    if keep is None:
        return {exclude for _, exclude in filters} != {False}
    return keep


def _reference_targets(project: Project) -> list[str]:
    return [
        found.group(1)
        for page in _pages(project)
        for _, line in _prose(_lines(project, page))
        if (found := DIRECTIVE.match(line))
    ]


def _extension(project: Project, spec: object) -> str | dict[str, dict[str, object]]:
    """A griffe extension as `zensical.toml` names it; a script path is read from the root."""
    if isinstance(spec, str):
        return str(project.root / spec) if spec.split(":", 1)[0].endswith(".py") else spec
    if isinstance(spec, dict):
        return spec
    raise RunError(f"zensical.toml names a griffe extension griffe cannot read: {spec!r}")


@rule("CON916")
def reference_docstrings(project: Project) -> Iterator[Finding]:
    """Every class, function, method and property the API reference renders has a docstring.

    Reason: a signature alone does not say what a call does, what it blocks on or when it raises.
    Writing the docstring states the contract; a name nobody can state one for is a name the API
    should not export. A field is exempt: its name and type are its contract.

    Fix: write the docstring, or make the name private.

    Scope: what each `::: dotted.path` block on a page renders, as `zensical.toml` configures the
    python handler: its `paths`, and the `filters` and `extensions` among its `options`; a block's
    own options are not read. A module renders its `__all__`, or without one its own public members;
    a class its own members, not inherited ones. A docstring an extension copies from a base
    (griffe-inherited-docstrings) counts. A name re-exported from a package no block names is not
    followed.
    """
    if not _site(project):
        return
    targets = _reference_targets(project)
    if not targets:
        return
    handler = _python_handler(project)
    options = handler.get("options", {})
    assert isinstance(options, dict)
    filters = [
        (re.compile(f.removeprefix("!")), f.startswith("!"))
        for f in options.get("filters", DEFAULT_FILTERS)
    ]
    paths = handler.get("paths", ["."])
    assert isinstance(paths, list)
    try:
        extensions = griffe.load_extensions(
            *(_extension(project, spec) for spec in options.get("extensions", []))
        )
        loader = griffe.GriffeLoader(
            search_paths=[str(project.root / path) for path in paths], extensions=extensions
        )
        for target in dict.fromkeys(target.split(".")[0] for target in targets):
            loader.load(target)
        loader.resolve_aliases(implicit=False, external=False)
    except (ImportError, griffe.GriffeError) as e:
        raise RunError(f"griffe cannot read the API reference's targets: {e}") from None
    seen: set[str] = set()

    def check(obj: griffe.Object, shown: str) -> Iterator[Finding]:
        if obj.path in seen:
            return
        seen.add(obj.path)
        if not obj.docstring and obj.filepath is not None and not isinstance(obj.filepath, list):
            path = project.relative(obj.filepath)
            yield Finding(path, obj.lineno, "CON916", f"{shown} has no docstring")

    def members(cls: griffe.Class, shown: str) -> Iterator[Finding]:
        yield from check(cls, shown)
        for name, member in cls.members.items():
            if isinstance(member, griffe.Alias) or not _kept(name, filters):
                continue
            if isinstance(member, griffe.Class):
                yield from members(member, f"{shown}.{name}")
            elif member.is_function or (member.is_attribute and "property" in member.labels):
                yield from check(member, f"{shown}.{name}")

    for target in targets:
        try:
            module = loader.modules_collection[target]
        except KeyError:
            raise RunError(f"griffe finds no module {target}, which a `:::` block names") from None
        for name, member in module.members.items():
            # Exports first: asking an import what it is resolves it; `__future__` does not resolve.
            if (module.exports is not None or member.is_alias) and not member.is_exported:
                continue
            if not _kept(name, filters):
                continue
            try:
                obj = member.final_target if member.is_alias else member
            except griffe.AliasResolutionError:
                continue
            if isinstance(obj, griffe.Class):
                yield from members(obj, f"{target}.{name}")
            elif obj.is_function:
                yield from check(obj, f"{target}.{name}")


# --- Links ---


def pypi_readmes(project: Project) -> list[tuple[Path, Path]]:
    """`(pyproject, readme)` for each `readme =`: the repository's and each package's."""
    candidates = [
        project.root / "pyproject.toml",
        *(root.parent / "pyproject.toml" for root in project.package_roots.values()),
    ]
    found = []
    for pyproject in dict.fromkeys(path for path in candidates if path.is_file()):
        readme = tomllib.loads(project.text(pyproject) or "").get("project", {}).get("readme")
        name = readme.get("file") if isinstance(readme, dict) else readme
        if name:
            found.append((pyproject, pyproject.parent / name))
    return found


@rule("CON910", "CON911", "CON912")
def links(project: Project) -> Iterator[Finding]:
    """A link resolves where it is read.

    CON910: A relative link between pages resolves.

    Reason: the build does not fail on a dead internal link, so nothing else reports one.

    Fix: correct the target, relative to the page.

    CON911: A README that ships to PyPI carries no relative link.

    Reason: `readme =` names the file that becomes the PyPI project page, rendered with no
    repository under it, where a relative link is a 404. Every one resolves in a checkout.

    Fix: link absolutely, to the published site or the repository.

    CON912: A `readme =` names a file that exists.

    Reason: the build of the distribution fails on it, late.

    Fix: correct the name in `pyproject.toml`.
    """
    if not _site(project):
        return
    for page in _pages(project):
        for number, line in _prose(_lines(project, page)):
            for target in _links(line):
                if target.startswith(EXTERNAL):
                    continue
                path = target.split("#", 1)[0]
                if path and not (page.parent / path).resolve().exists():
                    yield Finding(
                        project.relative(page),
                        number,
                        "CON910",
                        f"link to {target!r} resolves to nothing",
                    )
    for pyproject, readme in pypi_readmes(project):
        if not readme.is_file():
            yield Finding(
                project.relative(pyproject),
                project.line_of(pyproject, "readme"),
                "CON912",
                f"`readme =` names {readme.name}, which does not exist",
            )
            continue
        for number, line in _prose(_lines(project, readme)):
            for target in _links(line):
                if not target.startswith(EXTERNAL):
                    yield Finding(
                        project.relative(readme), number, "CON911", f"relative link {target!r}"
                    )


# --- Snippets ---


def python_blocks(lines: Sequence[str]) -> list[tuple[int, str]]:
    """Every fenced python block, as `(first content line, dedented source)`."""
    blocks, i = [], 0
    while i < len(lines):
        if PY_FENCE.match(lines[i]):
            start = i + 1
            end = start
            while end < len(lines) and not lines[end].lstrip().startswith("```"):
                end += 1
            blocks.append((start + 1, textwrap.dedent("\n".join(lines[start:end]))))
            i = end
        i += 1
    return blocks


@rule("CON913")
def snippets_parse(project: Project) -> Iterator[Finding]:
    """A python block on a page parses.

    Reason: a block that does not parse cannot be run or checked, and readers copy it.

    Fix: correct the block, or mark it as another language if it is not Python. A block indented
    inside a list item is dedented before parsing.
    """
    if not _site(project):
        return
    for page in _pages(project):
        for first, source in python_blocks(_lines(project, page)):
            try:
                ast.parse(source)
            except SyntaxError as e:
                yield Finding(
                    project.relative(page),
                    first + (e.lineno or 1) - 1,
                    "CON913",
                    f"does not parse: {e.msg}",
                )


# --- Snippets that claim to be complete ---

# Context-manager helpers the pages lean on, in no package's public surface.
CONTEXT_HELPERS = frozenset({"closing", "aclosing", "contextmanager", "asynccontextmanager"})
# Names a module-level public surface never holds: a vocabulary containing one read locals.
NEVER_PUBLIC = frozenset({"conn", "settings"})


def _source_root(project: Project, package: str) -> Path:
    """Where `package`'s source is: one of the project's packages, or a dependency in `.venv`."""
    if package in project.package_roots:
        return project.package_roots[package]
    for site in project.site_packages:
        if (site / package).is_dir():
            return site / package
    raise RunError(
        f"{package} is in [tool.conventions.docs] vocabulary, and it is neither one of "
        "[tool.conventions] packages nor installed in .venv"
    )


def public_surface(project: Project, package: str) -> set[str]:
    """The names a caller can import from `package`, read from its source by griffe.

    A module is public when no part of its path starts with an underscore. A public module offers
    the names it exports (its `__all__`), submodules included; griffe imports nothing to read them.
    A package that is not the project's own is read from its copy in `.venv`.
    """
    root = _source_root(project, package)
    try:
        top = griffe.load(package, search_paths=[root.parent], resolve_aliases=False)
    except (ImportError, griffe.GriffeError) as e:
        raise RunError(f"griffe cannot read {package} from {project.relative(root)}: {e}") from None
    names: set[str] = set()
    modules = [top]
    while modules:
        module = modules.pop()
        for name, member in module.members.items():
            if member.is_alias:
                if member.is_exported:
                    names.add(name)
            elif member.is_module:
                if member.is_exported:
                    names.add(name)
                if member.is_public:
                    modules.append(member)
            elif member.is_exported:
                names.add(name)
    return names


# The seam the tests replace: `(project, package) -> names`.
surface: Callable[[Project, str], set[str]] = public_surface


def importable(project: Project) -> frozenset[str]:
    """Every name a snippet could have imported, checked against the sentinels.

    Raises:
        RunError: the vocabulary cannot be built, lost a sentinel, or holds a local name; a scan
            over a broken vocabulary would pass a page full of defects.
    """

    def build() -> frozenset[str]:
        settings = project.settings.get("docs", {})
        packages = settings.get("vocabulary", list(project.package_roots))
        names = set(settings.get("modules", [])) | CONTEXT_HELPERS
        for package in packages:
            names |= surface(project, package)
        if missing := sorted({"closing", *settings.get("sentinels", [])} - names):
            raise RunError(
                f"the docs vocabulary lost names it must contain: {missing}; correct "
                "[tool.conventions.docs] sentinels or vocabulary"
            )
        if leaked := sorted(NEVER_PUBLIC & names):
            raise RunError(f"the docs vocabulary holds local names: {leaked}")
        return frozenset(names)

    return project.memo("docs.importable", build)


class _Bindings(ast.NodeVisitor):
    """What a block binds and what it reads, as one flat scope: the lenient reading."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.used: dict[str, int] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.bound.add(node.name)
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            self.bound.add(arg.arg)
        for maybe in (args.vararg, args.kwarg):
            if maybe is not None:
                self.bound.add(maybe.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used.setdefault(node.id, node.lineno)
        else:
            self.bound.add(node.id)
        self.generic_visit(node)


@rule("CON950")
def snippets_import(project: Project) -> Iterator[Finding]:
    """A python block that imports anything imports every importable name it uses.

    Reason: most blocks are fragments whose context is the page, but a block carrying its own import
    list claims to be complete, and one that reaches for a public name it never imported does not
    run when copied.

    Fix: add the import to the block.

    Scope: importable means a module-level public name of `[tool.conventions.docs] vocabulary` (the
    project's packages when unset), as `public_surface` reads it from source, the modules in
    `modules`, and the context-manager helpers. A block that does not parse is CON913's.
    """
    if not _site(project):
        return
    names = importable(project)
    for page in _pages(project):
        for first, source in python_blocks(_lines(project, page)):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            if not any(isinstance(n, ast.Import | ast.ImportFrom) for n in ast.walk(tree)):
                continue
            seen = _Bindings()
            seen.visit(tree)
            for name, line in sorted(seen.used.items()):
                if name not in seen.bound and name in names:
                    yield Finding(
                        project.relative(page),
                        first + line - 1,
                        "CON950",
                        f"{name} is never imported",
                    )


# --- Fixers ---


@fixer("CON902")
def fix_highlights(project: Project) -> Fixes:
    """Rewrite each drifted `hl_lines` to the ranges its sub-sections' markers give, or add it.

    Only the `hl_lines` attribute on the fence line changes. A fence without a steering comment, or
    one whose markers are missing, is left as it is: CON900 and CON901 report it.
    """
    rewrites: dict[Path, str] = {}
    if not _site(project):
        return Fixes(rewrites)
    for page in _pages(project):
        lines = list(_lines(project, page))
        changed = False
        for fence in hl_fences(lines):
            source = project.root / fence.include
            if not fence.subs or not source.is_file():
                continue
            try:
                want = expected_hls(
                    (project.text(source) or "").splitlines(), fence.section, fence.subs
                )
            except LookupError:
                continue
            index = fence.line - 1
            attr = f'hl_lines="{fmt_hl(want)}"'
            if fence.hl_raw is None:
                opener = FENCE_LANG.match(lines[index])
                assert opener is not None  # hl_fences found a fence here
                lines[index] = f"{opener.group(0)} {attr}{lines[index][opener.end() :]}"
                changed = True
            elif parse_hl(fence.hl_raw) != want:
                lines[index] = HL_ATTR.sub(attr, lines[index])
                changed = True
        if changed:
            rewrites[page] = "\n".join(lines) + "\n"
    return Fixes(rewrites)
