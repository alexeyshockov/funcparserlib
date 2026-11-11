"""CON8xx: a tool is configured in one place, a second place cannot appear beside it, pytest's
configuration fails a test on an exception nothing receives, the lists rendered from the packages
name every package, and the support a distribution claims is what the noxfile gates.
"""

import ast
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .._engine import Finding, rule
from .._project import Project
from .._versions import distributions
from .bounds import test_roots

PYPROJECT = "pyproject.toml"
SHARED_RUFF = "shared.ruff.toml"

# The shapes a tool's own configuration file takes. `<name>` is filled from each `[tool.<name>]`
# section.
FILENAMES = (
    "{name}.toml",
    ".{name}.toml",
    "_{name}.toml",
    "{name}.cfg",
    ".{name}.cfg",
    "{name}.ini",
    ".{name}.ini",
    ".{name}.json",
    ".{name}.yaml",
    ".{name}.yml",
    ".{name}rc",
)

# The keys that replace the shared lists rather than extend them. Each has an `extend-` spelling
# that adds, and that spelling is the only one the project may use.
REPLACING_LINT_KEYS = ("select", "ignore", "per-file-ignores")
# The width is generated into both `shared.ruff.toml` and `.editorconfig`; set here it would
# silently disagree with the latter.
REPLACING_TOP_KEYS = ("line-length",)

HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(?:#.*)?$")


def _table_lines(text: str) -> Iterator[tuple[int, str, str]]:
    """Each line with its number and the dotted name of the table it sits in."""
    table = ""
    for number, line in enumerate(text.splitlines(), 1):
        if match := HEADER.match(line):
            table = re.sub(r"\s*\.\s*", ".", match.group(1)).replace('"', "")
        yield number, table, line


def _header_line(text: str, table: str) -> int | None:
    """The line opening `[table]` or its first subtable."""
    for number, current, line in _table_lines(text):
        if HEADER.match(line) and (current == table or current.startswith(f"{table}.")):
            return number
    return None


def _key_line(text: str, table: str, key: str) -> int | None:
    """The line setting `key` in `[table]`, or setting it dotted from a parent table.

    Falls back to the table's header when the key is spelled some other way.
    """
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for number, current, line in _table_lines(text):
        if HEADER.match(line):
            if current == f"{table}.{key}":
                return number
            continue
        if current == table and key_pattern.match(line):
            return number
        if table.startswith(f"{current}.") or not current:
            rest = table.removeprefix(f"{current}.") if current else table
            if re.match(rf"^\s*{re.escape(rest)}\s*\.\s*{re.escape(key)}\s*=", line):
                return number
    return _header_line(text, table)


def _code_line(text: str, table: str, key: str, code: str) -> int | None:
    """The line naming `code` inside `key`'s value in `[table]`, else the key's line."""
    start = _key_line(text, table, key)
    if start is None:
        return None
    lines = text.splitlines()
    for number in range(start, len(lines) + 1):
        line = lines[number - 1]
        if f'"{code}"' in line or f"'{code}'" in line:
            return number
        if number > start and (HEADER.match(line) or re.match(r"^\s*[\w.-]+\s*=", line)):
            break
    return start


def _pyproject(project: Project) -> tuple[str, dict[str, Any]]:
    return project.text(project.root / PYPROJECT) or "", dict(project.pyproject)


@rule("CON800")
def shadows(project: Project) -> Iterator[Finding]:
    """A tool configured in `pyproject.toml` has no competing file beside it.

    Reason: a tool resolves its dedicated file ahead of `pyproject.toml` and does not merge the two,
    so a file such as `_typos.toml` beside `[tool.typos]` switches off every setting in the section,
    silently.

    Fix: move the file's settings into the `[tool.<name>]` section and delete the file, or move the
    section into the file.
    """
    text, data = _pyproject(project)
    for name in sorted(data.get("tool", {})):
        for shape in FILENAMES:
            candidate = project.root / shape.format(name=name)
            if candidate.exists():
                yield Finding(
                    PYPROJECT,
                    _header_line(text, f"tool.{name}"),
                    "CON800",
                    f"{candidate.name} shadows [tool.{name}]: the tool reads the file and "
                    "ignores the section",
                )


@rule("CON801", "CON802")
def ruff_reaches_the_shared_file(project: Project) -> Iterator[Finding]:
    """Ruff's rule set is the shared file, and the shared file has a rule set.

    CON801: `[tool.ruff]` extends `shared.ruff.toml`.

    Reason: without the `extend`, the project lints with its own rule set rather than the shared
    one.

    Fix: set `extend = "shared.ruff.toml"` in `[tool.ruff]`.

    CON802: `shared.ruff.toml` is present, native ruff configuration, and selects rules.

    Reason: an `extend` of a missing or empty file, or of one whose settings sit under a
    `[tool.ruff]` prefix ruff ignores there, lints with ruff's defaults.

    Fix: restore the file from the default branch
    (`git checkout origin/<branch> -- shared.ruff.toml`).
    """
    text, data = _pyproject(project)
    ruff = data.get("tool", {}).get("ruff", {})
    if ruff.get("extend") != SHARED_RUFF:
        yield Finding(
            PYPROJECT,
            _key_line(text, "tool.ruff", "extend") or 1,
            "CON801",
            f'[tool.ruff] does not `extend = "{SHARED_RUFF}"`',
        )
    shared_path = project.root / SHARED_RUFF
    if not shared_path.is_file():
        yield Finding(PYPROJECT, None, "CON802", f"{SHARED_RUFF} is missing")
        return
    shared_text = project.text(shared_path) or ""
    try:
        shared = tomllib.loads(shared_text)
    except tomllib.TOMLDecodeError as e:
        yield Finding(SHARED_RUFF, None, "CON802", f"{SHARED_RUFF} does not parse: {e}")
        return
    if "tool" in shared:
        yield Finding(
            SHARED_RUFF,
            _header_line(shared_text, "tool"),
            "CON802",
            f"{SHARED_RUFF} carries a [tool] table, which ruff ignores in an extended file",
        )
    if not shared.get("lint", {}).get("select"):
        yield Finding(SHARED_RUFF, None, "CON802", f"{SHARED_RUFF} selects nothing")


@rule("CON803", "CON804")
def ruff_only_adds(project: Project) -> Iterator[Finding]:
    """The project's own ruff configuration is additions, and additions only.

    CON803: The project uses the `extend-` spelling of every shared list.

    Reason: `select`, `ignore` and `per-file-ignores` beside `extend` replace the shared lists
    rather than merge with them, and `line-length` would disagree with `.editorconfig`, which is
    generated with the same width. Ruff says nothing either way.

    Fix: use `extend-select`, `extend-ignore` and `extend-per-file-ignores`; leave the width to
    `shared.ruff.toml`.

    CON804: No code is both extend-selected and extend-ignored.

    Reason: ignore wins over select within one file, so the code is off and the comment beside its
    select describes a rule that does not run.

    Fix: remove the code from one of the two lists.
    """
    text, data = _pyproject(project)
    ruff = data.get("tool", {}).get("ruff", {})
    lint: dict[str, Any] = ruff.get("lint", {})
    for key in REPLACING_TOP_KEYS:
        if key in ruff:
            yield Finding(
                PYPROJECT,
                _key_line(text, "tool.ruff", key),
                "CON803",
                f"[tool.ruff] {key} replaces the shared setting",
            )
    for key in REPLACING_LINT_KEYS:
        if key in lint:
            yield Finding(
                PYPROJECT,
                _key_line(text, "tool.ruff.lint", key),
                "CON803",
                f"[tool.ruff.lint] {key} replaces the shared list; use extend-{key}",
            )
    both = set(lint.get("extend-select", ())) & set(lint.get("extend-ignore", ()))
    for code in sorted(both):
        yield Finding(
            PYPROJECT,
            _code_line(text, "tool.ruff.lint", "extend-ignore", code),
            "CON804",
            f"{code} is both extend-selected and extend-ignored, so it is off",
        )


# The categories a `filterwarnings` entry can name and still cover pytest's unraisable warning.
UNRAISABLE_CATEGORIES = frozenset(
    {
        "",
        "Warning",
        "pytest.PytestWarning",
        "pytest.PytestUnraisableExceptionWarning",
        "_pytest.warning_types.PytestWarning",
        "_pytest.warning_types.PytestUnraisableExceptionWarning",
    }
)


def _unraisable_action(entries: list[str]) -> str:
    """The action pytest takes on an unraisable exception under `entries`: the last broad match.

    An entry with a message or module pattern exempts one known case and decides nothing here.
    """
    action = "default"
    for entry in entries:
        parts = [part.strip() for part in entry.split(":")]
        parts += [""] * (5 - len(parts))
        verb, message, category, module = parts[0], parts[1], parts[2], parts[3]
        if not message and not module and category in UNRAISABLE_CATEGORIES:
            action = verb
    return action


@rule("CON807")
def unraisable_fails(project: Project) -> Iterator[Finding]:
    """An exception nothing can receive fails the tests.

    Reason: Python sends an exception raised where no caller can take it (a `__del__`, a callback, a
    telemetry observer) to `sys.unraisablehook`, and pytest reports it as a
    `PytestUnraisableExceptionWarning`, which by default passes the test. A bug that only ever
    surfaces there is then never seen.

    Fix: add `"error"` (or `"error::pytest.PytestUnraisableExceptionWarning"`) to `filterwarnings`
    in `[tool.pytest.ini_options]`, after any broad `ignore`. A known case is exempted by an entry
    with a message pattern.

    Scope: the pytest configuration each `tests/` runs under: the `pyproject.toml` beside it when
    that file configures pytest, the repository's otherwise.
    """
    seen: set[Path] = set()
    for root in test_roots(project):
        config = next(
            (
                candidate
                for candidate in (root.parent / PYPROJECT, project.root / PYPROJECT)
                if candidate.is_file() and "pytest" in _tool_tables(project, candidate)
            ),
            None,
        )
        if config is None or config in seen:
            continue
        seen.add(config)
        text = project.text(config) or ""
        options = tomllib.loads(text)["tool"]["pytest"].get("ini_options", {})
        entries = options.get("filterwarnings", [])
        entries = [entries] if isinstance(entries, str) else list(entries)
        if _unraisable_action(entries) != "error":
            yield Finding(
                project.relative(config),
                _key_line(text, "tool.pytest.ini_options", "filterwarnings")
                or _header_line(text, "tool.pytest"),
                "CON807",
                'an unraisable exception passes the tests; add "error" to filterwarnings',
            )


def _read(project: Project, name: str) -> str | None:
    path = project.root / name
    return project.text(path) if path.is_file() else None


def _interrogate_args(text: str) -> tuple[int, list[str]] | None:
    """The line of the interrogate hook's `args:` and the paths in its list, when it has one."""
    lines = text.splitlines()
    hook = next((n for n, line in enumerate(lines, 1) if line.strip() == "- id: interrogate"), None)
    if hook is None:
        return None
    for number, line in enumerate(lines[hook:], hook + 1):
        if line.strip().startswith(("- id:", "- repo:")):
            return None
        if line.strip() == "args:":
            paths: list[str] = []
            indent = None
            for item in lines[number:]:
                stripped = item.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                depth = len(item) - len(item.lstrip())
                indent = depth if indent is None else indent
                if depth != indent or not stripped.startswith("- "):
                    break
                if not (value := stripped[2:].strip()).startswith("-"):
                    paths.append(value)
            return number, paths
    return None


@rule("CON808")
def rendered_package_lists(project: Project) -> Iterator[Finding]:
    """The interrogate hook's arguments name each package.

    Reason: the template renders the list once, from `[tool.conventions] packages`. `copier update`
    carries only what the template changed, so a package added later never reaches it, and
    interrogate would not measure that package's docstrings. `copier recopy --overwrite` re-renders
    it, and also deletes every line the project added to a file it rewrites. mypy and pyrefly are
    not held: the recipes pass them explicit paths (`conventions packages`, `--examples`).

    Fix: add the path the message names to the hook's `args`, or remove the one it names as no
    package. `conventions packages` prints what the list should hold.
    """
    packages = list(project.settings["packages"])
    text = _read(project, ".pre-commit-config.yaml")
    if text is not None and (found := _interrogate_args(text)) is not None:
        line, listed_args = found
        yield from _compare(
            ".pre-commit-config.yaml", line, "interrogate's args", packages, listed_args
        )


def _compare(
    path: str, line: int | None, where: str, expected: list[str], listed: list[str]
) -> Iterator[Finding]:
    for missing in (p for p in expected if p not in listed):
        yield Finding(path, line, "CON808", f"{where} lacks {missing}")
    for extra in (p for p in listed if p not in expected):
        yield Finding(path, line, "CON808", f"{where} names {extra}, which is no package here")


def _tool_tables(project: Project, path: Path) -> dict[str, Any]:
    try:
        return dict(tomllib.loads(project.text(path) or "").get("tool", {}))
    except tomllib.TOMLDecodeError:
        return {}


# --- Support claims ---

VERSION = re.compile(r"\d+\.\d+t?")
PYTHON_CLASSIFIER = re.compile(r"Programming Language :: Python :: (\d+\.\d+)")
LINUX = "Operating System :: POSIX :: Linux"
WINDOWS = "Operating System :: Microsoft :: Windows"


def _strings(node: ast.expr | None) -> list[str] | None:
    """A list or tuple literal of string constants, else `None`."""
    if not isinstance(node, ast.List | ast.Tuple):
        return None
    if not all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        return None
    return [str(e.value) for e in node.elts if isinstance(e, ast.Constant)]  # narrowed above


def gating_sessions(tree: ast.Module) -> list[tuple[str, frozenset[str]]]:
    """Each statically readable session the CI matrix gates on, as its Python and its tags.

    A session is a `nox.param("3.14", tags=[...])` or an `@nox.session(python="3.14", tags=[...])`
    with literal arguments; one tagged `ci` and not `optional` gates, and `3.14t` is read as the
    `3.14t` it is spelled. An interpreter or a tag list computed at import time is not read.
    """
    found: list[tuple[str, frozenset[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        tags = next((_strings(k.value) for k in node.keywords if k.arg == "tags"), None)
        if tags is None:
            continue
        python = None
        if name == "param" and node.args:
            python = node.args[0]
        elif name == "session":
            python = next((k.value for k in node.keywords if k.arg == "python"), None)
        if (
            isinstance(python, ast.Constant)
            and isinstance(python.value, str)
            and VERSION.fullmatch(python.value)
            and "ci" in tags
            and "optional" not in tags
        ):
            found.append((python.value, frozenset(tags)))
    return found


def _oldest(versions: set[str]) -> str:
    return min(versions, key=lambda v: tuple(map(int, v.split("."))))


@rule("CON809", "CON810", "CON811")
def support_claims(project: Project) -> Iterator[Finding]:
    """The Python versions and platforms a distribution claims are the ones the noxfile gates.

    CON809: A Python-version classifier is exactly a version a gating session runs.

    Reason: PyPI must not advertise a version nothing runs, nor omit one that does; the first is
    support claimed and never tested, the second work done and not credited, and how a version gets
    dropped by accident when the matrix is tidied.

    Fix: change the classifiers and the noxfile's sessions in the same commit. A session tagged
    `optional` runs under `continue-on-error` and claims nothing.

    CON810: `requires-python` is `>=` the oldest version the classifiers claim.

    Reason: the two are one boundary written twice.

    Fix: change `requires-python` to the version the message names, or the classifiers.

    CON811: The Linux and Windows classifiers are exactly the platforms a gating session runs on.

    Reason: for the reason of CON809, on the platform axis. A session tagged `windows` runs on
    Windows and any other on Linux; a platform whose only run may be red is not claimed.

    Fix: promote the Windows session (drop its `optional` tag) and add `Operating System ::
    Microsoft :: Windows` in one change.

    Scope: configuration, so CON8xx; the noxfile is the evidence and the classifiers the claim.
    Every distribution with `classifiers` is held against the root `noxfile.py`. The noxfile is read
    statically, so a matrix computed at import time (as the template's own derives it from the
    classifiers) yields no sessions and nothing is held. CON811 is skipped for a distribution with
    no operating-system classifier or with `Operating System :: OS Independent`, and ignores
    platforms other than Linux and Windows, whose gate a project may keep outside the noxfile.
    """
    noxfile = project.root / "noxfile.py"
    tree = project.tree(noxfile) if noxfile.is_file() else None
    sessions = gating_sessions(tree) if tree is not None else []
    if not sessions:
        return
    gated = {python.removesuffix("t") for python, _ in sessions}
    platforms = {WINDOWS if "windows" in tags else LINUX for _, tags in sessions}
    for member in distributions(project.root).members:
        classifiers = _classifiers(project, member.pyproject)
        if classifiers is None:
            continue
        path = project.relative(member.pyproject)
        claimed = {m[1] for c in classifiers if (m := PYTHON_CLASSIFIER.fullmatch(c))}
        line = project.line_of(member.pyproject, "Programming Language :: Python ::")
        for version in sorted(claimed - gated):
            yield Finding(
                path, line, "CON809", f"{version} is claimed and no gating session runs it"
            )
        for version in sorted(gated - claimed):
            yield Finding(path, line, "CON809", f"{version} is gated and no classifier claims it")
        requires = _project_table(project, member.pyproject).get("requires-python")
        if claimed and requires != f">={_oldest(claimed)}":
            yield Finding(
                path,
                project.line_of(member.pyproject, "requires-python"),
                "CON810",
                f"requires-python is {requires!r}; the oldest claimed is {_oldest(claimed)}",
            )
        systems = {c for c in classifiers if c.startswith("Operating System ::")}
        if not systems or "Operating System :: OS Independent" in systems:
            continue
        line = project.line_of(member.pyproject, "Operating System ::")
        for platform in sorted((systems & {LINUX, WINDOWS}) - platforms):
            yield Finding(
                path, line, "CON811", f"`{platform}` is claimed and no gating session runs on it"
            )
        for platform in sorted(platforms - systems):
            yield Finding(path, line, "CON811", f"`{platform}` is gated and not claimed")


def _project_table(project: Project, path: Path) -> dict[str, Any]:
    try:
        table = tomllib.loads(project.text(path) or "").get("project", {})
    except tomllib.TOMLDecodeError:
        return {}
    return dict(table)


def _classifiers(project: Project, path: Path) -> list[str] | None:
    listed = _project_table(project, path).get("classifiers")
    return [str(c) for c in listed] if isinstance(listed, list) else None
