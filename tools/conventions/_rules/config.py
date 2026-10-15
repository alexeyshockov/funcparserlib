"""CON8xx: a tool is configured in one place, a second place cannot appear beside it, and pytest's
configuration fails a test on an exception nothing receives.
"""

import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .._engine import Finding, rule
from .._project import Project
from .bounds import test_roots

PYPROJECT = "pyproject.toml"
SHARED_RUFF = "shared.ruff.toml"

# The shapes a tool's own configuration file takes. `<name>` is filled from each `[tool.<name>]`
# section, so the list is about file conventions rather than any particular tool.
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

    Fix: restore the file from the main branch (`git checkout origin/main -- shared.ruff.toml`).
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


@rule("CON805")
def no_contracts_table(project: Project) -> Iterator[Finding]:
    """`pyproject.toml` carries no `[tool.contracts]` table.

    Reason: the shared rules read `[tool.conventions]`; a `[tool.contracts]` table left from before
    the rename configures nothing, and its carve-outs and initialisms silently stop applying.

    Fix: rename the table to `[tool.conventions]`.
    """
    text, data = _pyproject(project)
    if "contracts" in data.get("tool", {}):
        yield Finding(
            PYPROJECT,
            _header_line(text, "tool.contracts"),
            "CON805",
            "[tool.contracts] is read by nothing; the table is [tool.conventions]",
        )


@rule("CON806")
def no_contracts_directory(project: Project) -> Iterator[Finding]:
    """Nothing is left in `tools/contracts/`: the repository's own checks are in `tools/checks/`.

    Reason: `just tools::conventions` runs `tools/checks/` only, so a file left in
    `tools/contracts/` from before the rename is a rule nothing runs, and the recipe reports that
    the repository has none.

    Fix: move each `tools/contracts/<topic>_contract.py` to `tools/checks/<topic>_check.py`.
    """
    for path in project.files:
        relative = project.relative(path)
        if relative.startswith("tools/contracts/"):
            yield Finding(relative, None, "CON806", "move to tools/checks/<topic>_check.py")


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


def _tool_tables(project: Project, path: Path) -> dict[str, Any]:
    try:
        return dict(tomllib.loads(project.text(path) or "").get("tool", {}))
    except tomllib.TOMLDecodeError:
        return {}
