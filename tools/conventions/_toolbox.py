"""The `toolbox` command: a tool run with what it needs to know about the workspace.

`just` holds what each verb does; this holds how a tool has to be run in this repository: which
files a path names, which distribution a file belongs to, the search path ty needs, and the
interpreter and floor each type checker spells its own way.

    toolbox <tool> [TOOL OPTIONS...] [-- PATHS...]

Options before `--` go to the tool as given; paths after it are the targets. Exit status: the
tool's, the first failing one's when it runs once per distribution, 2 when a path is refused.
"""

import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn, final

from ._project import Project, RunError

REFUSED = 2

PYTHON = ".venv"
INTERPRETER = ".venv/bin/python"

# What ty reads as Python. It has no file filter of its own and reports diagnostics for any file
# handed to it.
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})

# Where a distribution's typesafety suite lives: `assert_type` cases that exist to be judged by
# basedpyright in `just types-api`, and that ty disagrees with on some lines by design.
TYPESAFETY = "tests/typesafety"

# The toolbox's own files run on the toolbox's interpreter, which `just types-checks` checks them
# at. From the repository root ty reads the project's `requires-python` and `.venv` instead, and
# reports `ast.TypeAlias` missing in a file that only ever runs on the toolbox's interpreter.
TOOLBOX = "tools"

# The noxfile imports nox, which the tool environment installs and `.venv` does not: ty checks it
# against `.venv-tools`, in a run of its own, and leaves it out of the project's runs.
NOXFILE = "noxfile.py"
TOOLS_PYTHON = ".venv-tools"


@final
class Refused(Exception):
    """A path that would make a tool judge nothing, or something other than what was named."""


def _refuse(message: str) -> NoReturn:
    print(f"toolbox: {message}", file=sys.stderr)
    sys.exit(REFUSED)


def toplevel() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout
    except OSError, subprocess.CalledProcessError:
        _refuse("not inside a git repository")
    return Path(out.strip())


def distributions(root: Path) -> list[str]:
    """Each distribution's directory under `root`, `.` for the root one, in configured order."""
    try:
        packages = Project(root).settings["packages"]
    except RunError as e:
        _refuse(str(e))
    return list(dict.fromkeys(package.rpartition("/")[0] or "." for package in packages))


def _visible(root: Path, under: Sequence[str] = ()) -> list[str]:
    """The files git shows under `under`, everything when empty: tracked, or new and not ignored."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *under],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise Refused(
            f"git cannot list {' '.join(under) or 'the tree'}: {e.stderr.strip()}"
        ) from None
    return [name for name in dict.fromkeys(listing.split("\0")) if name and (root / name).is_file()]


def _existing(root: Path, paths: Sequence[str]) -> None:
    """Refuse a path that does not exist or lies outside the repository.

    Raises:
        Refused: one such path.
    """
    if missing := [path for path in paths if not (root / path).exists()]:
        raise Refused(
            "no such path: "
            + ", ".join(missing)
            + " (a path containing a space arrives split in two)"
        )
    outside = [p for p in paths if not (root / p).resolve().is_relative_to(root.resolve())]
    if outside:
        raise Refused("outside the repository: " + ", ".join(outside))


def targets(root: Path, paths: Sequence[str]) -> list[str]:
    """The files `paths` name, relative to `root`; the whole working tree when none is given.

    A directory names the files git shows under it. A file names itself, ignored or not. A path that
    does not exist is refused, and so is a directory holding nothing git shows: either would leave a
    tool judging nothing and reporting clean. `just` joins its variadic arguments with spaces, so a
    file name containing one arrives as two paths, which do not exist.

    Raises:
        Refused: a path names nothing, or lies outside the repository.
    """
    if not paths:
        return _visible(root)
    _existing(root, paths)
    found: list[str] = []
    for path in paths:
        if (root / path).is_dir():
            under = _visible(root, [path])
            if not under:
                raise Refused(f"no file git shows under {path}")
            found += under
        else:
            found.append(Path(path).as_posix())
    return list(dict.fromkeys(found))


def member_of(path: str, dists: Sequence[str]) -> str:
    """The distribution `path` belongs to: the deepest of `dists` holding it, else `.`."""
    parts = Path(path).parts
    holding = [
        dist for dist in dists if dist != "." and parts[: len(Path(dist).parts)] == Path(dist).parts
    ]
    return max(holding, key=lambda dist: len(Path(dist).parts), default=".")


def suites(root: Path, named: Sequence[str], dists: Sequence[str]) -> list[str]:
    """The typesafety suites a ty run over `named` would sweep up, to leave out of it.

    A suite under a directory named is swept up, and every suite by a run over the whole project.
    They exist for basedpyright, and ty disagrees with some of their lines by design. ty checks a
    path named on its command line even when it is excluded, so a suite named is still checked.
    """
    found = []
    for dist in dists:
        suite = TYPESAFETY if dist == "." else f"{dist}/{TYPESAFETY}"
        holding = [p for p in named if Path(suite).parts[: len(Path(p).parts)] == Path(p).parts]
        if (root / suite).is_dir() and (holding or not named):
            found.append(suite)
    return found


def search_paths(member: str, dists: Sequence[str]) -> list[str]:
    """ty's extra search paths for a member's files: its own directory, the other members, the root.

    The root is there for what a member's tests import from it (`examples`), and it goes last. Ahead
    of a member, ty reads `<pkg>/<pkg>/m.py` as `<pkg>.<pkg>.m`, its relative imports fail, and
    every name the package re-exports is `Unknown` with no diagnostic (astral-sh/ty#1730).
    """
    others = [dist for dist in dists if dist not in (member, ".")]
    return [member, *others, "."]


def floor(root: Path, dists: Sequence[str]) -> str:
    """The `X.Y` the first distribution's `requires-python` starts from: the checkers' floor."""
    pyproject = root / dists[0] / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            requires = tomllib.load(f).get("project", {}).get("requires-python", "")
    except (OSError, tomllib.TOMLDecodeError) as e:
        _refuse(f"cannot read {pyproject}: {e}")
    if not (match := re.search(r"(\d+\.\d+)", requires)):
        _refuse(f'{pyproject} states no requires-python = ">=3.N" to check at')
    return match[1]


def split(args: Sequence[str]) -> tuple[list[str], list[str] | None]:
    """The tool's own options, and the paths after `--`; `None` when there is no `--`."""
    if "--" not in args:
        return list(args), None
    at = list(args).index("--")
    return list(args[:at]), list(args[at + 1 :])


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command).returncode


def _each(
    groups: Mapping[str, Sequence[str]], command: Callable[[str, Sequence[str]], list[str]]
) -> int:
    """`command` once per distribution, over its paths; the first failing exit status, else 0."""
    status = 0
    for member, paths in groups.items():
        code = _run(command(member, paths))
        status = status or code
    return status


def _grouped(root: Path, paths: Sequence[str], dists: Sequence[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(member_of(path, dists), []).append(path)
    return groups


def prek(root: Path, options: Sequence[str], paths: Sequence[str]) -> int:
    """`prek run` over the targets, every file by name.

    `--all-files` means tracked files only, and a new file skipped in silence is what this exists to
    stop.
    """
    return _run(["prek", "run", *options, "--files", *targets(root, paths)])


def ty(root: Path, options: Sequence[str], paths: Sequence[str] | None) -> int:
    """`ty check`, once per distribution its paths belong to, each with that distribution's view.

    A workspace member is checked as its own project: every member's suite is a top-level `tests`
    package, and from the root they would all resolve to the first. `--exclude PATH`, a path from
    the repository root as ty reads it, goes to the run of the distribution holding it; so does each
    typesafety suite a directory would sweep up without naming it (`suites`).
    """
    dists = distributions(root)
    excludes: list[str] = []
    rest: list[str] = []
    it = iter(options)
    for option in it:
        if option == "--exclude":
            if not (value := next(it, "")):
                raise Refused("--exclude needs a path")
            excludes.append(value)
        elif option.startswith("--exclude="):
            excludes.append(option.removeprefix("--exclude="))
        else:
            rest.append(option)
    if paths is None:
        paths = []
    named = []
    noxfile = paths == [] and (root / NOXFILE).is_file()
    skipped_toolbox = False
    _existing(root, paths)
    for path in paths:
        if Path(path).parts[:1] == (TOOLBOX,):
            skipped_toolbox = True
        elif Path(path) == Path(NOXFILE):
            noxfile = True
        elif Path(path) == Path(".") and (root / NOXFILE).is_file():
            noxfile = True
            named.append(path)
        elif (root / path).is_dir() or Path(path).suffix in PYTHON_SUFFIXES:
            named.append(path)
    if skipped_toolbox:
        print("ty: tools/ is checked by 'just types-checks', on the Python in .venv-tools")
    if paths and not named and not noxfile:
        print("ty: skipped, no Python files among the given paths")
        return 0
    groups = _grouped(root, named, dists) if named else {".": []}
    excluded = _grouped(root, excludes, dists)

    def command(member: str, members_paths: Sequence[str]) -> list[str]:
        view = [] if member == "." else ["--project", member]
        for path in search_paths(member, dists) if member != "." else []:
            view += ["--extra-search-path", path]
        dropped = [*excluded.get(member, []), *suites(root, members_paths, dists)]
        if member == ".":
            dropped.append(NOXFILE)
        drop = [arg for path in dropped for arg in ("--exclude", path)]
        return ["ty", "check", "--python", PYTHON, *view, *drop, *rest, *members_paths]

    status = _each(groups, command) if named or not paths else 0
    if noxfile:
        code = _run(["ty", "check", "--python", TOOLS_PYTHON, *rest, NOXFILE])
        status = status or code
    return status


type Tool = Callable[[Path, Sequence[str], Sequence[str] | None], int]


def _checker(flags: Callable[[str], list[str]], subcommands: frozenset[str] = frozenset()) -> Tool:
    """A type checker run at the floor against `.venv`, once per distribution when given paths."""

    def run(root: Path, options: Sequence[str], paths: Sequence[str] | None) -> int:
        name, *args = options
        lead: list[str] = []
        while args and args[0] in subcommands:
            lead.append(args.pop(0))
        base = [name, *lead, *flags(floor(root, distributions(root))), *args]
        if paths is None:
            return _run(base)
        _existing(root, paths)
        return _each(_grouped(root, paths, distributions(root)), lambda _, p: [*base, *p])

    return run


CHECKERS: Mapping[str, Tool] = {
    "mypy": _checker(lambda v: ["--python-executable", INTERPRETER, "--python-version", v]),
    "basedpyright": _checker(lambda v: ["--pythonpath", INTERPRETER, "--pythonversion", v]),
    "pyrefly": _checker(
        lambda v: ["--python-interpreter-path", INTERPRETER, "--python-version", v],
        frozenset({"check", "coverage"}),
    ),
}

USAGE = """usage: toolbox <tool> [TOOL OPTIONS...] [-- PATHS...]

  prek          prek run over the files PATHS name (the working tree when none)
  ty            ty check, per distribution, with its search path
  mypy          mypy at the floor against .venv
  basedpyright  basedpyright at the floor against .venv
  pyrefly       pyrefly at the floor against .venv"""


def main(argv: Sequence[str] | None = None) -> NoReturn:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0 if args else REFUSED)
    tool, rest = args[0], args[1:]
    options, paths = split(rest)
    root = toplevel()
    try:
        if tool == "prek":
            sys.exit(prek(root, options, paths or []))
        if tool == "ty":
            sys.exit(ty(root, options, paths))
        if tool in CHECKERS:
            sys.exit(CHECKERS[tool](root, [tool, *options], paths))
    except Refused as e:
        _refuse(str(e))
    _refuse(f"unknown tool {tool}\n{USAGE}")
