"""The `toolbox` command: a tool run with what it needs to know about the workspace.

`just` holds what each verb does; this holds how a tool has to be run in this repository: which
files a path names, which distribution a file belongs to, the search path ty needs, and the
interpreter and Python version each type checker spells its own way.

A type checker runs at the floor, the version `requires-python` starts from. A file that ruff's
`per-file-target-version` (`[tool.ruff]` in the root `pyproject.toml`) gives a version of its own is
checked at that version instead, in a run of its own, and left out of the floor's run.

    toolbox <tool> [TOOL OPTIONS...] [-- PATHS...]

Options before `--` go to the tool as given; paths after it are the targets. Exit status: the
tool's, the first failing one's when it runs once per distribution, 2 when a path is refused.
"""

import configparser
import functools
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn, final

from ._project import Project, RunError

REFUSED = 2

PYTHON = ".venv"
INTERPRETER = ".venv/bin/python"

# What mypy, basedpyright and pyrefly read as Python.
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})

# What ty reads as Python, notebooks included. It has no file filter of its own and reports
# diagnostics for any file handed to it.
TY_SUFFIXES = frozenset({".py", ".pyi", ".ipynb"})

# The toolbox's own files run on the toolbox's interpreter, which `just tools::types` checks them
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


def target_versions(root: Path) -> dict[str, str]:
    """ruff's `per-file-target-version` from the root `pyproject.toml`: each glob and its `X.Y`."""
    pyproject = root / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            ruff = tomllib.load(f).get("tool", {}).get("ruff", {})
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as e:
        _refuse(f"cannot read {pyproject}: {e}")
    versions = {}
    for glob, target in ruff.get("per-file-target-version", {}).items():
        if not (match := re.fullmatch(r"py3(\d+)", str(target))):
            _refuse(f"per-file-target-version: {target!r} for {glob!r} is not a pyXY version")
        versions[glob] = f"3.{match[1]}"
    return versions


@functools.cache
def _glob(glob: str) -> re.Pattern[str]:
    """`glob` as ruff matches it: `*` crosses `/`, `**/` is also no directory, `{a,b}` is either."""
    out, i = [], 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("*", i):
            out.append(".*")
            i += 2 if glob.startswith("**", i) else 1
        elif glob[i] == "?":
            out.append(".")
            i += 1
        elif glob[i] == "[" and (end := glob.find("]", i + 2)) != -1:
            body = glob[i + 1 : end]
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = end + 1
        elif glob[i] == "{" and (end := glob.find("}", i)) != -1:
            either = "|".join(_glob(a).pattern for a in glob[i + 1 : end].split(","))
            out.append(f"(?:{either})")
            i = end + 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("".join(out))


def version_of(path: str, versions: Mapping[str, str]) -> str | None:
    """The version the globs give `path`, `None` when no glob matches it.

    A glob is matched as ruff matches it, against the path from the repository root, and a glob
    without a `/` against the file name too. Two globs that match one file with different versions
    are refused.

    Raises:
        Refused: two matching globs disagree.
    """
    name = PurePosixPath(path).name
    found = {
        version
        for glob, version in versions.items()
        if _glob(glob).fullmatch(path) or ("/" not in glob and _glob(glob).fullmatch(name))
    }
    if len(found) > 1:
        raise Refused(
            f"per-file-target-version gives {path} more than one version: {sorted(found)}"
        )
    return found.pop() if found else None


def pinned(
    root: Path,
    under: Sequence[str],
    versions: Mapping[str, str],
    suffixes: frozenset[str] = PYTHON_SUFFIXES,
) -> dict[str, str]:
    """Each file of `suffixes` under `under` (the whole tree when empty) the globs pin, and its
    version.

    `tools/` and the noxfile run on the toolbox's interpreter and are never pinned.
    """
    if not versions:
        return {}
    pins = {}
    for path in _visible(root, under):
        if Path(path).suffix not in suffixes or Path(path).parts[:1] == (TOOLBOX,):
            continue
        if path != NOXFILE and (version := version_of(path, versions)):
            pins[path] = version
    return pins


def _all_pinned(root: Path, path: str, pins: Mapping[str, str]) -> bool:
    """Whether `path` is a directory every file of which ty reads is pinned."""
    if not pins or not (root / path).is_dir():
        return False
    files = [p for p in _visible(root, [path]) if Path(p).suffix in TY_SUFFIXES]
    return bool(files) and all(p in pins for p in files)


def _by_version(pins: Mapping[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path, version in pins.items():
        groups.setdefault(version, []).append(path)
    return groups


def _relative(root: Path, path: str) -> str:
    """`path` as git lists it: from the repository root, with no `./` and no `..`."""
    given = Path(path)
    full = Path(os.path.normpath(given if given.is_absolute() else root / given))
    for base in (root, root.resolve()):
        if full.is_relative_to(base):
            return full.relative_to(base).as_posix()
    return path


def _under(path: str, scope: Iterable[str]) -> bool:
    parts = Path(path).parts
    return any(parts[: len(Path(s).parts)] == Path(s).parts for s in scope)


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
    the repository root as ty reads it, goes to the run of the distribution holding it.
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
    paths = [_relative(root, path) for path in paths]
    for path in paths:
        if Path(path).parts[:1] == (TOOLBOX,):
            skipped_toolbox = True
        elif Path(path) == Path(NOXFILE):
            noxfile = True
        elif Path(path) == Path(".") and (root / NOXFILE).is_file():
            noxfile = True
            named.append(path)
        elif (root / path).is_dir() or Path(path).suffix in TY_SUFFIXES:
            named.append(path)
    if skipped_toolbox:
        print("ty: tools/ is checked by 'just tools::types', on the Python in .venv-tools")
    if paths and not named and not noxfile:
        print("ty: skipped, no Python files among the given paths")
        return 0
    excluded = _grouped(root, excludes, dists)
    versions = target_versions(root)
    found = pinned(root, named, versions, TY_SUFFIXES) if named or not paths else {}
    pins = {path: v for path, v in found.items() if not _under(path, excludes)}
    floor_paths = [p for p in named if p not in pins and not _all_pinned(root, p, pins)]
    groups = _grouped(root, floor_paths, dists) if named else {".": []}

    def command(member: str, members_paths: Sequence[str]) -> list[str]:
        view = [] if member == "." else ["--project", member]
        for path in search_paths(member, dists) if member != "." else []:
            view += ["--extra-search-path", path]
        dropped = list(excluded.get(member, []))
        if member == ".":
            dropped.append(NOXFILE)
        dropped += [path for path in pins if member_of(path, dists) == member]
        drop = [arg for path in dropped for arg in ("--exclude", path)]
        return ["ty", "check", "--python", PYTHON, *view, *drop, *rest, *members_paths]

    status = _each(groups, command) if groups and (named or not paths) else 0
    for version, files in _by_version(pins).items():
        for member, members_files in _grouped(root, files, dists).items():
            code = _run([*command(member, members_files), "--python-version", version])
            status = status or code
    if noxfile:
        code = _run(["ty", "check", "--python", TOOLS_PYTHON, *rest, NOXFILE])
        status = status or code
    return status


type Tool = Callable[[Path, Sequence[str], Sequence[str] | None], int]


def _member_config(root: Path, member: str) -> list[str]:
    """`--config-file` for a member whose mypy configuration is its own `pyproject.toml`.

    mypy reads one configuration, found from where it runs, and the toolbox runs it from the root:
    without a `mypy.ini` there, a member's `[tool.mypy]` would go unread.
    """
    if member == "." or (root / "mypy.ini").is_file():
        return []
    pyproject = root / member / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            configured = "mypy" in tomllib.load(f).get("tool", {})
    except FileNotFoundError:
        return []
    return ["--config-file", f"{member}/pyproject.toml"] if configured else []


def _python_paths(root: Path, name: str, paths: Sequence[str]) -> list[str]:
    """The paths a checker of owned code reads: directories and Python files, outside `tools/`.

    `tools/` runs on the toolbox's interpreter (`just tools::types`), and the noxfile imports nox,
    which `.venv` does not install.
    """
    kept = []
    for path in paths:
        if Path(path).parts[:1] == (TOOLBOX,) or Path(path) == Path(NOXFILE):
            print(f"{name}: {path} is checked on the Python in .venv-tools, not here")
        elif (root / path).is_dir() or Path(path).suffix in PYTHON_SUFFIXES:
            kept.append(path)
    if not kept:
        print(f"{name}: skipped, no Python files among the given paths")
    return kept


def _mypy_files(root: Path) -> list[str]:
    """`[mypy] files` in `mypy.ini`: what a mypy run without paths checks."""
    config = configparser.ConfigParser()
    config.read(root / "mypy.ini", encoding="utf-8")
    return [f.strip() for f in config.get("mypy", "files", fallback="").split(",") if f.strip()]


def _pyrefly_includes(root: Path) -> list[str]:
    """`project-includes` in `pyrefly.toml`: what a pyrefly run without paths checks."""
    try:
        with (root / "pyrefly.toml").open("rb") as f:
            return list(tomllib.load(f).get("project-includes", []))
    except FileNotFoundError:
        return []
    except tomllib.TOMLDecodeError as e:
        _refuse(f"cannot read pyrefly.toml: {e}")


def pyrefly_config(root: Path) -> list[str]:
    """`--config pyrefly.toml` when the project has one.

    Named, not found: pyrefly 1.4 imports `mypy.ini` as a migration when the files it reaches
    include a dependency that ships a `pyproject.toml` inside its package (pandas does). It prints a
    warning per mypy error code it cannot map, and misses errors it reports with the config named.
    """
    return ["--config", "pyrefly.toml"] if (root / "pyrefly.toml").is_file() else []


def _checker(
    flags: Callable[[Path, str], list[str]],
    scope: Callable[[Path], list[str]] | None,
    subcommands: frozenset[str] = frozenset(),
    *,
    gate: bool = False,
) -> Tool:
    """A type checker run against `.venv`, once per distribution and version when given paths.

    A file the globs pin is checked at its version. A bare run, without paths and without targets
    among the tool's own arguments, covers what the checker's configuration names (`scope`); when a
    file there is pinned, it runs as if those paths were named, and a checker whose configuration
    the toolbox cannot read refuses. A run whose targets are the tool's own arguments runs once, at
    the floor.

    The checker that gates owned code (`gate`) is handed any paths, as `just check` names them: it
    skips what is not Python, `tools/` and the noxfile. It checks a `typing_tests/` suite as it does
    the package, whose level the suite is at.
    """

    def run(root: Path, options: Sequence[str], paths: Sequence[str] | None) -> int:
        name, *args = options
        lead: list[str] = []
        while args and args[0] in subcommands:
            lead.append(args.pop(0))
        dists = distributions(root)
        at = floor(root, dists)
        versions = target_versions(root)

        def base(version: str) -> list[str]:
            return [name, *lead, *flags(root, version), *args]

        if paths is None and any(not arg.startswith("-") for arg in args):
            return _run(base(at))
        if paths is None:
            if scope is None:
                if versions:
                    raise Refused(
                        f"{name} cannot tell what per-file-target-version pins; name the paths"
                    )
                return _run(base(at))
            covered = scope(root)
            if not (pinned(root, covered, versions) if covered else {}):
                return _run(base(at))
            paths = covered
        _existing(root, paths)
        paths = [_relative(root, path) for path in paths]
        if gate:
            paths = _python_paths(root, name, paths)
            if not paths:
                return 0
        dirs = [p for p in paths if (root / p).is_dir()]
        pins = pinned(root, dirs, versions) if dirs else {}
        for path in paths:
            if path not in dirs and (version := version_of(Path(path).as_posix(), versions)):
                pins[path] = version
        files: list[str] = []
        for path in paths:
            if path in pins:
                continue
            if (root / path).is_dir() and any(_under(p, [path]) for p in pins):
                files += [
                    p
                    for p in _visible(root, [path])
                    if Path(p).suffix in PYTHON_SUFFIXES
                    and p not in pins
                    and Path(p).parts[:1] != (TOOLBOX,)
                    and p != NOXFILE
                ]
            else:
                files.append(path)

        def config(member: str) -> list[str]:
            return _member_config(root, member) if gate else []

        status = (
            _each(_grouped(root, files, dists), lambda m, p: [*base(at), *config(m), *p])
            if files
            else 0
        )
        for version, pinned_files in _by_version(pins).items():
            for member, members_files in _grouped(root, pinned_files, dists).items():
                code = _run([*base(version), *config(member), *members_files])
                status = status or code
        return status

    return run


CHECKERS: Mapping[str, Tool] = {
    "mypy": _checker(
        lambda _, v: ["--python-executable", INTERPRETER, "--python-version", v],
        _mypy_files,
        gate=True,
    ),
    "basedpyright": _checker(
        lambda _, v: ["--pythonpath", INTERPRETER, "--pythonversion", v], None
    ),
    "pyrefly": _checker(
        lambda root, v: [
            *pyrefly_config(root),
            "--python-interpreter-path",
            INTERPRETER,
            "--python-version",
            v,
        ],
        _pyrefly_includes,
        frozenset({"check", "coverage"}),
    ),
}

USAGE = """usage: toolbox <tool> [TOOL OPTIONS...] [-- PATHS...]

  prek          prek run over the files PATHS name (the working tree when none)
  ty            ty check, per distribution, with its search path
  mypy          mypy at the floor against .venv
  basedpyright  basedpyright at the floor against .venv
  pyrefly       pyrefly at the floor against .venv

A file ruff's per-file-target-version gives a version is checked at that version, in a run of
its own."""


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
