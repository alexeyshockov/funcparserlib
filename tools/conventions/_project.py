"""The repository under check: its files, its configuration and its packages, each read once."""

import ast
import configparser
import subprocess
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, cast, final

# Files under this directory are the runner's own and never scanned: its sources spell the
# violations its rules look for.
SELF = Path(__file__).resolve().parent

# `[tool.conventions]` keys, with the nested tables' own keys. Keys read only by rules not yet
# ported are listed too, so a project's data is never reported unknown while it is in use.
KNOWN_KEYS: Mapping[str, frozenset[str] | None] = {
    "packages": None,
    "initialisms": None,
    "header_names": None,
    "consumers": None,
    "timeout_exceptions": None,
    "subclassable": None,
    "informational": None,
    "dependencies": None,
    "messages": frozenset({"known_names", "ambiguous", "singular", "builders", "builders_home"}),
    "docs": frozenset({"modules", "sentinels", "vocabulary"}),
}


@final
class RunError(Exception):
    """The run cannot produce a verdict: the repository or its configuration is not readable."""


@dataclass(frozen=True, slots=True)
class Module:
    name: str
    path: Path
    public: bool


@final
class Project:
    """One repository, read lazily and cached for every rule in the run."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._texts: dict[Path, str | None] = {}
        self._trees: dict[Path, ast.Module | None] = {}
        self._memo: dict[str, object] = {}
        self.unparsable: dict[Path, SyntaxError] = {}

    def memo[T](self, key: str, compute: Callable[[], T]) -> T:
        """`compute()` once per run: an index more than one rule reads."""
        if key not in self._memo:
            self._memo[key] = compute()
        return cast(T, self._memo[key])

    @cached_property
    def pyproject(self) -> Mapping[str, Any]:
        path = self.root / "pyproject.toml"
        try:
            with path.open("rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            raise RunError(f"no pyproject.toml at {self.root}") from None
        except tomllib.TOMLDecodeError as e:
            raise RunError(f"pyproject.toml does not parse: {e}") from None

    @cached_property
    def settings(self) -> Mapping[str, Any]:
        """`[tool.conventions]`, validated, with `packages` defaulted from `[project] name`."""
        table = dict(self.pyproject.get("tool", {}).get("conventions", {}))
        for key, value in table.items():
            if key not in KNOWN_KEYS:
                raise RunError(f"unknown key `{key}` in [tool.conventions]")
            nested = KNOWN_KEYS[key]
            if nested is not None:
                if not isinstance(value, dict):
                    raise RunError(f"[tool.conventions.{key}] is not a table")
                if unknown := sorted(set(value) - nested):
                    raise RunError(f"unknown keys in [tool.conventions.{key}]: {unknown}")
        subclassable = table.get("subclassable", {})
        if not isinstance(subclassable, dict) or not all(
            isinstance(reason, str) for reason in subclassable.values()
        ):
            raise RunError(
                "[tool.conventions.subclassable] maps each qualified class name to its reason"
            )
        if "packages" not in table:
            table["packages"] = self._default_packages()
        return table

    def _default_packages(self) -> list[str]:
        """The import package of the root distribution, else of each workspace member."""

        def importable(name: str) -> str:
            return name.replace("-", "_").replace(".", "_")

        name = self.pyproject.get("project", {}).get("name")
        if isinstance(name, str):
            return [importable(name)]
        patterns = self.pyproject.get("tool", {}).get("uv", {}).get("workspace", {})
        found = []
        for pattern in patterns.get("members", []):
            for directory in sorted(self.root.glob(pattern)):
                try:
                    with (directory / "pyproject.toml").open("rb") as f:
                        member = tomllib.load(f).get("project", {}).get("name")
                except FileNotFoundError, tomllib.TOMLDecodeError:
                    continue
                if isinstance(member, str):
                    relative = directory.relative_to(self.root).as_posix()
                    found.append(f"{relative}/{importable(member)}")
        if not found:
            raise RunError(
                "[tool.conventions] packages is unset, and neither a [project] name nor the "
                "workspace members default it"
            )
        return found

    @cached_property
    def example_dirs(self) -> Sequence[str]:
        """The examples directories that hold Python, each relative to the root.

        The root's `examples`, then each distribution's `<dir>/examples`, in `packages` order.
        """
        dirs = (path.rpartition("/")[0] for path in self.settings["packages"])
        candidates = ["examples", *(f"{d}/examples" for d in dirs if d)]
        return tuple(
            path
            for path in dict.fromkeys(candidates)
            if (self.root / path).is_dir() and next((self.root / path).rglob("*.py"), None)
        )

    @cached_property
    def mypy_files(self) -> Sequence[str] | None:
        """`[mypy] files` in `mypy.ini`, what a mypy run without paths checks; `None` without the
        file.
        """
        path = self.root / "mypy.ini"
        if not path.is_file():
            return None
        config = configparser.ConfigParser()
        try:
            config.read_string(path.read_text(encoding="utf-8"))
        except configparser.Error as e:
            raise RunError(f"cannot read mypy.ini: {e}") from None
        files = config.get("mypy", "files", fallback="")
        return tuple(f.strip() for f in files.split(",") if f.strip())

    @cached_property
    def pyrefly_includes(self) -> Sequence[str] | None:
        """`project-includes` in `pyrefly.toml`, what a pyrefly run without paths checks; `None`
        without the file.
        """
        path = self.root / "pyrefly.toml"
        if not path.is_file():
            return None
        try:
            includes = tomllib.loads(path.read_text(encoding="utf-8")).get("project-includes", [])
        except tomllib.TOMLDecodeError as e:
            raise RunError(f"cannot read pyrefly.toml: {e}") from None
        return tuple(str(include) for include in includes)

    @cached_property
    def package_roots(self) -> Mapping[str, Path]:
        """Each package's import name and its directory."""
        roots = {}
        for relative in self.settings["packages"]:
            path = self.root / relative
            if not any(path.rglob("*.py")):
                raise RunError(
                    f"no modules under {relative}; [tool.conventions] packages names each "
                    "package directory, relative to the repository root"
                )
            roots[path.name] = path
        return roots

    @cached_property
    def site_packages(self) -> Sequence[Path]:
        """Where `.venv` installs the project's dependencies; empty without a `.venv`."""
        venv = self.root / ".venv"
        sites = [*venv.glob("lib/python*/site-packages"), venv / "Lib" / "site-packages"]
        return tuple(site for site in sites if site.is_dir())

    @cached_property
    def files(self) -> Sequence[Path]:
        """The working tree as git lists it: tracked files present, untracked ones not ignored."""
        try:
            listing = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            raise RunError(f"cannot list the repository's files with git: {e}") from None
        found = []
        for name in dict.fromkeys(listing.split("\0")):
            path = self.root / name
            if name and path.is_file() and not path.is_symlink() and not path.is_relative_to(SELF):
                found.append(path)
        if not found:
            raise RunError(f"git lists no files under {self.root}")
        return tuple(found)

    @cached_property
    def modules(self) -> Sequence[Module]:
        """Every module in the packages, walked from the filesystem, never imported.

        A stub with no source beside it is a compiled module's, and stands in for it.
        """
        found = []
        for package, root in self.package_roots.items():
            stubs = (p for p in root.rglob("*.pyi") if not p.with_suffix(".py").exists())
            for path in sorted([*root.rglob("*.py"), *stubs]):
                if "__pycache__" in path.parts:
                    continue
                relative = path.relative_to(root)
                parts = list(relative.parts[:-1])
                if relative.name != "__init__.py":
                    parts.append(relative.stem)
                public = not any(part.startswith("_") for part in parts)
                found.append(Module(".".join([package, *parts]), path, public))
        return tuple(found)

    def module_path(self, name: str) -> Path | None:
        """A module's source, or its stub where it is compiled; `None` outside the packages."""
        package, *parts = name.split(".")
        if package not in self.package_roots:
            return None
        path = self.package_roots[package].joinpath(*parts)
        for candidate in (path / "__init__.py", path.with_suffix(".py"), path.with_suffix(".pyi")):
            if candidate.is_file():
                return candidate
        return None

    def text(self, path: Path) -> str | None:
        """The file's text, or `None` if it is not UTF-8."""
        if path not in self._texts:
            try:
                self._texts[path] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                self._texts[path] = None
        return self._texts[path]

    def tree(self, path: Path) -> ast.Module | None:
        """The parsed module, or `None` if it does not parse (recorded in `unparsable`)."""
        if path not in self._trees:
            text = self.text(path)
            try:
                self._trees[path] = ast.parse(text or "", filename=str(path))
            except SyntaxError as e:
                self._trees[path] = None
                self.unparsable[path] = e
        return self._trees[path]

    def under(self, roots: Sequence[Path], suffixes: frozenset[str]) -> Iterator[Path]:
        """The listed files with one of `suffixes` below any of `roots`."""
        for path in self.files:
            if path.suffix in suffixes and any(path.is_relative_to(root) for root in roots):
                yield path

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def line_of(self, path: Path, needle: str) -> int | None:
        """The first line holding `needle`, or `None`."""
        for number, line in enumerate((self.text(path) or "").splitlines(), 1):
            if needle in line:
                return number
        return None
