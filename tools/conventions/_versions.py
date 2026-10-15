"""The repository's distributions, their one version, and the commands that write it.

A repository is a uv workspace whose members are `[tool.uv.workspace] members`, or a single
distribution at the root. Every member carries the same version; a member's requirement on another
is pinned with `==` at it; a Cargo workspace version, where `Cargo.toml` has one, is the same
version in Cargo's spelling (`X.Y.Z-dev.N` for `X.Y.Z.devN`), which maturin renders back.
"""

import re
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from ._project import RunError

# A release tag: `v` and a PEP 440 version.
TAG = re.compile(r"^v(\d+(?:\.\d+)*\S*)$")
# Cargo's spelling of a PEP 440 development release, the only pre-release this layout writes.
CARGO_DEV = re.compile(r"^(\d+(?:\.\d+)*)-dev\.(\d+)$")
PROJECT_VERSION = re.compile(r'^(version\s*=\s*")([^"]*)(")', re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Member:
    """One distribution of the repository."""

    name: str
    pyproject: Path
    # `None` when `[project] version` is dynamic: maturin reads it from the Cargo workspace.
    version: str | None
    requirements: Sequence[str]


@dataclass(frozen=True, slots=True)
class Distributions:
    root: Path
    members: Sequence[Member]
    # The Cargo workspace's `[workspace.package] version`, converted to PEP 440, or `None`.
    cargo: str | None

    @property
    def names(self) -> frozenset[str]:
        return frozenset(m.name for m in self.members)

    def versions(self) -> Mapping[str, str]:
        """Each place a version is written, by the file that holds it, in PEP 440."""
        found = {}
        for member in self.members:
            if member.version is not None:
                found[_relative(self.root, member.pyproject)] = member.version
        if self.cargo is not None:
            found["Cargo.toml"] = self.cargo
        return found


def _load(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise RunError(f"{path.name} does not parse: {e}") from None


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def to_cargo(version: str) -> str:
    """`X.Y.Z.devN` in Cargo's spelling; a final release is spelled the same in both."""
    parsed = Version(version)
    if parsed.is_devrelease and not parsed.pre and not parsed.post:
        return f"{parsed.base_version}-dev.{parsed.dev}"
    if parsed.is_prerelease or parsed.is_postrelease or parsed.local:
        raise RunError(f"{version} has no Cargo spelling; only X.Y.Z and X.Y.Z.devN do")
    return parsed.base_version


def from_cargo(version: str) -> str:
    if match := CARGO_DEV.fullmatch(version):
        return f"{match[1]}.dev{match[2]}"
    return version


def _member(path: Path) -> Member | None:
    project = _load(path).get("project")
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        return None
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    version = project.get("version")
    return Member(
        canonicalize_name(project["name"]),
        path,
        version if isinstance(version, str) else None,
        tuple(requirements),
    )


def distributions(root: Path) -> Distributions:
    """The repository's distributions, read from the root `pyproject.toml` and `Cargo.toml`."""
    pyproject = _load(root / "pyproject.toml")
    workspace = pyproject.get("tool", {}).get("uv", {}).get("workspace", {})
    paths = [root / "pyproject.toml"]
    if patterns := workspace.get("members"):
        excluded = {p for pattern in workspace.get("exclude", []) for p in root.glob(pattern)}
        paths += sorted(
            {
                directory / "pyproject.toml"
                for pattern in patterns
                for directory in root.glob(pattern)
                if directory not in excluded and (directory / "pyproject.toml").is_file()
            }
        )
    members = [m for m in map(_member, paths) if m is not None]
    cargo = _load(root / "Cargo.toml").get("workspace", {}).get("package", {}).get("version")
    return Distributions(
        root, tuple(members), from_cargo(cargo) if isinstance(cargo, str) else None
    )


def sibling_pins(dists: Distributions, member: Member) -> Sequence[tuple[str, str]]:
    """`member`'s requirements on other members: each requirement as written, and its name."""
    found = []
    for text in member.requirements:
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            continue
        name = canonicalize_name(requirement.name)
        if name in dists.names and name != member.name:
            found.append((text, name))
    return found


def is_exact(text: str, version: str) -> bool:
    specifier = list(Requirement(text).specifier)
    return len(specifier) == 1 and specifier[0].operator == "==" and specifier[0].version == version


def release_tags(root: Path) -> tuple[Sequence[Version], bool, bool]:
    """Every `vX.Y.Z` tag's version, whether one points at HEAD, and whether the clone is shallow.

    Raises `RunError` where git cannot answer.
    """

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            raise RunError(f"cannot read the release tags with git: {e}") from None

    def versions(listing: str) -> list[Version]:
        found = []
        for line in listing.split():
            if match := TAG.fullmatch(line):
                try:
                    found.append(Version(match[1]))
                except InvalidVersion:
                    continue
        return found

    shallow = git("rev-parse", "--is-shallow-repository").strip() == "true"
    # A repository with no commit yet has no HEAD to ask about.
    born = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "HEAD"], cwd=root, capture_output=True
        ).returncode
        == 0
    )
    head = born and bool(versions(git("tag", "--points-at", "HEAD")))
    return versions(git("tag", "--list", "v*")), head, shallow


def _rewrite_project_version(text: str, version: str) -> str:
    """`text` with the `version` key of its `[project]` table set to `version`."""
    start = re.search(r"^\[project\]\s*$", text, re.MULTILINE)
    if start is None:
        return text
    end = re.search(r"^\[", text[start.end() :], re.MULTILINE)
    stop = start.end() + end.start() if end else len(text)
    table = text[start.end() : stop]
    table, count = PROJECT_VERSION.subn(rf"\g<1>{version}\g<3>", table, count=1)
    return text[: start.end()] + table + text[stop:] if count else text


def _rewrite_pins(text: str, names: frozenset[str], version: str) -> str:
    """`text` with every quoted `==` requirement on one of `names` pinned at `version`."""

    def pin(match: re.Match[str]) -> str:
        if canonicalize_name(match["name"]) not in names:
            return match[0]
        quote, extras = match["quote"], match["extras"] or ""
        return f"{quote}{match['name']}{extras}=={version}{match['rest']}{quote}"

    pattern = re.compile(
        r"(?P<quote>[\"'])(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<extras>\[[^\]]*\])?"
        r"\s*==\s*[^\"';\s]+(?P<rest>[^\"']*)(?P=quote)"
    )
    return pattern.sub(pin, text)


def _rewrite_cargo(text: str, version: str) -> str:
    head, sep, tail = text.partition("[workspace.package]\n")
    if not sep:
        return text
    end = re.search(r"^\[", tail, re.MULTILINE)
    stop = end.start() if end else len(tail)
    table, count = re.subn(
        r'^(version\s*=\s*")[^"]*(")',
        rf"\g<1>{to_cargo(version)}\g<2>",
        tail[:stop],
        count=1,
        flags=re.MULTILINE,
    )
    return head + sep + table + tail[stop:] if count else text


def write_version(dists: Distributions, version: str) -> Sequence[Path]:
    """Write `version` into every member, every pin between members and Cargo; the files changed."""
    Version(version)
    changed = []
    for member in dists.members:
        text = member.pyproject.read_text(encoding="utf-8")
        new = _rewrite_pins(text, dists.names - {member.name}, version)
        if member.version is not None:
            new = _rewrite_project_version(new, version)
        if new != text:
            member.pyproject.write_text(new, encoding="utf-8")
            changed.append(member.pyproject)
    cargo = dists.root / "Cargo.toml"
    if dists.cargo is not None:
        text = cargo.read_text(encoding="utf-8")
        if (new := _rewrite_cargo(text, version)) != text:
            cargo.write_text(new, encoding="utf-8")
            changed.append(cargo)
    return changed


def changelog_section(root: Path, version: str) -> bool | None:
    """Whether `CHANGELOG.md` has a `## [version]` section; `None` without a `CHANGELOG.md`."""
    path = root / "CHANGELOG.md"
    if not path.is_file():
        return None
    heading = re.compile(rf"^## \[{re.escape(version)}\]", re.MULTILINE)
    return heading.search(path.read_text(encoding="utf-8")) is not None
