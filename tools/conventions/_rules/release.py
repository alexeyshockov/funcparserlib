"""CON410-CON414: one development version, across every distribution, above the last release."""

from collections.abc import Iterator

from packaging.version import InvalidVersion, Version

from .._engine import Finding, rule
from .._project import Project
from .._versions import distributions, is_exact, release_tags, sibling_pins


@rule("CON410", "CON411", "CON412", "CON413", "CON414")
def versions(project: Project) -> Iterator[Finding]:
    """The repository holds one development version, above its newest release.

    CON410: Every distribution, and the Cargo workspace where there is one, has the same version.

    Reason: a repository is released as one version; a member at another number would ship under a
    version its source never stated.

    Fix: `just version X.Y.Z.devN` writes the one version everywhere.

    CON411: A distribution's requirement on another distribution of the repository is pinned with
    `==` at the shared version.

    Reason: members are built and tested together at one commit and nowhere else; a range claims
    compatibility with releases nothing tested.

    Fix: write `<member>==<version>`; `just version` keeps it in step from then on.

    CON412: The version in the tree is a development release, `X.Y.Z.devN`.

    Reason: the release job writes the released version from the tag; a tree that already holds one
    is a release commit, which this flow does not make.

    Fix: `just version X.Y.Z.dev0`, with `X.Y.Z` the next release.

    CON413: The version in the tree is above the newest release tag, unless a release tag points at
    the commit.

    Reason: after a release the tree still holds that release's development version, and everything
    committed on top would be built as an older version than the one shipped.

    Fix: set the next development version with `just version`.

    CON414: The clone has the release tags to compare against.

    Reason: a shallow clone without tags would pass CON413 on every commit.

    Fix: check out with `fetch-depth: 0` (or `fetch-tags: true`).
    """
    dists = distributions(project.root)
    places = dists.versions()
    if not places:
        return
    (first_path, first), *rest = places.items()
    for path, version in rest:
        if version != first:
            yield Finding(
                path,
                _line(project, path),
                "CON410",
                f"version {version} differs from {first_path}'s {first}",
            )
    for member in dists.members:
        for text, _name in sibling_pins(dists, member):
            if not is_exact(text, first):
                path = project.relative(member.pyproject)
                yield Finding(
                    path,
                    project.line_of(member.pyproject, text),
                    "CON411",
                    f"`{text}` is not pinned at `=={first}`",
                )
    try:
        current = Version(first)
    except InvalidVersion:
        yield Finding(first_path, _line(project, first_path), "CON412", f"{first} is not PEP 440")
        return
    if not current.is_devrelease:
        yield Finding(
            first_path,
            _line(project, first_path),
            "CON412",
            f"{first} is not a development version; the tree holds X.Y.Z.devN",
        )
    tags, on_head, shallow = release_tags(project.root)
    if not tags and shallow:
        yield Finding(".", None, "CON414", "a shallow clone without tags; fetch the release tags")
        return
    if on_head or not tags:
        return
    newest = max(tags)
    if current <= newest:
        yield Finding(
            first_path,
            _line(project, first_path),
            "CON413",
            f"v{newest} is released and the tree still holds {first}; set the next version",
        )


def _line(project: Project, relative: str) -> int | None:
    return project.line_of(project.root / relative, "version")
