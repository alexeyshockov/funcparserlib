"""The `conventions` command.

Exit status: 0 when the repository is clean, 1 when there are findings, 2 when the run could not
produce a verdict (configuration, repository, or an internal error).
"""

import os
import signal
import subprocess
import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import click
from packaging.version import InvalidVersion, Version

from . import _rules  # noqa: F401  # registers the rules
from ._api import diff
from ._engine import FIXERS, RULES, is_warning, run
from ._output import FORMATS, render
from ._project import Project, RunError
from ._rules.release import versions as version_rules
from ._versions import TAG, changelog_section, distributions, write_version

CLEAN, FINDINGS, NO_VERDICT = 0, 1, 2

# Seconds a run may take, with room for the docs vocabulary, which runs basedpyright.
DEADLINE = 120.0


def _toplevel() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout
    except OSError, subprocess.CalledProcessError:
        raise RunError("not inside a git repository; pass --root") from None
    return Path(out.strip())


def _fail(message: str) -> NoReturn:
    click.echo(f"conventions: {message}", err=True)
    sys.exit(NO_VERDICT)


class _Overdue(BaseException):
    """The deadline passed. A `BaseException`, so no rule's `except Exception` can swallow it."""


@contextmanager
def _verdict() -> Generator[None]:
    """Turn what keeps a run from a verdict into exit 2: config, a crash, or the deadline."""

    def overdue(_signum: int, _frame: object) -> None:
        raise _Overdue

    previous = signal.signal(signal.SIGALRM, overdue)
    signal.setitimer(signal.ITIMER_REAL, DEADLINE)
    try:
        yield
    except RunError as e:
        _fail(str(e))
    except _Overdue:
        _fail(f"no verdict after {DEADLINE:g}s; a rule is looping, which is a bug in the runner")
    except Exception:
        traceback.print_exc()
        _fail("internal error; the traceback above is a bug in the runner, not a finding")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@click.group()
def cli() -> None:
    """Check the repository against the rules in `CONVENTIONS.md`."""


@cli.command()
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The repository root (default: the git toplevel of the working directory).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(FORMATS),
    help="Output format (default: github under GitHub Actions, else text).",
)
def check(root: Path | None, fmt: str | None) -> None:
    """Run every rule over the whole repository."""
    if fmt is None:
        fmt = "github" if os.environ.get("GITHUB_ACTIONS") == "true" else "text"
    with _verdict():
        found = run(Project(root or _toplevel()))
    findings = [f for f in found if not is_warning(f)]
    warnings = [f for f in found if is_warning(f)]
    if fmt == "text":
        click.echo(render(findings, fmt), nl=False)
        for line in render(warnings, fmt).splitlines():
            click.secho(f"warning: {line}", err=True, bold=True, fg="yellow")
    else:
        click.echo(render(found, fmt), nl=False)
    if findings and fmt == "text":
        click.echo(
            f"{len(findings)} findings; `conventions explain CODE` gives the reason and the fix.",
            err=True,
        )
    sys.exit(FINDINGS if findings else CLEAN)


@cli.command()
@click.argument("codes", nargs=-1)
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The repository root (default: the git toplevel of the working directory).",
)
def fix(codes: tuple[str, ...], root: Path | None) -> None:
    """Rewrite what the fixers can resolve: every fixer, or those for CODES.

    Exits 1 when a fixer left something for a person to do, and prints it.
    """
    if unknown := sorted(set(codes) - set(FIXERS)):
        _fail(f"no fixer for {', '.join(unknown)}; fixers exist for {', '.join(sorted(FIXERS))}")
    selected = list(dict.fromkeys(FIXERS[code] for code in (codes or sorted(FIXERS))))
    root = root or _toplevel()
    refused: list[str] = []
    for each in selected:
        # Each fixer reads the tree the previous one left.
        with _verdict():
            fixes = each(Project(root))
        for path, text in fixes.rewrites.items():
            path.write_text(text, encoding="utf-8")
            click.echo(f"fixed {path.relative_to(root.resolve()).as_posix()}")
        refused += fixes.refused
    for line in refused:
        click.echo(line, err=True)
    sys.exit(FINDINGS if refused else CLEAN)


@cli.command()
@click.argument("code")
def explain(code: str) -> None:
    """Print a rule's reason and fix."""
    if code not in RULES:
        _fail(f"no rule {code}; `conventions rules` lists them")
    click.echo(f"{code}: {RULES[code].doc}")


@cli.command()
def rules() -> None:
    """List every code with its summary."""
    for code in sorted(RULES):
        click.echo(f"{code}  {RULES[code].summary}")


ROOT_OPTION = click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The repository root (default: the git toplevel of the working directory).",
)


@cli.command("api-diff")
@click.argument("base")
@ROOT_OPTION
def api_diff(base: str, root: Path | None) -> None:
    """Print what a caller of the public API at BASE meets in the working tree, one change a line.

    Removals and changes only: an addition breaks no caller. Exit 1 when anything changed.
    """
    with _verdict():
        found = diff(root or _toplevel(), base)
    for line in found:
        click.echo(line)
    if not found:
        click.echo(f"no change to the public API since {base}")
    sys.exit(FINDINGS if found else CLEAN)


@cli.command()
@click.option("--dirs", is_flag=True, help="Print each distribution's directory once instead.")
@ROOT_OPTION
def packages(dirs: bool, root: Path | None) -> None:
    """Print each package as `<distribution directory>/<import name>`, separated by spaces.

    The packages are `[tool.conventions] packages`, else the root distribution's, else each
    workspace member's. A package of the root distribution prints as its import name alone, and its
    directory as `.`.
    """
    with _verdict():
        paths = Project(root or _toplevel()).settings["packages"]
    if dirs:
        paths = dict.fromkeys(path.rpartition("/")[0] or "." for path in paths)
    click.echo(" ".join(paths))


@cli.command()
@click.argument("version", required=False)
@ROOT_OPTION
def version(version: str | None, root: Path | None) -> None:
    """Print the repository's version, or write VERSION into every distribution.

    Writing sets each member's `[project] version`, every `==` pin between members and the Cargo
    workspace version. The lock is the caller's to update.
    """
    root = root or _toplevel()
    with _verdict():
        dists = distributions(root)
        if not dists.members:
            raise RunError("no distribution: neither the root nor a workspace member has [project]")
        if version is None:
            click.echo("\n".join(sorted(set(dists.versions().values()))))
            return
        for path in write_version(dists, version):
            click.echo(f"wrote {path.relative_to(root.resolve()).as_posix()}")


@cli.command()
@click.argument("tag")
@click.option("--write", is_flag=True, help="Rewrite the tree to the released version.")
@ROOT_OPTION
def release(tag: str, write: bool, root: Path | None) -> None:
    """Check that TAG (`vX.Y.Z`) may release this tree; with --write, write X.Y.Z into it.

    The tree must pass the version rules (CON410-CON412), hold `X.Y.Z.devN`, and have a `## [X.Y.Z]`
    section in `CHANGELOG.md` when there is one. Exits 1 when it does not.
    """
    root = root or _toplevel()
    if not (match := TAG.fullmatch(tag)):
        _fail(f"{tag} is not a release tag, vX.Y.Z")
    released = match[1]
    try:
        final = Version(released).base_version == released
    except InvalidVersion:
        final = False
    if not final:
        _fail(f"{tag} is not a release tag, vX.Y.Z")
    with _verdict():
        project = Project(root)
        dists = distributions(project.root)
        problems = [
            f"{f.path}:{f.line or 0}: {f.code} {f.message}"
            for f in version_rules(project)
            if f.code in {"CON410", "CON411", "CON412"}
        ]
        held = sorted(set(dists.versions().values()))
        if not problems and {Version(v).base_version for v in held} != {released}:
            problems.append(f"the tree holds {', '.join(held)}, not {released}.devN")
        if changelog_section(root, released) is False:
            problems.append(f"CHANGELOG.md has no `## [{released}]` section")
        if problems:
            click.echo("\n".join(problems), err=True)
            sys.exit(FINDINGS)
        if not write:
            return
        for path in write_version(dists, released):
            click.echo(f"wrote {path.relative_to(root.resolve()).as_posix()}")
        rewritten = Project(root)
        if left := [
            f"{f.path}:{f.line or 0}: {f.code} {f.message}"
            for f in version_rules(rewritten)
            if f.code in {"CON410", "CON411"}
        ]:
            click.echo("\n".join(left), err=True)
            sys.exit(FINDINGS)


def main() -> None:
    cli(prog_name="conventions")
