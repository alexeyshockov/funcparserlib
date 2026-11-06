"""Every environment the tests run in.

`just tests` runs the sessions tagged `local`, one after another; `.github/workflows/tests.yaml`
runs the ones tagged `ci`, one job each. This file is this project's to edit.

In a uv workspace each distribution is tested on its own: its session installs that distribution and
its declared dependencies only, so an import of another distribution it does not declare fails here
even though the shared `.venv` holds it.
"""

from pathlib import Path

import nox

nox.needs_version = ">=2026.8.17"
nox.options.default_venv_backend = "uv"

ROOT = nox.project.load_toml("pyproject.toml")
WORKSPACE = ROOT.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
# Each member's directory and distribution name; the root itself when it is not a workspace.
MEMBERS = {
    path.as_posix(): nox.project.load_toml(path / "pyproject.toml")["project"]["name"]
    for pattern in WORKSPACE
    for path in sorted(Path().glob(pattern))
    if (path / "pyproject.toml").is_file()
} or {".": ROOT["project"]["name"]}
FIRST = nox.project.load_toml(Path(next(iter(MEMBERS))) / "pyproject.toml")
VERSIONS = nox.project.python_versions(FIRST)
FLOOR, NEWEST = VERSIONS[0], VERSIONS[-1]


def sync(session: nox.Session, *args: str) -> None:
    """Install into the session's environment; `uv sync` removes whatever it was not asked for.

    `--python` keeps the session's interpreter; uv would otherwise use `.python-version`'s.
    """
    session.run_install(
        "uv",
        "sync",
        "--locked",
        "--python",
        str(session.python),
        *args,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session
@nox.parametrize(
    "python",
    [nox.param(v, tags=["ci", "local"] if v == FLOOR else ["ci"]) for v in VERSIONS]
    + [nox.param(f"{NEWEST}t", tags=["ci", "local"])],
)
@nox.parametrize("member", list(MEMBERS))
def tests(session: nox.Session, member: str) -> None:
    """One member's suite, on every supported version and the newest free-threaded build."""
    sync(session, *(["--package", MEMBERS[member]] if WORKSPACE else []))
    session.chdir(member)
    session.run("pytest", *session.posargs)


@nox.session(python=FLOOR, tags=["ci", "local"])
def shadow(session: nox.Session) -> None:
    """The suite with every parse also run by the compiled parser: see `tests/shadow.py`."""
    sync(session)
    session.run("pytest", "-p", "tests.shadow", *session.posargs)


# The micro-benchmarks (`just bench::micro`), where the project has them: one session per
# interpreter. The one tagged `bench` is what `just bench::micro` runs: the newest, unless a
# dependency the benchmarks build has no wheel for it yet, in which case the tag moves to another
# version in BENCH_PYTHONS.
BENCH_PYTHONS = [nox.param(v, tags=["bench"] if v == NEWEST else []) for v in VERSIONS]

if any(Path(member, "benchmarks").is_dir() for member in MEMBERS):

    @nox.session
    @nox.parametrize("python", BENCH_PYTHONS)
    def bench(session: nox.Session) -> None:
        """The micro-benchmarks on one interpreter; the arguments go to pytest."""
        sync(session, "--group", "bench", *(["--all-packages"] if WORKSPACE else []))
        session.run("pytest", *session.posargs)
