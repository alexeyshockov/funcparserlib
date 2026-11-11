"""Every environment the tests run in.

`just tests` runs the sessions tagged `local`, one after another; `.github/workflows/tests.yaml`
runs the ones tagged `ci`, one job each. This file is this project's to edit.

In a uv workspace each distribution is tested on its own: its session installs that distribution and
its declared dependencies only, so an import of another distribution it does not declare fails here
even though the shared `.venv` holds it.
"""

import os
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
# A third of the cores, at least 2.
WORKERS = ("-n", str(max(2, (os.cpu_count() or 2) // 3)), "--dist", "worksteal")


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
    # `shadow` runs this suite on the floor locally.
    [nox.param(v, tags=["ci"]) for v in VERSIONS] + [nox.param(f"{NEWEST}t", tags=["ci", "local"])],
)
@nox.parametrize("member", list(MEMBERS))
def tests(session: nox.Session, member: str) -> None:
    """One member's suite, on every supported version and the newest free-threaded build."""
    sync(session, *(["--package", MEMBERS[member]] if WORKSPACE else []))
    session.chdir(member)
    session.run("pytest", *WORKERS, *session.posargs)


@nox.session(python=FLOOR, tags=["ci", "local"])
def shadow(session: nox.Session) -> None:
    """The suite with every parse also run by the compiled parser: see `tests/shadow.py`."""
    sync(session)
    session.run("pytest", *WORKERS, "-p", "tests.shadow", *session.posargs)


@nox.session(python=FLOOR, tags=["ci", "local"])
def exhaustive(session: nox.Session) -> None:
    """Every small grammar, compiled and interpreted: see `tests/test_exhaustive.py`."""
    sync(session)
    session.run(
        "pytest", *WORKERS, "-m", "exhaustive", "tests/test_exhaustive.py", *session.posargs
    )


# One interpreter produces one report, so coverage is outside the matrix.
@nox.session(python=Path(".python-version").read_text().strip())
def coverage(session: nox.Session) -> None:
    """The suite with branch coverage, which `just coverage` and `ci.yaml`'s coverage job run."""
    sync(session)
    session.run(
        "pytest", *WORKERS, "--cov-report=term", "--cov-report=xml", "--cov", *session.posargs
    )


# One micro-benchmark session per interpreter; the one tagged `bench` is what `just bench::micro`
# runs. It is the newest unless a benchmark dependency has no wheel for it, then the tag moves.
BENCH_PYTHONS = [nox.param(v, tags=["bench"] if v == NEWEST else []) for v in VERSIONS]

if any(Path(member, "benchmarks").is_dir() for member in MEMBERS):

    @nox.session
    @nox.parametrize("python", BENCH_PYTHONS)
    def bench(session: nox.Session) -> None:
        """The micro-benchmarks on one interpreter; the arguments go to pytest."""
        sync(session, "--group", "bench", *(["--all-packages"] if WORKSPACE else []))
        session.run("pytest", *session.posargs)
