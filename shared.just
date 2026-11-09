# The common recipes, imported by the justfile.
# Generated; propose changes in an issue rather than editing this file.
#
# Dev tooling comes from three places:
#   1. Linters and formatters (ruff, typos, prettier, actionlint, zizmor, ...): pinned by rev in
#      .pre-commit-config.yaml and run through `prek run`, so no local binary can drift from CI's.
#   2. Tools that read the project from outside (type checkers, import-linter, vulture, griffe,
#      the docs builder): one uv project under `tools/` on its own interpreter, never installed
#      into .venv, so they do not constrain the project's resolution.
#   3. Tools that run inside the project's environment (deptry, slotscheck): the project's `lint`
#      dependency group.
# uv, just and prek are the host tools `just doctor` installs; they are the only unpinned ones.
#
# A recipe either asserts or writes: `check` and `gate` are read-only, and `fix` is the only
# recipe here that changes a file. The hooks are read-only at the `pre-commit` stage for the same
# reason (see .pre-commit-config.yaml).
#
# The importing justfile provides `conventions` (the first step of `gate`), `frozen_revs` (the
# project's own frozen hook revs, empty when it has none), and its own recipes.

# Runs a tool from the toolbox, syncing its environment on demand. `--locked` fails when
# tools/pyproject.toml is ahead of tools/uv.lock. `env -u VIRTUAL_ENV` silences uv's warning about
# an activated .venv. `UV_PROJECT_ENVIRONMENT` (relative to `tools/`) puts the environment at
# `.venv-tools` beside `.venv`; it is not exported, or `just venv` would sync there too.
tools := "env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../.venv-tools uv run --locked --project tools"

# Variables are evaluated when a recipe uses them, so the lookups below cost nothing elsewhere.
set lazy

# Each package as `<distribution directory>/<import name>`, and each distribution's directory once
# (`.` for the root), space-separated: `[tool.conventions] packages`, else the root distribution's,
# else each workspace member's.
packages := `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../.venv-tools uv run --locked --project tools conventions packages`
dists := `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../.venv-tools uv run --locked --project tools conventions packages --dirs`

# Each distribution's tests, and the `typing_tests/` and `benchmarks/` directories that exist.
test_dirs := shell('for d in $1; do [ "$d" = . ] && printf "tests " || printf "%s/tests " "$d"; done', dists)
suites := shell('for d in $1; do s=$([ "$d" = . ] && echo typing_tests || echo "$d/typing_tests"); [ -d "$s" ] && printf "%s " "$s"; done; true', dists)
bench_dirs := shell('for d in $1; do b=$([ "$d" = . ] && echo benchmarks || echo "$d/benchmarks"); [ -d "$b" ] && printf "%s " "$b"; done; true', dists)

[doc("The pinned tool environment (tools/): the versions in force, the upgrade, the host setup")]
mod tools

import? 'docs.just'
[doc("The benchmarks (benchmarks/justfile): run, compare, type-check")]
mod bench 'benchmarks/justfile'

[doc("Install the host tools (uv, just, prek), build the pinned tool environment, and install the commit hooks")]
doctor:
    @just tools::doctor

# `uv sync` removes every group it does not name, so a re-sync names them all or drops `lint`.
[doc("Build .venv from the lock (uv sync, every group and extra)")]
venv:
    uv sync --all-groups --all-extras

[doc("Upgrade the locked dependency versions, then rebuild .venv")]
venv-upgrade:
    uv lock --upgrade
    uv sync --all-groups --all-extras

[doc("Upgrade the locked tool versions within tools/pyproject.toml's constraints, then rebuild .venv-tools")]
tools-upgrade:
    @just tools::upgrade

[doc("Run any tool from the pinned tool environment (tools/): `just tool toolbox ty -- <path>`")]
tool *args:
    @just tools::run {{ args }}

[doc("Report every tool version in force, and which could move (read-only)")]
pins:
    @just tools::pins

# The shared hooks' revs pinned on purpose, as `<repo>=<rev>` with the last segment of the hook's
# repository URL; the reason is commented beside the hook in .pre-commit-config.yaml.
shared_frozen_revs := "typos=v1.50.3"

[doc("Verify the pre-commit revs that must not move (shared_frozen_revs, frozen_revs) have not moved")]
frozen-revs:
    @just tools::frozen-revs '{{ shared_frozen_revs }},{{ frozen_revs }}'

# --- Types ---

# `toolbox` runs each checker once per distribution, at the floor (the version the first
# distribution's `requires-python` starts from), against `.venv` through the checker's
# interpreter option. mypy reads the packages strict and the examples, tests and benchmarks at
# the depth `mypy.ini` gives them.
[doc("Check types of the packages, the examples, the tests and the benchmarks (mypy, blocking)")]
types:
    {{ tools }} toolbox mypy -- {{ packages }} examples {{ test_dirs }} {{ bench_dirs }}

# `-` ignores a failure: these run so a finding is seen. basedpyright reads
# `tools/basedpyright.json` here only.
[doc("Second opinions on the packages from ty, pyrefly and basedpyright (informational, never blocks)")]
types-info:
    -{{ tools }} toolbox ty -- {{ packages }}
    -{{ tools }} toolbox pyrefly check {{ packages }}
    -{{ tools }} toolbox basedpyright --project=tools/basedpyright.json -- {{ packages }}

[doc("Check types (the gates and the second opinions)")]
types-all: types types-api types-info tools::types

# `--public-only` counts the symbols reachable through the public modules and their re-exports,
# each reported where it is defined. The threshold is pyrefly's default, 100%. A public variable or
# attribute needs an annotation unless it is a literal constant; `Any` counts as annotated.
[doc("Check that the public API is fully annotated (pyrefly coverage, gated at 100%)")]
type-coverage:
    {{ tools }} toolbox pyrefly coverage check --public-only {{ packages }}

# Each distribution's `typing_tests/` asserts (`assert_type`) what a user's checker infers from the
# public API, with one ignore per checker on each call the API must reject. Every checker gates at
# the package's strictness, fails on a warning, reports an ignore that suppresses nothing, and reads
# only its own ignores (pyrefly by `enabled-ignores`). basedpyright reads its defaults here, not
# `tools/basedpyright.json`. One run per distribution: every suite is a `typing_tests` package, and
# one run over two is a duplicate module. All checkers run whatever the others report.
[doc("Type-check the public API as users' checkers see it (mypy, ty, pyrefly, basedpyright; blocking)")]
types-api:
    #!/usr/bin/env bash
    [ -n "{{ suites }}" ] || { echo "types-api: no distribution has a typing_tests/"; exit 1; }
    status=0
    {{ tools }} toolbox mypy -- {{ suites }} || status=1
    {{ tools }} toolbox ty --error-on-warning -- {{ suites }} || status=1
    {{ tools }} toolbox pyrefly check --error unused-ignore -- {{ suites }} || status=1
    {{ tools }} toolbox basedpyright --warnings -- {{ suites }} || status=1
    exit "$status"

# --- Tests ---

# The noxfile's sessions tagged `local` run here in turn; those tagged `ci` are the matrix of
# `.github/workflows/tests.yaml`. Arguments pass to nox (`just tests --list`). nox runs the union of
# every selection, so `--tags local` is added only when the arguments select nothing.
[positional-arguments]
[doc("Run the local test matrix: the noxfile's sessions tagged `local`")]
tests *args:
    #!/usr/bin/env bash
    set -euo pipefail
    for arg in "$@"; do
        case "$arg" in
            -s | --session* | --sessions* | -t | --tags* | -k | --keywords* | -p | --python*) exec just tools::nox "$@" ;;
        esac
    done
    exec just tools::nox --tags local "$@"

# pytest runs in `.venv` with every distribution installed, so an undeclared import of another
# distribution passes here and fails in `tests`. It runs from the current directory and takes its
# configuration from the distribution the paths are in, so one call covers one distribution.
[no-cd]
[positional-arguments]
[doc("Run pytest in .venv from the current directory, on one distribution's tests (not part of the PR checks, `just gate`)")]
test *args:
    "{{ justfile_directory() }}/.venv/bin/python" -m pytest "$@"

# --- The gate ---

# Cheapest first. `.github/workflows/ci.yaml`'s `gate` job runs each dependency as its own step and
# the conventions runner holds the two lists equal (CON700-CON704); `tests` is the exception,
# run as the matrix of `.github/workflows/tests.yaml`. The hooks are not here: pre-commit.ci runs
# them.
[doc("Everything a pull request must pass, the local test matrix last")]
gate: conventions imports deps slots types type-coverage types-api tools::types tests


# --- Structure ---

# Runs per distribution, from its directory: import-linter finds the package from where it runs.
[doc("Check import-graph architecture contracts (import-linter, config in pyproject.toml)")]
imports:
    #!/usr/bin/env bash
    set -euo pipefail
    for dist in {{ dists }}; do just tools::imports "$dist"; done

# Runs per distribution, so an undeclared import of another distribution is a finding though the
# shared `.venv` holds it.
[doc("Check that declared dependencies are the ones the code imports (deptry, config in pyproject.toml)")]
deps:
    #!/usr/bin/env bash
    set -euo pipefail
    for package in {{ packages }}; do just tools::deps "$(dirname "$package")" "$(basename "$package")"; done

[doc("Check __slots__ effectiveness across the class hierarchy (slotscheck, config in pyproject.toml)")]
slots:
    #!/usr/bin/env bash
    set -euo pipefail
    for package in {{ packages }}; do just tools::slots "$(dirname "$package")" "$(basename "$package")"; done

[doc("Find unused code with vulture (config in pyproject.toml). Pass extra args to override.")]
deadcode *args:
    @just tools::deadcode . {{ args }}

# --- Check and fix ---

[doc("Read-only check: every applicable commit hook, plus mypy for Python paths. No paths = whole tree.")]
check *paths:
    @just tools::check {{ paths }}

[doc("Apply every fixer whose finding the PR checks (`just gate`) can only report. `check` and `gate` never write.")]
fix *paths:
    @just tools::fix {{ paths }}
