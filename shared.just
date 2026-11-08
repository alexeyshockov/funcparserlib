# The common recipes, imported by the justfile.
# Generated; propose changes in an issue rather than editing this file.
#
# Dev tooling comes from three places:
#
#   1. Self-contained linters/formatters (ruff, typos, prettier, actionlint, zizmor, ...) —
#      pinned by rev in .pre-commit-config.yaml, run through `prek run`. Nothing here
#      calls a bare `ruff`, so the local binary can never drift from the CI one.
#   2. Tools that read the project from outside (type checkers, import-linter, vulture, griffe,
#      the docs builder) — one uv project under `tools/`, constrained by its pyproject.toml
#      and pinned by its uv.lock, on its own interpreter. Never installed into .venv: such a
#      tool must not constrain the project's resolution.
#   3. Tools that run inside the project's environment (deptry, slotscheck) — the project's
#      `lint` dependency group, resolved and locked with the project.
#
# uv itself, just and prek are the host tools `just doctor` installs; nothing gates on their
# versions, which is why they are the only three left unpinned.
#
# Cutting across both: a recipe either asserts or it writes, never both. `check` and
# `gate` are read-only; `fix` is the only recipe in this file that changes a file. That is
# why the hooks are configured read-only at the `pre-commit` stage and their write-mode
# twins sit at the `manual` stage — see the header of .pre-commit-config.yaml for what a
# gate that rewrote its own subject was able to hide.
#
# The importing justfile provides `conventions` (the first step of `gate`), the `frozen_revs`
# list of the project's own frozen hook revs (empty when it has none), plus whatever is the
# project's own.

# Runs a tool from the toolbox. `--project tools` syncs the toolbox environment on demand.
# `--locked` fails when tools/pyproject.toml is ahead of tools/uv.lock, instead of re-resolving and
# rewriting it. `env -u VIRTUAL_ENV` silences uv's warning about an activated .venv.
#
# `UV_PROJECT_ENVIRONMENT` puts the toolbox's environment at `.venv-tools` in the repository root
# rather than nesting one inside `tools/`. It is resolved against the project directory, so
# `../.venv-tools` lands beside `.venv`, which is where `.gitignore` and `.claudeignore` already
# expect a `.venv-*` to be. It is set per invocation and never exported: exported, it would send
# `just venv` — an ordinary `uv sync` of the project — to the same place.
tools := "env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../.venv-tools uv run --locked --project tools"

# A variable is evaluated when a recipe uses it, so the lookups below cost nothing elsewhere.
set lazy

# Each package as `<distribution directory>/<import name>`, and each distribution's directory once
# (`.` for the root), space-separated: `[tool.conventions] packages`, else the root distribution's,
# else each workspace member's.
packages := `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../.venv-tools uv run --locked --project tools conventions packages`
dists := `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=../.venv-tools uv run --locked --project tools conventions packages --dirs`

# Each distribution's tests, and the typesafety suites beside its package that exist. `toolbox`
# runs each type checker at the floor (the version the first distribution's `requires-python`
# starts from) against `.venv`, a file ruff's `per-file-target-version` pins at its own version,
# and ty with each distribution's own view of the workspace.
test_dirs := shell('for d in $1; do [ "$d" = . ] && printf "tests " || printf "%s/tests " "$d"; done', dists)
typesafety_dirs := shell('for d in $1; do [ "$d" = . ] && printf "typesafety " || printf "%s/typesafety " "$d"; done', dists)
api_suites := shell('for d in $1; do [ -d "$d" ] && printf "%s " "$d"; done; true', typesafety_dirs)

# The recipes of the pinned tool environment (`tools/`) live in `tools/justfile`, as a just module.
# The recipes below delegate to it, passing what differs per distribution as arguments.
[doc("The pinned tool environment (tools/): the versions in force, the upgrade, the host setup")]
mod tools

# A feature's recipes, where the project has the feature.
import? 'docs.just'
[doc("The benchmarks (benchmarks/justfile): run, compare, type-check")]
mod bench 'benchmarks/justfile'

[doc("Install the host tools (uv, just, prek), build the pinned tool environment, and install the commit hooks")]
doctor:
    @just tools::doctor

# Makes `.venv` match the lock; it does not check or fetch dependencies. `deps` judges them.
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

# The shared hooks' revs pinned on purpose, as `<repo>=<rev>`, where `<repo>` is the last segment
# of the hook's repository URL. The reason for each is commented beside its hook in
# .pre-commit-config.yaml, which arrives with this list in the same update.
shared_frozen_revs := "typos=v1.50.3"

[doc("Verify the pre-commit revs that must not move (shared_frozen_revs, frozen_revs) have not moved")]
frozen-revs:
    @just tools::frozen-revs '{{ shared_frozen_revs }},{{ frozen_revs }}'


# --- Types ---

# mypy gates the code the project owns: the packages and their typesafety suites strict, the examples
# and the tests at the depth `mypy.ini` gives them. It analyses `.venv` through `--python-executable`
# while running from the toolbox, once per distribution.
# Presence of annotations on the public API is pyrefly's question (`type-coverage`); this one is
# correctness.
[doc("Check types of the packages, the typesafety suites, the examples and the tests (mypy, blocking)")]
types:
    {{ tools }} toolbox mypy -- {{ packages }} {{ api_suites }} examples {{ test_dirs }}

# `-` is just's ignore-failure prefix, and it is the point: ty, pyrefly and basedpyright run so a
# disagreement is seen, not so it can block. basedpyright reads `tools/basedpyright.json` here only.
# ty reads the examples and the tests too, as it did when it gated them; `toolbox ty` leaves out a
# typesafety suite it is not named.
[doc("Second opinions from ty, pyrefly and basedpyright (informational, never blocks)")]
types-info:
    -{{ tools }} toolbox ty -- {{ packages }} examples {{ test_dirs }}
    -{{ tools }} toolbox pyrefly check {{ packages }}
    -{{ tools }} toolbox basedpyright --project=tools/basedpyright.json -- {{ packages }}

# The repository's own checks import pytest and the standard library only. They walk `ast`, where
# reading an attribute off the wrong node is the likely mistake and a checker catches it. They run
# from inside `tools/`; see the module.
[doc("Check the repository's own checks with mypy (blocking)")]
types-checks:
    @just tools::types-checks

[doc("Check the benchmarks with mypy (blocking)")]
types-benchmarks: bench::types

[doc("Check types (the gates, the second opinions, the benchmarks and the checks)")]
types-all: types types-info types-benchmarks types-checks

# `--public-only` counts the symbols a caller reaches through the public modules and their
# re-exports, each reported where it is defined. The threshold is pyrefly's default, 100%. A public
# variable or attribute needs an annotation unless it is a literal constant, which is the typing
# guide's completeness rule; `Any` counts as annotated (no `--strict`).
[doc("Check that the public API is fully annotated (pyrefly coverage, gated at 100%)")]
type-coverage:
    {{ tools }} toolbox pyrefly coverage check --public-only {{ packages }}

# End-user experience: each distribution's `typesafety/`, beside its package, asserts (via `assert_type`) the
# types a downstream checker infers from the public API. basedpyright gates here (it's the Pylance experience);
# mypy gates the suites in `types`, as package code; ty and pyrefly run informational — they disagree on some
# spots, so they must not fail the recipe. `toolbox` hands each checker the synced env through its own
# interpreter option, never by installing it there, and holds it to the floor. It runs each once per
# distribution: every suite is a `typesafety` package, and one run over two of them is a duplicate module.
# The informational two first: just stops at the first failing line, and they must be seen
# whatever basedpyright says.
[doc("Type-check the public API surface as end users' checkers see it (basedpyright gates)")]
types-api:
    @{{ if api_suites == "" { error("types-api: no distribution has a typesafety/") } else { "true" } }}
    -{{ tools }} toolbox ty -- {{ api_suites }}
    -{{ tools }} toolbox pyrefly check -- {{ api_suites }}
    {{ tools }} toolbox basedpyright -- {{ api_suites }}

# --- Tests ---

# The noxfile defines every environment the tests run in. The sessions tagged `local` run here, one
# after another; the ones tagged `ci` are `.github/workflows/tests.yaml`'s matrix. Arguments pass to
# nox: `just tests --list`, `just tests --session "tests(python='3.14t')"`.
#
# nox runs the union of every selection it is given, so `--tags local` is added only when the
# arguments select nothing themselves.
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

# The quick loop while editing; `just tests` is the check: pytest in `.venv`, the environment `venv`
# builds, on its interpreter only, with every distribution installed, so an import of another
# distribution that a distribution does not declare passes here and fails in `tests`. It runs from the current directory, and pytest
# takes its configuration from the distribution the paths are in, so one call covers one
# distribution: `just test tests/x.py -k name -x`.
[no-cd]
[positional-arguments]
[doc("Run pytest in .venv from the current directory, on one distribution's tests (not part of the PR checks, `just gate`)")]
test *args:
    "{{ justfile_directory() }}/.venv/bin/python" -m pytest "$@"

# --- The gate ---

# One list, cheapest first, and CI names the same one: `.github/workflows/ci.yaml`'s `gate` job installs
# just and runs each dependency below as its own step, so GitHub reports which one failed, and
# the conventions runner holds the two lists to each other (CON700-CON704). Add a recipe here and
# that check fails until the workflow names it too; nothing here is spelled out a second time in YAML.
# `tests` is the exception: CI runs the tests as the matrix of `.github/workflows/tests.yaml`, which
# `ci.yaml` calls.
#
# Not here, on purpose: `lint` (pre-commit.ci runs the hooks from the same config). The type-checking
# rows are the target-level table in CONVENTIONS.md, one recipe per row that gates.
[doc("Everything a pull request must pass, the local test matrix last")]
gate: conventions imports deps slots types type-coverage types-api types-benchmarks types-checks tests


# --- Structure ---

# Per distribution, from its own directory: import-linter finds the package from where it runs.
[doc("Check import-graph architecture contracts (import-linter, config in pyproject.toml)")]
imports:
    #!/usr/bin/env bash
    set -euo pipefail
    for dist in {{ dists }}; do just tools::imports "$dist"; done

# Per distribution, from its own directory and `pyproject.toml`: an import of another distribution
# that it does not declare is a finding, though the shared `.venv` holds it.
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

[doc("Read-only check: every applicable commit hook, plus ty for Python paths. No paths = whole tree.")]
check *paths:
    @just tools::check {{ paths }}

[doc("Apply every fixer whose finding the PR checks (`just gate`) can only report. `check` and `gate` never write.")]
fix *paths:
    @just tools::fix {{ paths }}
