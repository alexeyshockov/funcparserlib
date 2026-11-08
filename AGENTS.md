# funcparserlib

A recursive descent parsing library for Python 3.11+ based on functional combinators: a grammar is
written as Python expressions over parsers (`p1 + p2`, `p1 | p2`, `p >> f`, `many(p)`), and a
regex-based lexer turns text into the tokens they read. The flat-layout package is
`funcparserlib/`; tests are `tests/`.

The general rules (tooling, types, source, exception messages, specs, commits) are
`CONVENTIONS.md`; `CONTRIBUTING.md` is the contributor path. This file holds what is funcparserlib's
own. `AGENTS.local.md`, if present, adds local instructions: read it before starting work. This
file stays authoritative for repository facts and invariants, and nothing tracked depends on the
local file.

@AGENTS.local.md

## Layout

The module-level codemap, the diagrams and the architectural invariants are [`ARCHITECTURE.md`](ARCHITECTURE.md);
read it before changing the package.

```text
funcparserlib/            the package, flat layout
tests/                    unit tests, example grammars, and the `shadow` pytest plugin
typing_tests/             what a user's checker infers, at the package's level: all four checkers gate it
conftest.py               leaves the examples' tests, docs/ and README.md out below the examples' Python
tools/conventions/        the shared conventions runner, rendered; never edited here (just conventions)
docs/                     the site (zensical.toml); every `>>>` session in it runs as a doctest
docs/adr/                 the decision record
examples/                 runnable programs using public APIs only; tests/examples/test_examples.py runs each
benchmarks/               pytest-benchmark micro-benchmarks, bench_*.py; results/ is local and untracked
```

## Commands

```shell
just doctor              # host tools and the commit hooks
just venv                # every dependency group
just check [FILES]       # read-only: every applicable hook, + mypy for Python paths
just fix [FILES]         # apply the fixers
just types               # mypy over the package, examples, tests, benchmarks: the correctness gate
just gate                # everything a pull request must pass, the local test sessions last
just tests               # the noxfile's local sessions: free-threaded 3.15t, shadow on 3.11, exhaustive
just docs-build          # the site into site/
just bench::micro        # the micro-benchmarks, saved under benchmarks/results/
just bench::compare      # the last two saved runs side by side
just mutants [FILTER]    # mutation testing, over an hour, on demand and in no gate
uv run pytest tests/test_parsing.py::ParsingTest::test_many_backtracking -v
```

## Invariants

The architectural invariants (boundaries, dependency direction, what a part never knows) are in
[`ARCHITECTURE.md`](ARCHITECTURE.md#invariants); these are the rules for working in this repository.

Rule, then what enforces it. Challenge one in discussion rather than routing around it.

- **A grammar written for 1.0 parses to the same value, and its errors say what each doc example
  says.** The doctests in `docs/` and in the docstrings (pytest's `--doctest-glob` and
  `--doctest-modules`), which compare error messages too, `tests/test_parsing.py`.
- **The combinators' types are the product.** What `p1 + p2`, `-p`, `p1 | p2`, `p >> f`, `many()`,
  `maybe()`, `forward_decl()` infer for a user is asserted in `typing_tests/` (`just types-api`,
  ADR 0001).
- **An error names everything expected where parsing stopped**, a label instead of a parser's parts, and an error
  from a nested `parse()` unchanged. `tests/test_errors.py`.
- **Values are `@valueclass`**: `Token`, `TokenSpec`, `State` and the parsing results are frozen to
  type checkers and assign their fields directly at runtime. The conventions runner, CON513.
- **`Parser` is the one exported class open to subclassing**; everything else exported is `@final`.
  `[tool.conventions.subclassable]`, CON200.
- **The docs pages are tests.** A `>>>` session in `docs/` or `README.md` runs under pytest. `examples/` is
  type-checked and run (`just types`, `tests/examples/test_examples.py`).
- **The package runs on the floor; what it teaches uses the newest syntax.** The package and `tests/` are
  written for `requires-python`, the oldest supported CPython. `examples/`, `docs/` and `README.md` are written
  for the version ruff's `per-file-target-version` gives `examples/**`, which the type checkers read and below
  which `conftest.py` leaves them out of the tests. The floor's nox sessions, ADR 0001.
