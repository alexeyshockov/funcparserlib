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

The module-level codemap, the diagrams and the invariants are [`ARCHITECTURE.md`](ARCHITECTURE.md); read it before
changing the package.

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

The invariants, each with what enforces it, are in [`ARCHITECTURE.md`](ARCHITECTURE.md#invariants). Challenge one
in discussion rather than routing around it.
