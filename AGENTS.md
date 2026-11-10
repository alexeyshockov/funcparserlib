# funcparserlib

A recursive descent parsing library for Python 3.11+ based on functional combinators: a grammar is
written as Python expressions over parsers (`p1 + p2`, `p1 | p2`, `p >> f`, `many(p)`), and a
regex-based lexer turns text into the tokens they read. The flat-layout package is
`funcparserlib/`; tests are `tests/`.

How code is written is `CONVENTIONS.md`; the process is `CONTRIBUTING.md`; the codemap and the
architectural invariants are `ARCHITECTURE.md`. This file holds what is funcparserlib's own.
`AGENTS.local.md`, if present, adds local instructions: read it before starting work. Nothing
tracked depends on it.

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

`just --list` shows every recipe; the shared ones are in `CONTRIBUTING.md`. This project's own:

```shell
just mutants [FILTER]    # mutation testing, over an hour, on demand and in no gate
uv run pytest tests/test_parsing.py::ParsingTest::test_many_backtracking -v
```

The development environment is Python 3.12+ (the examples and the docs are written for it; the package
runs from 3.11).

## Working rules

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
- **A 1.0 grammar keeps working.** The combinators, their results and their error messages are the
  compatibility surface; `CHANGELOG.md` names every break.

## Tests

`tests/test_parsing.py` and `tests/test_combinators.py` cover the combinators, `tests/test_lexer.py` the
tokenizer, and `tests/test_errors.py` error reporting: what an error expects, labels, `commit()`,
`Parser.context()`, `Parser.validated()`, `describe=` and `render()`. `tests/test_json.py` and
`tests/test_dot.py` drive the example grammars in `tests/json.py` and `tests/dot.py`, and
`tests/examples/test_examples.py` runs `examples/`.

Compiled and interpreted parsers have to agree, and three things check that. The property tests in
`tests/test_compile.py`, `tests/test_text.py` and `tests/test_binary.py` draw random grammars, and a test
fails if a form of generated code is out of their reach, so a new combinator goes into those generators.
`tests/shadow.py` runs every parse of the suite on the compiled parser as well: mark a test
`interpreted_only` if it counts what a parse calls. `tests/test_exhaustive.py` walks every small grammar,
for the shapes random grammars rarely draw. A failure Hypothesis finds goes into a test of its own, as
`.hypothesis` is local.

Properties go in the same modules, written with [Hypothesis](https://hypothesis.readthedocs.io/): the
tokens reassemble the text and know their positions, a sequence keeps what it does not skip, the error
names the first token that differs, and the example JSON grammar reads what `json.dumps` writes as
`json.loads` does. A failing property prints the smallest input it found; keep it as an example-based test
when the fix lands.

## Maintainer setup

Releases are made from the upstream repository, `vlasovskikh/funcparserlib`, by its maintainer. Once per
repository, before the first release from this tooling:

- On PyPI, a trusted publisher for `funcparserlib`: this repository, workflow `pypi-publish.yaml`,
  environment `pypi`.
- On GitHub, the `pypi` environment (Settings → Environments), optionally restricted to `master` with
  required reviewers.
- [pre-commit.ci](https://pre-commit.ci/) enabled for the repository: CI runs the commit hooks there, not
  in `ci.yaml`.
- The version rules read `v*` tags only, so the first release made this way is the first one they see.
