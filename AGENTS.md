# funcparserlib

A recursive descent parsing library for Python 3.12+ based on functional combinators: a grammar is
written as Python expressions over parsers (`p1 + p2`, `p1 | p2`, `p >> f`, `many(p)`), and a
regexp-based lexer turns text into the tokens they read. The flat-layout package is
`funcparserlib/`; tests are `tests/`.

The general rules (tooling, types, source, exception messages, specs, commits) are
`CONVENTIONS.md`; `CONTRIBUTING.md` is the contributor path. This file holds what is funcparserlib's
own. `AGENTS.local.md`, if present, adds local instructions: read it before starting work. This
file stays authoritative for repository facts and invariants, and nothing tracked depends on the
local file.

@AGENTS.local.md

## Layout

```text
funcparserlib/parser.py   Parser, the primitives (tok, a, some, forward_decl, finished), the combinators, the results
funcparserlib/lexer.py    make_tokenizer, TokenSpec, Token, LexerError
funcparserlib/util.py     pretty_tree, expand_tuple_args
funcparserlib/_commons.py the generated helpers: valueclass, refclass, with_hint
tests/test_*.py           unit tests; tests/dot.py and tests/json.py are example grammars they drive
tests/typesafety/         what a user's checker infers; pytest does not collect it
tools/conventions/        the shared conventions runner, rendered; never edited here (just conventions)
docs/                     the site (zensical.toml); every `>>>` session in it runs as a doctest
docs/adr/                 the decision record
examples/                 runnable programs using public APIs only; tests/test_examples.py runs each
```

## Commands

```shell
just doctor              # host tools and the commit hooks
just venv                # every dependency group
just check [FILES]       # read-only: every applicable hook, + ty for Python paths
just fix [FILES]         # apply the fixers
just types               # mypy over the package, the correctness gate
just gate                # everything a pull request must pass, the local test sessions last
just tests               # the noxfile's local sessions: the 3.12 floor and free-threaded 3.14t
just docs-build          # the site into site/
uv run pytest tests/test_parsing.py::ParsingTest::test_many_backtracking -v
```

## Invariants

Rule, then what enforces it. Challenge one in discussion rather than routing around it.

- **No runtime dependencies.** `dependencies = []` in `pyproject.toml`; `just deps`.
- **A grammar written for 1.0 parses the same and fails with the same message.** The doctests in
  `docs/` and in the docstrings (pytest's `--doctest-glob` and `--doctest-modules`),
  `tests/test_parsing.py`.
- **The combinators' types are the product.** What `p1 + p2`, `-p`, `p1 | p2`, `p >> f`, `many()`,
  `maybe()`, `forward_decl()` infer for a user is asserted in `tests/typesafety/` (`just types-api`,
  ADR 0001).
- **Values are `@valueclass`**: `Token`, `TokenSpec`, `State` and the parsing results are frozen to
  type checkers and assign their fields directly at runtime. The conventions runner, CON513.
- **`Parser` is the one exported class open to subclassing**; everything else exported is `@final`.
  `[tool.conventions.subclassable]`, CON200.
- **The docs pages are tests.** A `>>>` session in `docs/` runs under pytest. `examples/` is
  type-checked and run (`just types-examples`, `tests/test_examples.py`).
