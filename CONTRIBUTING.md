# Contributing to funcparserlib

`AGENTS.md` is the repository guide: layout, design invariants, and the verification ladder. The general
rules — the tooling tiers, the type gates, the source conventions, commits, pull requests and the
release flow — are in `CONVENTIONS.md`. This file is the contributor path for what is funcparserlib's own.

## Development setup

Python 3.12+, [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Then:

```shell
just doctor              # install host tools and the commit hooks
just venv                # sync every dependency group
```

Day to day:

```shell
just check path/to/file.py   # read-only verdict for one file (never rewrites it)
just types                   # mypy over the package — the correctness gate
just gate                    # everything a pull request must pass (CI's gate job runs its static part)
just tests                   # the suite on the 3.12 floor and on free-threaded 3.15t
just docs-serve              # the site with live reload
just fix                     # the only recipe that rewrites files
just --list                  # every recipe, always current
```

Run one test with `uv run pytest tests/test_parsing.py -k many -v`.

## Code style and conventions

- **The types are the product.** Preserve what a user's checker infers for the combinators, not only what
  they do at runtime; `tests/typesafety/` is where a change in inference shows.
- **Docs are tests.** A `>>>` session in a docs page or a docstring runs under pytest; when output
  changes, the page changes with it.
- **A 1.0 grammar keeps working.** The combinators, their results and their error messages are the
  compatibility surface; `CHANGELOG.md` names every break.

## Tests

`tests/test_parsing.py` and `tests/test_combinators.py` cover the combinators, `tests/test_lexer.py` the
tokenizer, and `tests/test_errors.py` error reporting: what an error expects, labels, `commit()`, `Parser.context()`,
`Parser.check()`, `describe=` and `render()`. `tests/test_json.py` and `tests/test_dot.py` drive the example grammars
in `tests/json.py` and `tests/dot.py`, and `tests/test_examples.py` runs `examples/`.

Properties go in the same modules, written with [Hypothesis](https://hypothesis.readthedocs.io/): the tokens
reassemble the text and know their positions, a sequence keeps what it does not skip, the error names the first token
that differs, and the example JSON grammar reads what `json.dumps` writes as `json.loads` does. A failing property
prints the smallest input it found; keep it as an example-based test when the fix lands.

## Maintainer setup

Releases are made from the upstream repository, `vlasovskikh/funcparserlib`, by its maintainer. Once per
repository, before the first release from this tooling:

- On PyPI, a trusted publisher for `funcparserlib`: this repository, workflow `pypi-publish.yaml`,
  environment `pypi`.
- On GitHub, the `pypi` environment (Settings → Environments), optionally restricted to `master` with
  required reviewers.
- [pre-commit.ci](https://pre-commit.ci/) enabled for the repository: CI runs the commit hooks there, not
  in `ci.yaml`.
- Releases are tagged `vX.Y.Z` (`just release X.Y.Z`). The version rules read `v*` tags only, so the
  first release made this way is the first one they see.

## Commits

Conventional Commits with a scope; the rules are in `CONVENTIONS.md`.
