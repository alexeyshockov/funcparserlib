# 0010 — The floor is the oldest supported Python

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

ADR 0001 made 3.12 the floor so that the package, its tests and its docs could all use PEP 695: type
parameters declared on the class or function that uses them, and `type` aliases. 3.11 is still
supported by CPython until October 2027, and a floor of 3.12 turned its users away from 2.0.

What a reader copies from the docs and the examples is a separate question from what the package
needs to run. The docs teach the newest syntax, and that does not depend on the floor.

## Decision

- **`requires-python` starts from the oldest CPython still supported**, and the test matrix runs every
  version from there to the newest, plus the newest free-threaded build. Today that is 3.11. When
  3.11 reaches its end of life, the floor becomes 3.12 by this rule.
- **The package and its tests are written for the floor.** Below 3.12 that means module-level,
  private `TypeVar`s and `TypeAlias` annotations. Each class's type variables have the variance PEP
  695 inferred for it, as mypy reported it (a probe that assigns `C[Sub]` to `C[Base]` and back for
  each parameter): `ParsingResult`, `ParsingSuccess` and `Located` are covariant, and every parser
  class is invariant in every parameter. An alias that names a class defined later in its module
  comes after that class, so a public alias stays a subscriptable object at runtime.
- **The examples, the docs and the README teach the newest syntax**, whatever the floor. ruff's
  `per-file-target-version` gives `examples/**` the version they are written for. Every type
  checker reads it from there, and the root `conftest.py` leaves the examples' tests, the docs and
  the README out of a run on an older interpreter. One note on the getting-started page says how to
  write the same thing for the floor.

Rejected:

- Keeping 3.12 as the floor, as ADR 0001 decided. It turns away the users of a supported Python to
  keep one spelling in the package's own source, which users do not read.
- Writing the examples and the docs for the floor. They are what users copy, and they would teach
  a spelling that is on its way out.
- Each code sample in two tabs, one per syntax. The floor's tab has a fixed end date, every reader
  sees a tab most of them do not need, and 14 places in the docs would need a second variant kept in
  step.
- `typing_extensions`, as ADR 0001 rejected it. `Self`, `TypeGuard` and `dataclass_transform` are
  in the standard library from 3.11, so nothing calls for it.

## Consequences

- A method that uses a module-level `TypeVar` other than its class's is generic on its own and
  checks nothing. `tests/typesafety/` asserts what a user's checker infers for the combinators,
  under mypy, ty, pyrefly and basedpyright, so that mistake fails the gate.
- The docs' doctests, the README and the examples run on every version from the examples' version
  up, not on the floor.
- Moving the floor to 3.12 turns the package back to PEP 695 syntax, which ruff's UP040, UP046 and
  UP047 rewrite, and drops the note on the getting-started page.
- Users of 3.10 and older stay on `funcparserlib~=1.0`.
