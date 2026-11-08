# 0001 — The floor is the oldest supported Python

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

funcparserlib's value to a user is a grammar that reads like the grammar and is type-checked like the
code around it: `p1 + p2` is a parser of a tuple of both values, `-p` drops its value, `p >> f` is a
parser of what `f` returns. The typing of those operators carried most of the code's complexity on
older Pythons: `TypeVar`s declared away from the classes that use them, `typing_extensions` for
`Self`, and version gates around `dataclass(slots=True)`. PEP 695, from 3.12, declares a type
parameter on the class or function that uses it.

What a reader copies from the docs and the examples is a separate question from what the package
needs to run. The docs teach the newest syntax, and that does not depend on the floor.

## Decision

- **`requires-python` starts from the oldest CPython still supported**, and the test matrix runs every
  version from there to the newest, plus the newest free-threaded build. Today that is 3.11,
  supported until October 2027; then the floor becomes 3.12 by this rule. Users of older Pythons
  stay on `funcparserlib~=1.0`, which 2.0 does not replace for them.
- **The package and its tests are written for the floor.** Below 3.12 that means module-level
  `TypeVar`s and `TypeAlias` annotations. A type variable has the name its type parameter had and
  will have again, with no underscore and no variance suffix, so the API reference reads the same
  on either side of 3.12; only two variables that would share a name keep apart, as `B` and
  `B_co`. Each class's type variables have the variance PEP 695 infers for it, as mypy reports
  it (a probe that assigns `C[Sub]` to `C[Base]` and back for each parameter): `ParsingResult`,
  `ParsingSuccess` and `Located` are covariant. The parser classes are covariant in their value
  by their structure: `run` and `_fn` are `Final` to checkers, and no member takes the value as
  input, which moves the private helpers that did off `Parser`. Their input is invariant, as
  `parse()` passes it to `describe` and each combinator method takes a `Parser[A, C]`.
  `ForwardDeclParser` stays invariant: `define()` takes a `Parser[A, B]`. An invariant value
  refused a recursive union, `forward_decl()` of `Json` defined from alternatives that each give
  one member. An alias that names a class defined later in its module comes after that class, so a
  public alias stays a subscriptable object at runtime. Value classes are slotted and frozen for
  type checkers.
- **The examples, the docs and the README teach the newest syntax**, whatever the floor. ruff's
  `per-file-target-version` gives `examples/**` the version they are written for. Every type
  checker reads it from there, and the root `conftest.py` leaves the examples' tests, the docs and
  the README out of a run on an older interpreter. One note on the getting-started page says how to
  write the same thing for the floor.

Rejected:

- A fixed floor of 3.12. It turns away the users of a supported Python to keep one spelling in the
  package's own source, which users do not read.
- Keeping 3.9 as the floor through `typing_extensions`. It keeps the old spelling of every generic
  for years, not until the next end of life, and needs a runtime dependency for `Self`.
- Writing the examples and the docs for the floor. They are what users copy, and they would teach
  a spelling that is on its way out.
- Each code sample in two tabs, one per syntax. The floor's tab has a fixed end date, every reader
  sees a tab most of them do not need, and 14 places in the docs would need a second variant kept in
  step.

## Consequences

- The package has no runtime dependencies: `Self`, `TypeGuard` and `dataclass_transform` are in the
  standard library from 3.11.
- A method that uses a module-level `TypeVar` other than its class's is generic on its own and
  checks nothing. mypy over the package fails it where the method's body uses the class's type, and
  `typing_tests/`, which asserts what a user's checker infers for the combinators under mypy,
  ty, pyrefly and basedpyright, fails it in an overload's signature, which has no body.
- The docs' doctests, the README and the examples run on every version from the examples' version
  up, not on the floor.
- Moving the floor to 3.12 turns the package back to PEP 695 syntax, which ruff's UP040, UP046 and
  UP047 rewrite, and drops the note on the getting-started page.
- PEP 695 syntax cannot declare variance, and infers for the parser classes the variance they
  declare, so they take PEP 695 syntax at that move as every other class does.
- Typing tuple parsers of any length with `TypeVarTuple` (native since 3.11) is open to a later
  change; today `Tuple2Parser` to `Tuple5Parser` cover sequences up to five values.
