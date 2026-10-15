# 0001 — Python 3.12 is the floor

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

funcparserlib's value to a user is a grammar that reads like the grammar and is type-checked like the
code around it: `p1 + p2` is a parser of a tuple of both values, `-p` drops its value, `p >> f` is a
parser of what `f` returns. The typing of those operators carried most of the code's complexity on
older Pythons: `TypeVar`s declared away from the classes that use them, `typing_extensions` for
`Self`, and version gates around `dataclass(slots=True)`. Python 3.11 and older are near or past the
end of their support.

## Decision

`requires-python` is `>=3.12`. The type parameters use PEP 695 syntax, aliases are `type`
statements, and value classes are slotted and frozen for type checkers. The test matrix is every
Python from 3.12 to the newest, plus the newest free-threaded build. Users of older Pythons stay on
`funcparserlib~=1.0`, which 2.0 does not replace for them.

Rejected: keeping 3.9 as the floor through `typing_extensions`. It keeps the old spelling of every
generic, and the spelling is what made the methods of `Parser` easy to get wrong: a method using a
module-level `TypeVar` instead of the class's own is generic on its own and checks nothing.

## Consequences

- The package has no runtime dependencies.
- `tests/typesafety/` asserts what a user's checker infers for the combinators, under mypy, ty,
  pyrefly and basedpyright, so a typing regression fails the gate.
- Typing tuple parsers of any length with `TypeVarTuple` (native since 3.11) is open to a later
  change; today `Tuple2Parser` to `Tuple5Parser` cover sequences up to five values.
