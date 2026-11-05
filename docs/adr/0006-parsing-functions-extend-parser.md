# 0006 — Parsing functions extend Parser

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

`Parser` is three things: the type in every annotation, a concrete wrapper of a parsing function, and a
base class. The labels on it disagreed: its docstring said extensible by design, while `Parser.__init__`
and `Parser.run` were marked internal, so no supported way to extend it existed.

`run` is a field, and `Parser.named()`, `ForwardDeclParser.define()` and `debug` replace it on the
instance. A subclass's `run` method is ignored when the subclass calls `super().__init__(f)`, and makes
`named()` raise `AttributeError` when it does not.

Most of the subclasses are types for checkers: `x + x` and `-x` are plain `Parser` objects at runtime.
Only `EndOfInput` and `ForwardDeclParser` have instances.

## Decision

`Parser` is concrete, and a parsing function is what a grammar adds to the library. `Parser(f)`, the
`parser()` decorator, `ParserFn` and `Parser.run` are public, so a function may call other parsers,
reading `p.run` at each call. A subclass may add to a parser; overriding `run` is not supported.

`run` stays a field: the call of a parser is the call of its function. Making it a method over a stored
function adds a frame to each call. Best of 6 alternating processes on an arm64 laptop, against the
field, on a run-to-run spread of up to 8.5%:

| | 3.12 | 3.14 | 3.15t |
| --- | --- | --- | --- |
| JSON grammar, parse | +62% | +11% | +9% |
| DOT grammar, parse | +9% | +9% | +10% |

The 3.12 JSON figure is an outlier with no known cause.

Backtracking needs no new name. `State`'s constructor and fields are public, and a choice written with
`State(s.pos, st.max, st.expected, s.context, s.problems)` matched `|` on all of 1,023,000 parses of
random grammars, and on 511,500 compiled. `st.reset_pos(s.pos)` differed on 443 of them: it keeps the
`Parser.check()` errors of the failed branch.

Rejected:

- **An abstract `Parser` with a private concrete class.** `Parser(f)` stops working, which the 1.0
  compatibility of a parsing function relies on, unless `__new__` redirects it and undoes the
  abstraction. What it improves is writing subclasses, and nothing here needs them.
- **A class per combinator, each with a `run` method.** A toy model of the same grammar ran within 3% of
  closures, so speed does not rule it out. It rewrites every combinator and what `_compile.py` reads.
- **A named `State` method for backtracking.** No grammar here needs it, and adding one later breaks
  nothing.

## Consequences

- A function that keeps `p.run` instead of reading it at each call misses what `named()` and `define()`
  set later.
- A `ParserFn` and `Parser.run` keep their signature, as ADR 0002 already requires of `ParserFn`.
- `.compile()` cannot look into a parsing function, as ADR 0003 says, so a grammar of them runs without
  the fast text path.
