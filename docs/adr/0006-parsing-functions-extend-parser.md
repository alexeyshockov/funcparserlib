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

`run` stays a field: the call of a parser is the call of its function. It is `Final` to type checkers,
so a parser is covariant in its value (ADR 0001); the library assigns it when it builds a parser, and
in `named()` and `define()`. Making it a method over a stored function adds a frame to each call. Best of 6 alternating
processes on an arm64 laptop, against the field, with the range of the runs where there were several:

| | 3.12 (3 runs) | 3.14 (2 runs) | 3.15t (2 runs) |
| --- | --- | --- | --- |
| JSON grammar, parse | +61% to +64% | +11% | +9% to +10% |
| DOT grammar, parse | +7% to +12% | +9% to +12% | +10% to +16% |

The spread within a run reached 8.5%, so the 3.14 and 3.15t figures are a cost of about a tenth rather
than exact values. The 3.12 JSON figure repeats in every run; its cause is not known. What would change
this decision is a method that costs nothing on the supported interpreters.

Backtracking needs no new name. `State` is a dataclass with public fields, and a choice that goes on from
`dataclasses.replace(s, max=st.max, expected=st.expected)` matched `|` on all of 1,023,000 parses of
random grammars, and on 511,500 compiled. It carries any field `State` gains. `st.reset_pos(s.pos)`
differed on 443 of them: it keeps the errors the failed branch recorded.

A parsing function returns a result, and a failure is one. `Parser.run` gives a `ParsingSuccess` with the
value and the state after it, or a `ParsingFailure` with the state where parsing stopped, its message and
whether it is committed. A function holds a failure, reads its `state.max`, `state.expected` and
`state.problems`, and decides: return it, try another parser from that state as above, or record it and go
on. `NoParseError` is raised at the end of a parse, by `Parser.parse()` and its kin, and when a function
reads the `value` of a failure or unpacks it.

A result has the operations of the parser that made it, so a function composes results as a grammar
composes parsers:

| Result | Parser |
| --- | --- |
| `res.map(f)` | `p >> f` |
| `res.starmap(f)` | `p.starmap(f)` |
| `res.bind(f)` | `p.bind(f)` |
| `res1 + res2` | `p1 + p2` |

Each gives a failure back as it is, so the first failure is what a chain of them returns. `+` combines the
values as `+` of parsers does: a tuple that grows to the right, with an ignored value dropped. The
combinators do not call these methods: `Parser.__add__` and the others repeat the logic inline and test a
result by `res.__class__ is ParsingFailure`, which spares a call per parser. The methods are for parsing
functions.

Rejected:

- **An abstract `Parser` with a private concrete class.** `Parser(f)` stops working, which the 1.0
  compatibility of a parsing function relies on, unless `__new__` redirects it and undoes the
  abstraction. What it improves is writing subclasses, and nothing here needs them.
- **A class per combinator, each with a `run` method.** A toy model of the same grammar ran within 3% of
  closures, so speed does not rule it out. It rewrites every combinator and what `_compile.py` reads.
- **A named `State` method for backtracking.** No grammar here needs it, and adding one later leaves
  code written with `dataclasses.replace()` working.

## Consequences

- A function that keeps `p.run` instead of reading it at each call misses what `named()` and `define()`
  set later. So do `Parser(p)` and `define(p)`, which keep it to save a frame on each call; only a
  `forward_decl()` is read at each call.
- A `ParserFn` and `Parser.run` keep their signature, as ADR 0002 already requires of `ParserFn`.
- `.compile()` cannot look into a parsing function, as ADR 0003 says, so a grammar of them runs without
  the fast text path.
