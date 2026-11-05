# 0008 — A composed name is bounded

- **Status:** Accepted
- **Date:** 2026-11-06

## Context

The library names a parser it builds after its parts: `p1 + p2` is `(p1, p2)`, and `oneplus(p)` is
`(p, { p })`. The name is built with the parser, so a parser that appears twice doubles it at each level.
Nested `oneplus(g + tok("b"))` gave 4,846 characters at 8 levels, 1,245,166 at 16 and 4,980,718 at 18, and
`g + g` the same. A grammar 33 levels deep extrapolates to about 160 GB, in construction, before anything
parses or compiles.

`p | p` had a second growth: flattening the alternatives visited a repeated parser once per path to it, so
each level doubled the time, 5.2 ms at 15 levels.

A composed name reaches `Parser.name`, the parsing log and the `ValueError` of `many()`. `_label_of` keeps
it out of what an error expects.

## Decision

A name the library composes is cut at 200 characters, with `...`, where it is composed. A name given with
`Parser.named()` is kept whole, and so is a parser's name passed on unchanged. `Parser.but_not()` composes
a label, so an error can show that one cut.

`_alternatives` lists a parser once, by identity, in the order first seen.

At 40 levels, every combinator tried builds in under 1 ms, compiles in about 10 ms and stays near 25 MB.

Rejected:

- **Names computed when read.** Building costs nothing, but reading a deep name is as exponential as
  building it was, and `name` is an assignable public field.
- **Fixed names such as `sequence`.** No growth, but `Parser.name`, the log and the `ValueError` of `many()`
  change for every grammar, where a bound changes only names nobody can read whole.

## Consequences

- The tests and doctests pass unedited at limits from 40 to 1,000: nothing asserts a composed name longer
  than 40 characters.
- A repeated alternative runs once where `|` dispatches on the next token, so a `>>` function inside it can
  run fewer times than it did. The value and the error do not change.
- Tests of this stay at 20 levels: there, a regression costs tens of megabytes, where 33 would exhaust
  memory.
