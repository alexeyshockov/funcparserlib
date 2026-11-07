# 0008 — A combinator's form says what it does

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

A combinator is spelled in one of three ways: an operator, `p1 + p2`; a function, `many(p)`; or a method,
`p.when(pred)`. A new combinator needs one of them, and nothing said which.

Python decides part of this. It has binary and prefix operators, and no postfix one, so a grammar cannot write the
`*` and `?` of EBNF. Its precedence decides the rest: a method binds tighter than any operator, so
`a + b.named("x")` names only `b`, while a function shows what it covers, `many(a + b)`.

A grammar written for 1.0 keeps working, so `+`, `|`, `>>`, `-p`, `many()`, `maybe()`, `oneplus()`, `skip()`,
`named()` and `bind()` stay as they are.

## Decision

The form of a combinator says what it does:

- **An operator is the structure of a grammar**: sequence `+`, choice `|`, transformation `>>`, discarding `-p`.
- **A function changes how a parser is driven, or combines several**: `many()`, `maybe()`, `oneplus()`, `commit()`,
  `anything_but()`, `separated()`, `longest()`, `prefixed()`, `counted()`, `switch()`.
- **A method acts on what a finished parser gives, or on how it is named**: `named()`, `context()`, `when()`,
  `but_not()`, `validated()`, `map()`, `starmap()`, `bind()`.

`Parser.recover()` is the exception: it changes what a failure does, and is a method because it closes a whole rule, as
in `many(release.recover())`.

An operation has one spelling, with two exceptions. `skip()` stays beside `-p` for 1.0. `map()` stays beside `>>` as
the pair of `starmap()`, which has no operator, so a chain of methods reads alike with either.

Rejected:

- **`~p` for `anything_but(p)`.** `~` reads as a test that consumes nothing, as `!p` does in a PEG, and this parser
  consumes a token. No guide, recipe or example uses it.
- **Every combinator a method, `p.many()`.** The functions stay for 1.0, so each would get a second spelling.
- **Every combinator a function, `when(p, pred)`.** A rule then reads from the inside out, and its last step comes
  first.

## Consequences

- A new combinator takes its form from this list, and a new operator is a change to this record.
- A method needs parentheses around an expression of operators, as `(p1 + p2).named("x")`, and leaving them out applies
  it to the last operand. `.named()` and `.context()` on a whole rule are written that way.
- `compile()` does not depend on the form: it reads `_form`, which a function and a method set alike.
