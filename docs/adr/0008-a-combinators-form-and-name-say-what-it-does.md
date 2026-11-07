# 0008 — A combinator's form and name say what it does

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

A combinator is spelled in one of three ways: an operator, `p1 + p2`; a function, `many(p)`; or a method,
`p.when(pred)`. A new combinator needs one of them, and nothing said which.

Python decides part of this. It has binary and prefix operators, and no postfix one, so a grammar cannot write the
`*` and `?` of EBNF. Its precedence decides the rest: a method binds tighter than any operator, so
`a + b.named("x")` names only `b`, while a function shows what it covers, `many(a + b)`.

Its name has the same problem: a call reads as a statement about what it gives, and `p.matches(text)` reads as a test
that answers yes or no, where it gives the matches.

A grammar written for 1.0 keeps working, so `+`, `|`, `>>`, `-p`, `many()`, `maybe()`, `oneplus()`, `skip()`,
`named()` and `bind()` stay as they are, as do `some()`, `a()`, `tok()`, `pure()`, `finished`, `forward_decl()`,
`define()`, `run()` and `parse()`.

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

The name of a call says which of three things it does:

- **A call that makes a parser reads as a phrase about the parser it makes.** A noun names what it parses: `many()`,
  `maybe()`, `longest()`, `regex()`, `lit()`, `switch()`, `pure()`, `context()`. An adjective or a participle says what
  was done to its argument: `finished`, `separated()`, `prefixed()`, `counted()`, `skipping()`, `tokenized()`,
  `named()`, `validated()`, `located()`. A preposition gives a condition: `when()`, `but_not()`, `anything_but()`.
- **A call that runs a parser, or changes one, is an imperative verb for what it does**: `parse()`, `parse_text()`,
  `parse_bytes()`, `try_parse()`, `scan()`, `scan_text()`, `replace_text()`, `compile()`, `run()`, `define()`.
- **A call that answers yes or no is a question**, `is_…()` or `has_…()`. A third-person verb such as `matches()` is
  left for one.

The exceptions are verbs that make a parser. `skip()` and `bind()` are from 1.0. `map()` and `starmap()` keep the names
they have in Python and in functional programming. `commit()` and `recover()` keep the terms of parsing for a cut and
for error recovery.

Rejected:

- **`matches()` for the parts of a text a grammar reads.** It reads as a test. `scan()` says what the call does, as
  `scan_string()` does in pyparsing.
- **`find_all()`.** It reads as a list, as `re.findall()` gives one, where the matches come lazily.
- **`finditer()`.** The name `re` gives an iterator of matches, as a compound no other name here is.
- **`committed()` and `recovered()`.** A rule without exceptions, at the price of names that read as the state of a
  parser rather than what it does.
- **`~p` for `anything_but(p)`.** `~` reads as a test that consumes nothing, as `!p` does in a PEG, and this parser
  consumes a token. No guide, recipe or example uses it.
- **Every combinator a method, `p.many()`.** The functions stay for 1.0, so each would get a second spelling.
- **Every combinator a function, `when(p, pred)`.** A rule then reads from the inside out, and its last step comes
  first.

## Consequences

- A new combinator takes its form and its name from this record, and a new operator is a change to it.
- A method needs parentheses around an expression of operators, as `(p1 + p2).named("x")`, and leaving them out applies
  it to the last operand. `.named()` and `.context()` on a whole rule are written that way.
- `compile()` does not depend on the form: it reads `_form`, which a function and a method set alike.
