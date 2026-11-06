# 0007 — A rejected value goes on without a value

- **Status:** Accepted
- **Date:** 2026-11-06

## Context

Input can match a grammar and still be wrong: a date the calendar does not have, an escape the format
does not define. A checker of such a format should report every problem in one run, and the value of a
wrong part does not exist.

`Parser.check(pred, message)` recorded an error and went on with the value. A grammar that used it had
to carry something for each part that failed. `examples/adr.py` kept `number 0` for a title it could not
read, a changelog parser turned a broken heading into a title made of the whole line, and `unescape()` in
`examples/kdl.py` said an invalid escape stands for nothing. The message was a fixed string, so an error
could not name the version out of order. `Parser.when()` tests and `>>` converts, so a conversion that can
fail ran twice: `is_date()` and `date.fromisoformat`, `valid_escapes()` and `unescape()`, a `Text` grammar
on the text of a token.

Other parser libraries mostly fail the parser (nom `verify`, winnow `try_map`, pyparsing `add_condition`,
megaparsec `customFailure`). chumsky `validate` and megaparsec `registerParseError` record an error and go
on. chumsky's function always returns a value, and its `parse` gives the output together with the errors.

## Decision

`Parser.validated(f, expected=None)` replaces `Parser.check()`. `f` returns the new value or raises
`Invalid(message)`. The input was read, so a rejected value does not fail the parser: the error is
recorded and parsing goes on, and `|` tries nothing else.

A rejected part has no value:

- No function after `>>` runs over it, at any level above it. `make_record` is not called for a record
  with a bad status, and the rest of the record is still read.
- What needs the value stops without an error of its own: `bind()`, `counted()`, `prefixed()`, `switch()`.
  `when()` and `but_not()` leave the rejection as it is.
- `State.rejected_since(s)` says so to a parsing function, which has to ask before it reads a value, as
  ADR 0006 makes those functions public.

`Invalid(message, keep=True)` is for a value that is real and only breaks a rule: parsing goes on with it.

`Parser.recover()` is the one place a broken part gets a value. It closes the rejections below it, and
what comes after it runs on `default`. `until` is optional, as a rejection needs no point to go on from;
with `until` it also takes a failure past the first token. A grammar that wants a value for a broken part
says what that value is, with a type of its own if it wants, so `int | Broken` admits what it holds.

A `NoParseError` or `ParseErrors` from `f` rejects the value too, so `f` can run another grammar on a value:
a text grammar on the text of a markdown-it token, a token grammar on the children of one. `expected` is
then the label of the error, which is on the position of the outer grammar, and the inner error is its
`__cause__`.

`when()` stays for what decides among alternatives: it asks whether this is the thing the grammar is looking
for, and `validated()` whether the thing is valid. A section heading with another name is another part of
the record, so `when()` fails there. Rejecting it gave a false second error at the end of the file.

The rejected mark lives on the recorded error, `Problem.rejected`, so `State` gains no field and
`problems` stays the one record of what went wrong. An error is written with the `describe` of `parse()`
only when the parse ends, so a value made during parsing holds no error text.

`>>` adds an identity test of `problems`, and walks them only when they changed. On a grammar of nothing
but `>>`, Python 3.12 on an arm64 laptop, 4 alternating runs, best of 5: interpreted 2.19 to 2.28 ms before,
2.17 to 2.27 ms after, so the cost is within the noise. Calling a function for that test instead cost 8 to
13% on the same grammar.

In generated code `validated()` is a `>>`: what `f` raises leaves that code for the interpreted parser, as
any exception does under ADR 0002, and `f` then runs twice, as a function after `>>` does.

Rejected:

- **Failing the parser, with `recover()` to collect.** A record with three bad fields reports the first,
  and a check on one token has nothing to resynchronize on.
- **A function that always returns a value, as in chumsky.** It suits a `parse` that returns the output with
  the errors. Here `parse()` raises, so the value only reaches the rules above it, and there it does harm:
  with `date.min`, `date.max` or a fixed date for a bad date, a rule on the order of dates reported a false
  error; with `None` it crashed; with `0` for a bad count, what follows was misread.
- **The name `validate`.** It reads as a validation with a list of errors for a result. This gives a parser
  of the value, so the name is `validated`.
- **`keep` as a second verb.** `Invalid` already carries the decision, and a rule is a function either way.
- **A keyword naming the exceptions to catch, `validated(date.fromisoformat, on=ValueError)`.** It would turn
  a `ValueError` from a bug in `f` into a report on the input. A function that catches what it means is the
  level of control this asks of a grammar.
- **A separate combinator for the boundary.** `recover()` already turns a broken part into `default`.
- **Errors as a value next to `parse()`.** It is `except` under another name.

## Consequences

- One run reports every rejected value, and a structural failure still ends the parse unless a `recover()`
  with `until` is around it: past that, the position is lost.
- A rule above a rejected part waits for the next run. Above a `recover()` it runs, and meets what `default`
  says, so the grammar decides.
- A conversion that fails and should try the next alternative is still `when()` and `>>`, and runs twice.
- A parsing function that reads a value without asking gets `None` over a rejected part.
- Partial output stays open: an entry point that returns the value with the errors would need no change to
  the core, as `recover()` marks where a tree stops being broken. A broken part knows its tokens by index,
  and a tool that wants lines needs `describe`. What would prove the design is an editor or a formatter.
