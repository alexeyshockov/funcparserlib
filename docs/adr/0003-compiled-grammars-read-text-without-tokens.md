# 0003 — Compiled grammars read text without tokens

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

With a grammar compiled as in ADR 0002, the tokenizer was the larger cost. On the 21 MB KDL
benchmark it took 4.39 s of 7.60 s, and the tokens held most of the 1.67 GB the program peaked at.
On a 2 MB slice the regex engine took 0.10 s of 0.34 s: the rest went to `Token` objects, their
positions, and the garbage collections those allocations set off.

Generated code reads the type and the value of a token, and nothing else, as long as the grammar
reads tokens through `tok()`. A token's position is for errors, and errors come from the interpreted
parser.

## Decision

`Parser.compile(tokenizer)` generates a second set of functions, over two lists instead of tokens: the
values and the types. `CompiledParser.parse_text(text)` fills the lists from the tokenizer's joined
regex and runs those functions. Whenever that does not succeed, it runs the tokenizer and the
interpreted parser on the text, and what they return or raise is the result, as in ADR 0002.

A grammar that reads a token any other way, with `some()`, `a()` or a parsing function, needs `Token`
objects, and so do specs that do not join into one regex. `parse_text()` then tokenizes and calls
`CompiledParser.parse()`.

Compiled without a tokenizer, a grammar that reads no `tok()` gets those functions too, reading what
`parse_text()` or `parse_bytes()` is given. A grammar of `some()`, `a()` and `finished` cannot say what
it reads, so it is the call that says.

The types to leave out, usually whitespace and comments, have to be known without a wrapper around the
tokenizer, so they are an argument of the tokenizer: `make_tokenizer(specs, skip=...)`. It returns a
`Tokenizer`, the object `compile()` reads the regex from.

Measured on Python 3.12 on an arm64 laptop, the 21 MB KDL benchmark, wall clock, one run:

| Path | Total | Peak memory |
| --- | ---: | ---: |
| Tokens, then `CompiledParser.parse()` | 7.60 s | 1.67 GB |
| `CompiledParser.parse_text()` | 4.82 s | 0.60 GB |

Rejected:

- **`skip=` on `compile()`.** A program would list the same types in its tokenizer wrapper and in the
  call, and the wrapper every tokenizer had would stay.
- **`parse()` accepting text.** A grammar over characters takes a `str` as its tokens, so a `str` would
  mean two things depending on how the grammar was compiled.

## Consequences

- A failed parse runs the tokenizer on top of what ADR 0002 lists. `>>` functions run in two passes, as
  there.
- A grammar compiled with a tokenizer generates its functions twice, which takes milliseconds.
- `make_tokenizer()` returns a `Tokenizer` where it returned a function. It is still called with text
  and still gives tokens one at a time.
- Tokenizing stays a `str` matter: `TokenSpec.pattern` is a `str`, and positions count `"\n"`.
