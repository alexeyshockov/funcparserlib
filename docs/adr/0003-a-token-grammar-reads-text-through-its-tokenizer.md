# 0003 — A token grammar reads text through its tokenizer

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

A token grammar is a `Parser[Token, B]` over a sequence, and its tokenizer is a separate object: a program pairs
them, `grammar.parse(list(tokenize(text)))`. Whatever faces the text, as parsing a text, finding and rewriting its
parts, character offsets and errors with a line and a column, needs the two together.

With a grammar compiled as in ADR 0002, the tokenizer was the larger cost. On the 21 MB KDL benchmark it took 4.39 s
of 7.60 s, and the tokens held most of the 1.67 GB the program peaked at. On a 2 MB slice the regex engine took
0.10 s of 0.34 s: the rest went to `Token` objects, their positions, and the garbage collections those allocations set
off.

Generated code reads the type and the value of a token, and nothing else, as long as the grammar reads tokens through
`tok()`. A token's position is for errors, and errors come from the interpreted parser.

What else holds the design in place:

- A grammar written for 1.0 parses to the same value, and `parse(tokens)` takes tokens of any tokenizer, including
  those of other libraries.
- A token grammar's language is its global tokens (ADR 0004): read from the start of a text, it reads exactly
  `list(tokenize(text))`.
- `Token` has a line and a column and no offset (ADR 0009).
- `Parser[Token, B]` and `Parser[str, B]` are different types, and checkers keep them apart (ADR 0001).

## Decision

`tokenized(grammar, tokenizer)` gives a `Parser[str, B]`: the token grammar, reading text through the tokenizer. It is
a function, as ADR 0008 has it for what changes how a parser is driven, and it is the one way to join a grammar and
its tokenizer.

- **It is part of a grammar.** It combines with `lit()`, `regex()` and the parsers of `skipping()` by `+`, `|` and
  `many()`, so a grammar without a tokenizer can hold a language written with `tok()`. Where it is reached, it
  tokenizes from that offset as the token grammar reads, and it ends where the token grammar stops. The tokens of the
  tokenizer's `skip=` after its last token are its skip: the offset after them is where parsing goes on, and a span
  ends before them, as after `skipping()`. Where such a token is where it is reached, it fails, as what comes before
  it skips, so neither a match nor a span starts with a skipped token; `parse_text()` of it reads them at the start of
  the text.
- **From the start of a text it reads the global tokens.** The tokenizer reads in order from where it starts, so from
  offset 0 the tokens are those of `list(tokenize(text))`, and a whole document keeps the definition of ADR 0004.
- **Text no spec reads ends its tokens** with one more token, which no grammar reads: the token grammar fails there, and
  the error is a mistake among the tokens before it or what the grammar expected in place of that text (ADR 0004).
- **Spans are character offsets.** `located()` inside it gives offsets into the text, read from the offsets the tokens
  were read at, which the bound parser keeps beside its tokens. `parse(tokens)` of the token grammar gives token
  indexes.
- **Its features of text are those of any `Parser[str, B]`.** `parse_text()`, `matches()`, `replace()` and errors with
  a line and a column work as for a grammar without a tokenizer. `matches()` tries each offset, so text no spec reads
  never matches, and a match may start inside what the global tokens make one word.
- **`Parser.compile()` takes no tokenizer.** Compiled, a bound grammar at the root of what is compiled reads a text
  through generated functions over two lists instead of tokens, the values and the types, filled from the tokenizer's
  joined regex. Whenever that does not succeed, the tokenizer and the interpreted parser run on the text, and what they
  return or raise is the result, as in ADR 0002. A bound part inside another grammar is called through `run`, and
  makes `Token` objects for what it reads.
- **A grammar that reads a token any other way**, with `some()`, `a()` or a parsing function, needs `Token` objects, and
  so do specs that do not join into one regex: the bound grammar then tokenizes and runs the compiled pass of ADR 0002
  over the tokens.
- **A grammar that reads no `tok()`**, compiled, gets generated functions that read what `parse_text()` or
  `parse_bytes()` is given. A grammar of `some()`, `a()` and `finished` cannot say what it reads, so it is the call
  that says.
- **The token types to leave out**, usually whitespace and comments, are an argument of the tokenizer,
  `make_tokenizer(specs, skip=...)`, so that they are known without a wrapper around it. It returns a `Tokenizer`, the
  object `tokenized()` reads the regex from.

Measured on Python 3.12 on an arm64 laptop, the 21 MB KDL benchmark, wall clock, one run:

| Path | Total | Peak memory |
| --- | ---: | ---: |
| Tokens, then the compiled pass over them | 7.60 s | 1.67 GB |
| Values and types from the joined regex | 4.82 s | 0.60 GB |

Rejected:

- **A tokenizer given to `compile()`.** The pair exists only once compiled, so `Parser.parse_text()`, `matches()`,
  `replace()` and `located()` of a token grammar would need a second way in, and every feature of text would be built
  for each.
- **A `tokenizer=` argument on `parse_text()`, `matches()` and `replace()`.** The pair is restated at each call, and a
  bound part cannot sit inside a grammar.
- **Token parsers of text made from the tokenizer, `tokenizer.tok("name")`.** A grammar of `tok()` would get none of
  the features of text, and every grammar would have two spellings.
- **`parse()` accepting text.** A grammar over characters takes a `str` as its tokens, so a `str` would mean two things
  depending on the grammar.
- **`skip=` given with the grammar.** A program would list the same types in its tokenizer wrapper and in the call, and
  the wrapper every tokenizer had would stay.
- **Matches only where a token of the whole text starts.** A bound part inside a grammar has no tokens of the whole
  text, so `matches()` would need a case of its own, and a parser would scan differently where it is embedded.
- **`g.tokenized(t)` and `tokenizer.parser(g)`.** A method on `Parser` is for what acts on a finished parser's value or
  name (ADR 0008), and a method on `Tokenizer` reads as a feature of the lexer where it makes a parser.

## Consequences

- A failed parse runs the tokenizer on top of what ADR 0002 lists. `>>` functions run in two passes, as there.
- A bound grammar compiled with a joined regex generates its functions twice, which takes milliseconds.
- `make_tokenizer()` returns a `Tokenizer` where it returned a function. It is still called with text and still gives
  tokens one at a time.
- Tokenizing stays a `str` matter: `TokenSpec.pattern` is a `str`, and positions count `"\n"`.
- `Parser.parse_text()` of a grammar of `tok()` raises `ValueError`, with a hint naming `tokenized()`.
- `matches()` of a bound grammar starts a tokenizer at each offset it tries, and a parser reads the length of its
  tokens, which is the rest of the text. A tokenizer reads a token by the text alone, so the token at an offset is read
  once for a text, and every start shares the tokens of the offsets it reaches. On 400 KB, 800 KB and 1.6 MB of
  `"word x = 1 # prose; "` repeated, `matches()` of `tok("name") + -tok("op", "=") + tok("int")` took 0.97 s, 1.90 s and
  3.86 s, against 0.42 s, 0.88 s and 1.74 s for the same grammar without a tokenizer (Python 3.12, arm64 laptop, one
  run each).
- `tok("name", "if")` scanning `elif` matches at its second character, as `regex("if")` does. A grammar without a
  tokenizer guards with `\b`; a token grammar has no such place yet.
