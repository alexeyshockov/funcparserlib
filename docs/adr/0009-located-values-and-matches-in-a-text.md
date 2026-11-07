# 0009 — Located values and matches in a text

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

A parse gives values and forgets where they came from. `tok()` gives a `str`, a grammar without a tokenizer
works out a line and a column only for an error (ADR 0003), and a tool that reports on a value after parsing,
or slices the text it came from, has nothing to go on.

A grammar also reads its input from the start, so it cannot be used on the parts of a text it describes. A
text format that evolves needs this: each new version needs a migration of the files in the old one, which
rewrites some constructs and has to leave the rest of a file exactly as it is, and must not pass over a
construct it cannot read.

`State.pos` is already an index into what is being parsed, a character offset in a text and a token index in
tokens. `Token` has a line and a column and no offset. A `skipping()` parser reads the skip after itself, so
`State.pos` after a value lies past the skip.

## Decision

**A value with its position.** `p.located()` gives `Parser[A, Located[B]]`. It is a method, as ADR 0008 has it
for what acts on what a finished parser gives, and it works on any value, as `tok("int").located()` on a leaf or
`(name + args).located()` on a rule.

`Located` has `value`, `start` and `end`, indexes into what was parsed, `end` exclusive: character offsets for
text, token indexes for tokens. A span ends where its value ends, before the skip read after it. A token grammar
bound to its tokenizer reads text, so its spans are character offsets (ADR 0003). One given tokens reads the place
of a span from them, `tokens[m.start].start`, which also serves the tokens of another library.

`located()` has the form `("located", inner)`, so that what walks a grammar sees through it: a first skip inside it is
still the skip a text may start with, and a `tok()` inside it still makes a grammar of tokens. Generated code builds
the `Located` itself. Over tokens a span ends at the position. Over text, the compiler carries where the value read
last ends, beside the position: a part of text sets it before its skip, a test of a character or a byte at its
position, and every place where generated code joins paths keeps it as it keeps the position, as `|`, `many()`,
`commit()`, `longest()` and `counted()` do. A generated function gives it back in `Y[0]`, which a grammar of text has no
use for, and -1 where it read nothing. This code is generated only for a grammar of text with a `located()` in it, and
a `prefixed()` with a body in such a grammar stays with the parser, as a span in it is relative to its region.

**Line and column.** `Lines(text)`, in `funcparserlib.util`, is built once from a text. `lines.place(offset)`
gives a `Place` and `lines.offset(place)` an offset, so both kinds of grammar reach both kinds of position.

**Matches in a text.** Two runners sit on top of a grammar, as ADR 0004 has it, on `Parser` and
`CompiledParser`:

- `p.scan(text)` gives an iterator of `Located[B]`, from left to right, with no overlaps. It tries the grammar
  at each offset, goes past a match, and is lazy, so a program can stop at the first one.
- `p.replace(text, f)` gives the text with each match replaced by what `f` returns for its `Located[B]`, and every
  character outside a span as it was. The skip after a match is outside its span.

They read text, and a token grammar scans a text bound to its tokenizer (ADR 0003). A grammar of `tok()` itself
raises `ValueError`, as it reads tokens, and bytes are left out, as `funcparserlib.binary` is experimental.

A plain failure at an offset means no match there. A failed `commit()` and a value `Parser.validated()` rejected
raise what `Parser.parse_text()` raises, with the line and column, so `commit(link)` says that what starts as a
link has to be one. `replace()` then returns nothing, so a program that writes its result never leaves a file half
migrated. A match that reads nothing raises `ValueError`, as it does in `many()`.

Measured on Python 3.12 on an arm64 laptop, best of 5, on 50 KB of JSON without a tokenizer:

| Grammar | Interpreted | Compiled |
| --- | ---: | ---: |
| Plain | 26 ms | 5.6 ms |
| `located()` on each value | 30 ms | 7.4 ms |

On 800 KB of text, best of 3, `scan()` of a grammar of three parsers found all 40,000 matches in 502 ms and the first
one in 0.017 ms, and `replace()` took 506 ms. `tests/test_located.py` compares both with `re` over random texts.

Rejected:

- **Finding the end of a span in the text, as the earliest offset the rest of it fully matches the skip pattern.** The
  rest of `"a #b"` matches a comment from `#`, so the span of a string ended inside it and `replace()` left the rest
  of the string in the output. It also read the text from the start of each span.
- **A position parser that reads nothing, `position + p + position`.** It gives `State.pos`, past the skip: `foo`
  followed by a comment measured 0 to 11 where the value ends at 3.
- **`tok()` giving a `str` subclass with a position.** At runtime it keeps 1.0 grammars working: `==`, `hash`,
  `repr` and `int()` behave, and `type(v) is str` does not. The position is gone after `>> int`, `.upper()`, a slice
  or `+`, and a rule's value is not a `str`, so `located()` is needed anyway. Declared as a new type, it fails
  `def op(name) -> Parser[Token, str]` under mypy, as `Parser` is invariant in its value; declared as `str`,
  checkers cannot see the position. Making one per token took 1.85 ms of 13,144 tokens, against 10.9 ms for an
  interpreted parse and 3.0 ms for a compiled one, paid by grammars that do not use it.
- **An offset on `Token`.** Every program that tokenizes would pay for a field few read, and tokens are where
  ADR 0003 found the cost. `Lines.offset()` gives it to those who ask.
- **Line and column worked out during the parse.** No per-parse object can hold a table, and a lookup from the text
  each time took 4.2 s for 45,576 lookups on 400 KB without a newline, against 0.8 ms with a table.
- **A function, `scan(p)`, as a parser of `many(p.located() | -anything)`.** It reads to the end of the input, so
  nothing can follow it: an entry point written as a combinator. It also read tokens, which these runners do not,
  and took 1.16 s on 842 KB, as it is not lazy and pays for an item at each character.
- **A generated entry with a start position.** It would need new generated code and its own gates under ADR 0002,
  for a speed nothing has asked for yet.
- **`match()`.** In `re` it is one match at the start of the text. `matches()` reads as a test (ADR 0008).

## Consequences

- `Located`, `Lines`, `Parser.located()`, `scan()` and `replace()` are public, and the glossary gets their terms.
- A compiled grammar with `located()` on each value takes about a third more than one without, against four times as
  much when it ran through the parser. `scan()` of a compiled grammar reads through its parser; a generated entry can
  come later, without changing what it gives.
- `State` keeps how much the last parser of text skipped, so a span ends where that parser stopped, whatever the skip
  pattern matches inside the value. A parser function of a user's that skips by moving `pos` has its skip in the span.
- Inside a `prefixed()` region, positions are relative to the region, as it is parsed as its own input (ADR 0005).
- A function after `>>` runs for an attempt that fails later at that offset, as on any alternative that fails.
- Scanning a sequence of tokens is left out. `parse()` takes tokens and `parse_text()` text for the reason ADR 0003
  gives, so it needs a name of its own when a program needs it.
