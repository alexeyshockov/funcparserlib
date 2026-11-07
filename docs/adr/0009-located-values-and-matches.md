# 0009 — Located values and matches

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

The structure of such a file is often read by another library. A format on top of Markdown is read from markdown-it's
tokens, and a construct of it is a sequence of them: a heading and the blocks under it, in a list item or a block quote
as anywhere else. A grammar accepts the ways a construct is written and builds its value, and a migration replaces the
construct with the canonical text it renders from that value. Only the text outside the constructs has to stay to the
character. markdown-it reads a whole document, so the tokens it gives from an offset inside one are not those of the
document, and its tokens keep the lines of a block and none of its layout.

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

**Matches.** Three runners sit on top of a grammar, as ADR 0004 has it, on `Parser` and `CompiledParser`, named as
`parse()` and `parse_text()` are (ADR 0003, ADR 0008):

- `p.scan(tokens)` gives an iterator of `Located[B]` of a sequence of tokens, from left to right, with no overlaps.
  It tries the grammar at each token index, goes past a match, and is lazy, so a program can stop at the first one.
  A construct inside a list item or a block quote is found as one anywhere else, as its tokens are in the sequence.
  Its spans are token indexes. Given a function from a range of tokens to the span of the text they came from, as
  `describe=` is for the errors of `parse()`, it gives spans of that text.
- `p.scan_text(text)` does the same at each offset of a text, and a grammar that starts with a bound part at each token
  of the text (ADR 0003).
- `p.replace_text(text, f)` gives the text with each match replaced by what `f` returns for its `Located[B]`, and
  every character outside a span as it was. The skip after a match is outside its span.

A token grammar scans a text bound to its tokenizer (ADR 0003), and the tokens of another library as they are. A
grammar of `tok()` given text raises `ValueError`, and bytes are left out, as `funcparserlib.binary` is experimental.
A replacement over the tokens of another library is the program's: it writes the text from the spans, and a prefix
each line of a container has, as `>` in a block quote, is a matter of the format the library does not know.

A plain failure at an offset means no match there. A failed `commit()` and a value `Parser.validated()` rejected
raise what `Parser.parse_text()` or `Parser.parse()` raises, with its place, so `commit(link)` says that what starts
as a link has to be one. `replace_text()` then returns nothing, so a program that writes its result never leaves a file
half migrated. A match that reads nothing raises `ValueError`, as it does in `many()`.

Measured on Python 3.12 on an arm64 laptop, best of 5, on 50 KB of JSON without a tokenizer:

| Grammar | Interpreted | Compiled |
| --- | ---: | ---: |
| Plain | 26 ms | 5.6 ms |
| `located()` on each value | 30 ms | 7.4 ms |

On 800 KB of text, best of 3, `scan_text()` of a grammar of three parsers found all 40,000 matches in 502 ms and the
first one in 0.017 ms, and `replace_text()` took 506 ms. `tests/test_located.py` compares both with `re` over random texts.

Rejected:

- **A concrete syntax tree, a runner giving every character of the input in a tree.** The tokens of markdown-it keep
  no layout, so no tree of them gives a file back, and a migration needs only the text outside its constructs kept,
  which spans give. In a grammar without a tokenizer it would serve a formatter of a whole file's layout, which no
  use here asks for, against a second way of running every combinator.
- **`scan_tokens()` beside `scan(text)`.** The scan methods would be named the other way round from `parse()` and
  `parse_text()`.
- **A scan of tokens written by each program,** as a loop of `parse()` over `tokens[i:]`. Each program then decides
  what a failed `commit()`, a rejected value and a match that reads nothing mean, which `scan_text()` already
  decides, and copies the rest of the tokens at each index.
- **markdown-it bound with `tokenized()` and scanned as text.** It reads a whole document, so the tokens it gives from
  an offset inside one are not the document's.
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
- **`match()`.** In `re` it is one match at the start of the text. `matches()` reads as a test (ADR 0008).

## Consequences

- `Located`, `Lines`, `Parser.located()`, `scan()`, `scan_text()` and `replace_text()` are public, and the glossary
  gets their terms.
- A compiled grammar with `located()` on each value takes about a third more than one without, against four times as
  much when it ran through the parser.
- A compiled scan runs the generated code of `located()` of its grammar at each offset or index, as a generated
  function already takes the position it starts at, and the parser decides a position where that code cannot, as it
  decides a failed parse (ADR 0002). It took 0.12 s against 0.58 s for a regex over 792 KB, 0.19 s against 0.84 s for
  a grammar of `skipping()` over 800 KB, and 17 ms against 57 ms over 100,000 tokens. A scan of tokens reads the type
  of every token first, as a compiled parse does, and a bound part of `tokenized()` runs as the parser, as inside any
  compiled grammar. The shadow session compares each match of a scan with the compiled one.
- `State` keeps how much the last parser of text skipped, so a span ends where that parser stopped, whatever the skip
  pattern matches inside the value. A parser function of a user's that skips by moving `pos` has its skip in the span.
- Inside a `prefixed()` region, positions are relative to the region, as it is parsed as its own input (ADR 0005).
- A function after `>>` runs for an attempt that fails later at that offset, as on any alternative that fails.
- The Markdown recipe shows a migration over markdown-it: a construct found by `scan()`, its lines from the `map` of
  its tokens, and the canonical text with the prefix of each line put back.
