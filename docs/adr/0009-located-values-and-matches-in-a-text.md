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
reads the place of a span from its tokens, `tokens[m.start].start`, which also serves the tokens of another
library. The private `_Located` of `parser.py` takes another name.

**Line and column.** `Lines(text)`, in `funcparserlib.util`, is built once from a text. `lines.place(offset)`
gives a `Place` and `lines.offset(place)` an offset, so both kinds of grammar reach both kinds of position.

**Matches in a text.** Two runners sit on top of a grammar, as ADR 0004 has it, on `Parser` and
`CompiledParser`:

- `p.matches(text)` gives an iterator of `Located[B]`, from left to right, with no overlaps. It tries the grammar
  at each offset, goes past a match, and is lazy, so a program can stop at the first one.
- `p.replace(text, f)` gives the text with each match replaced by what `f` returns for its `Located[B]`, and every
  character outside a span as it was. The skip after a match is outside its span.

They read text. A grammar of `tok()` raises `ValueError`, as `CompiledParser.parse_text()` does without a tokenizer,
since a tokenizer stops at text it cannot read (ADR 0004), and bytes are left out, as `funcparserlib.binary` is
experimental.

A plain failure at an offset means no match there. A failed `commit()` and a value `Parser.validated()` rejected
raise what `Parser.parse_text()` raises, with the line and column, so `commit(link)` says that what starts as a
link has to be one. `replace()` then returns nothing, so a program that writes its result never leaves a file half
migrated. A match that reads nothing raises `ValueError`, as it does in `many()`.

Measured on Python 3.12 on an arm64 laptop, with prototypes over `ParserFn`, best of 5 on 50 KB of JSON with every
value located:

| Grammar | Interpreted | Compiled |
| --- | ---: | ---: |
| Plain | 25 ms | 5.6 ms |
| `located()` on each value, span ends past the skip | 41 ms | 41 ms |
| `located()` on each value, span ends before the skip | 67 ms | 65 ms |

`located()` has no form at first, so a compiled grammar calls it through `run` and the parsers inside it are
interpreted. A grammar that does not use it is not touched.

A loop over offsets found every match in 800 KB of text in 479 ms, and the first one in 0.013 ms, without
`located()`. 20,000 random texts rewritten through a grammar of `many()` and `|`, interpreted and compiled, gave
what `re.sub()` gives, which is the behaviour these runners have to keep.

Rejected:

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
- **A function, `matches(p)`, as a parser of `many(p.located() | -anything)`.** It reads to the end of the input, so
  nothing can follow it: an entry point written as a combinator. It also read tokens, which these runners do not,
  and took 1.16 s on 842 KB, as it is not lazy and pays for an item at each character.
- **A generated entry with a start position.** It would need new generated code and its own gates under ADR 0002,
  for a speed nothing has asked for yet.
- **`match()`.** In `re` it is one match at the start of the text.

## Consequences

- `Located`, `Lines`, `Parser.located()`, `matches()` and `replace()` are public, and the glossary gets their terms.
- A compiled grammar with `located()` runs at about the speed of the interpreted one, and `matches()` of a compiled
  grammar reads through its parser. A form for `located()` and a generated entry can come later, without changing
  what they give.
- Finding where a span ends reads the text from its start, so a span of a whole document costs more than a small
  one. Nested spans read their text once for each level.
- Inside a `prefixed()` region, positions are relative to the region, as it is parsed as its own input (ADR 0005).
- A function after `>>` runs for an attempt that fails later at that offset, as on any alternative that fails.
- Scanning a sequence of tokens is left out. `parse()` takes tokens and `parse_text()` text for the reason ADR 0003
  gives, so it needs a name of its own when a program needs it.
