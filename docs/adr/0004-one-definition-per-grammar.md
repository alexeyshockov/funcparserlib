# 0004 — One definition per grammar

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

The parser side of the library is composed from combinators, while the tokenizer is an ordered list of
regexps where the first match wins. Three things are hard with such a list: specs that fit together,
seeing what the tokenizer did, and tokens whose meaning depends on where they are. A regexp cannot
count, which is why the KDL recipe limits the nesting of comments, and the YAML and Markdown recipes
take their tokens from other libraries. Errors matter as much: what the tokenizer rejects gets
`cannot tokenize data`, with nothing about what was expected.

A prototype read tokens at the position the parser had reached, with the specs as a table by name. The
KDL grammar, its text unchanged but for the order of one `|`, passed all 338 cases of the official
suite, and the JSON and DOT test suites passed but for the tests that expect `LexerError` or read
tokens. Comments nested 40 levels became a recursive rule, and indentation took `bind()`.

Reading tokens that way and reading them globally can disagree. Over 240,000 mutated documents of the
KDL, JSON and DOT grammars they never did, as those specs were written to be read globally. Random
grammars over overlapping specs gave 169 different trees, 198 documents only global tokens accepted
and 3,809 only reading at the position accepted.

Use cases differ in what they want from a run: speed and a plain rejection for a web request that can
only log a failure, every error at once for a configuration read at startup. A grammar stays the same
across them.

## Decision

A grammar has one definition of its language, and there are two kinds.

- A **token grammar** is what the library has today: global tokens define it, and a 1.0 grammar stays
  one. Its fast path reads the same tokens, so the two cannot disagree.
- A **text grammar** is made of the parsers of `funcparserlib.text.Text`, which read text at the
  position the parser has reached, and that defines it. Only the tokens that make sense at a position
  are tried, so specs have no global order, and context is recursion and `bind()`. Each token reads
  what `skip` matches after it, and the first `skip` of the grammar is read before parsing starts, so a
  failed attempt moves nothing. Its generated code reads the text directly.

The two ways of reading never share a grammar, so they cannot disagree on what is accepted. When
`CompiledParser.parse_text()` of a token grammar stops at text no spec matches, the tokens before it
parse with one more token no grammar reads, and the error is theirs: a mistake among them, or what the
grammar expected in place of that text.

Runners sit on top of one grammar, and a program picks one by what it wants from a failure.
`CompiledParser.parse_text()` is fast and gives the full error, and `CompiledParser.try_parse()` gives
`None` without making one. Others are added when they earn it.

Text grammars come with two tools that earn their concepts. `longest()` removes the order of
alternatives: KDL passes 338 of 338 with `string`'s alternatives in the order that fails 16 cases under
`|`. A token written as grammar gives errors inside it:
`a "abc` fails at the end of the text with `expected: text or escape or '"' in string`, where the token
grammar fails at the quote.

Positions in a text grammar are worked out from the offset when an error happens, so a parse that
succeeds pays nothing for them.

Measured on Python 3.12 on an arm64 laptop, CPU time, on 2 MB of the KDL benchmark, with
`examples/kdl.py` and `examples/kdl_text.py`, which has strings as grammar and `longest()` over five
alternatives:

| Run | Token grammar | Text grammar |
| --- | ---: | ---: |
| Interpreted | 3.09 s | 7.46 s |
| `parse_text()`, compiled | 0.36 s | 0.54 s |
| `try_parse()`, failing at the last token | 0.36 s | 0.56 s |
| `try_parse()`, failing at the first token | 0.14 s | under 0.01 s |
| `parse_text()`, failing at the last token | 3.40 s | 8.07 s |

The 21 MB benchmark takes 6.89 s as a compiled text grammar, against 4.82 s, wall clock. 823 KB of JSON
written as a text grammar takes 0.120 s compiled and 2.06 s interpreted, against 0.15 s for
`parse_text()` of the token grammar in the tests.

Where both ways of reading rejected a document, they named the same place for nearly all grammar errors
in KDL and JSON and for 93% in DOT, where the differences read were at the end of input. For errors
the tokenizer raises, it was 48% to 76%: in the differences read, reading at the position found the
earlier mistake, as a tokenizer stops at the first text it cannot read, past any grammar error before
it.

Rejected:

- **Reading at the position as the definition of every grammar.** A 1.0 grammar has to parse to the same
  value, and overlapping specs show it would not always.
- **A check that specs cannot overlap.** It held for no real grammar here.
- **A check at run time that one spec matches at each token.** No disagreement passed it, but it took
  0.51 s on the 2 MB document, more than the whole fast path, and it rejects harmless overlaps.
- **Lexer states.** The position in a text grammar is the state already.
- **Combinators for the tokens themselves, as the main direction.** Defining atoms is not one of the
  three problems.

## Consequences

- Nothing changes for existing grammars unless they opt in.
- The library has two kinds of grammar to explain and test, and generated code for both.
- An interpreted text grammar is slow, as `Text` parsers cannot be fused as token tests are, and a failed
  parse pays for it: a text grammar that often fails on large input wants `try_parse()`.
- `longest()` runs every alternative, in generated code too.
- A `Text` skips after what it reads, so the parser after a `Text` without `skip` does not skip what
  comes before it, but for `Text.end`.
- `bytes` are not supported: `Text` takes `str` patterns, and positions count `"\n"`. Reporting every
  error of a document needs recovery.
