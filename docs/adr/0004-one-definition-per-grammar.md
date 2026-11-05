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

Generated code tests the first character before it runs an alternative of `|` or `longest()` that starts
with a regexp, as a failed regexp costs a call and a failed character test does not. What a regexp can
start with comes from `re._parser`, which is private, so anything read differently than expected, such as
a category like `\d`, a case-insensitive flag or a change in a later Python, leaves that alternative
without a test. A wrong test would skip an alternative and could return another's value, with no fallback,
so `tests/test_compile.py` compares the characters against real matches of random regexps. The tests are
for `str`: a `bytes` input goes to the interpreted parser, where a `str` regexp raises. When `|` is all a
function does, each alternative returns from it, without a flag for the ones after it.

Measured with `just bench::micro`, Python 3.15 on an arm64 laptop, minimum:

| Compiled grammar | Before | After |
| --- | ---: | ---: |
| KDL text, 46 KB generated | 18.57 ms | 13.30 ms |
| JSON text, 50 KB | 5.48 ms | 5.06 ms |
| JSON tokens, 50 KB | 3.20 ms | 3.13 ms |

Generating the KDL text grammar takes 13 ms against 6 ms, most of it the character sets.

Rejected:

- **First-character tests from literals alone.** No private module, and 1.00x to 1.03x on KDL text, as its
  alternatives start with regexps.
- **A regexp of each terminal and the skip after it.** At most 1.05x on JSON text, and nothing on KDL text,
  which skips nothing.
- **Token values in the first sets of token grammars.** The hand-written bound for JSON tokens, with branches
  on type and value, was 1.06x, of which returning from alternatives already gives 1.02x.
- **A cut where failure can only mean the parse fails, from FIRST and FOLLOW sets.** A wrong cut only falls
  back, but it took JSON tokens from 1.06x to 1.10x by hand, for an analysis of every parser's context.
- **Packrat memoization, on for a grammar.** It removes 0% to 2% of parser runs in the JSON, DOT, Markdown
  and YAML grammars, and made compiled KDL 1.4x slower and JSON 1.1x to 1.2x slower. Only a grammar that
  reparses nested input at every level gains, from exponential time to linear; a failed parse still reruns
  the interpreted parser, whose error state a memo would have to replay.
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
- `longest()` runs every alternative that can start with the character at the position, in generated code
  too.
- A `Text` skips after what it reads, so the parser after a `Text` without `skip` does not skip what
  comes before it, but for `Text.end`.
- `bytes` are not supported: `Text` takes `str` patterns, and positions count `"\n"`.
- `Parser.recover()` gives one parse every error of a document, through what `Parser.check()` collects.
  A failure it catches leaves the generated code, so such a document takes the time of the interpreted
  parser.
