# 0004 — One definition per grammar

- **Status:** Accepted, not built yet
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
- A **text grammar** reads text at the position the parser has reached, and that defines it. Only the
  tokens that make sense at a position are tried, so specs have no global order, and context is
  recursion and `bind()`. Its generated code reads the text directly.

The other way of reading never decides what is accepted. A token grammar may use reading at the
position to explain a failure, and when that explanation disagrees with the tokens, for instance by
accepting the text, the error is the one the tokens give.

Runners sit on top of one grammar, and a program picks one by what it wants from a failure: the default
is fast and gives the full error, one only rejects, and one diagnoses at length. Runners are added when
they earn it.

Text grammars come with two tools that earn their concepts. A choice that keeps the longest match
removes the order of alternatives: KDL passed 338 of 338 with `string`'s alternatives in the order that
fails 16 cases under `|`. A token written as grammar gives errors inside it:
`"abc` fails at `1,5` with `expected: text or escape or unicode escape or '"' in string`, where today it
fails at `1,3` with `cannot tokenize data`.

Positions in a text grammar are worked out from the offset when an error happens, so a parse that
succeeds pays nothing for them.

Measured on Python 3.12 on an arm64 laptop, CPU time, with a prototype of the generated code that has
no first-token tests yet:

| Input | Tokens, interpreted | `parse_text()` | Text grammar, compiled |
| --- | ---: | ---: | ---: |
| KDL, 2 MB | 2.86 s | 0.35 s | 0.45 s |
| JSON, 823 KB | 0.56 s | 0.15 s | 0.23 s |

The 21 MB KDL benchmark took 5.72 s as a compiled text grammar, against 4.82 s for `parse_text()`, wall
clock. Compiled reading at the position agreed with global tokens on 60,000 mutated KDL documents.

Rejecting without an error, on the 2 MB KDL document:

| Failure | Fast pass alone | `parse_text()` with the error | Text grammar, compiled |
| --- | ---: | ---: | ---: |
| At the last token | 0.37 s | 3.44 s | 0.48 s |
| At the first token | 0.14 s | 0.55 s | under 0.01 s |

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
- The library has two kinds of grammar to explain and test. `compile()` needs a backend for text
  grammars, and its speed depends on first-token tests the prototype leaves out.
- A grammar that ignores whitespace needs what `skip=` gives a tokenizer, applied before each token. The
  prototype's version took 205 KB of JSON from 0.12 s to 0.38 s, both interpreted.
- Still to settle when this is built: the public names of the primitive, the longest choice and the
  runners; what a position means at the end of input, where the two ways of reading differ; positions
  in `Parser.check()` errors; whether `bytes` come for free, as `re` reads them; and reporting every
  error of a document, which needs recovery and has no prototype.
