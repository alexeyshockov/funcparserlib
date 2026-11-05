# 0002 — A compiled pass for the parses that succeed

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

A parser is a tree of closures, and every step allocates a `State` and a result. On a 2 MB slice of
the official KDL benchmark, 453,043 tokens, the KDL recipe makes about 68 Python calls per token,
and no single call dominates. Tuning the combinators did not move that much: on Python 3.12 a faster
token test gave 3%, n-ary sequences and a dedicated `maybe()` gave nothing measurable, and turning
error tracking off altogether, which breaks error messages, gave 11%. A recursive descent parser
written by hand over the same tokens took 0.18 s where the grammar took 2.5 s.

The cost that cannot be tuned away is what makes the errors good. Every failed token test records
what was expected, and 72% of the token tests in that run failed. A successful parse throws all of
it away. The invariant that an error names everything expected where parsing stopped is what any
faster design has to keep, and a second implementation of those rules is a second place for them to
disagree.

## Decision

`Parser.compile()` generates Python functions from a finished grammar and returns a
`CompiledParser`. The generated code tracks no error state: positions are local variables, a failed
alternative costs a comparison, and a token type is read from a list built once per parse.

Whenever that pass does not succeed, `CompiledParser.parse()` runs the interpreted parser on the same
tokens, and what it returns or raises is the result. This covers a failed parse, a failed
`Parser.check()`, a failure inside `commit()` and any exception. Errors are therefore produced by the
one implementation there is, and the compiled pass has a single obligation: when it succeeds, the
interpreted parser would have given the same value.

Each combinator records what it was built from in `Parser._form`. A parser with no form, such as a
function passed to `Parser()`, is called through `run` from the generated code. A grammar that the
generated code cannot express stays interpreted: nesting past what CPython compiles, and a `many()`
of a parser that matches no tokens, whose `ValueError` a first-token test would skip.

Compiling is opt-in. Measured on Python 3.12 on an arm64 laptop, CPU time, best of three:

| Input | Interpreted | Compiled |
| --- | ---: | ---: |
| KDL slice, 453,043 tokens, parsing | 2.40 s | 0.24 s |
| JSON, 945 KB, 327,545 tokens, parsing | 0.296 s | 0.090 s |
| DOT, 342 KB, 96,004 tokens, parsing | 0.304 s | 0.026 s |
| KDL benchmark, 21 MB, tokenizing and parsing, wall clock, one run | 32.45 s | 8.80 s |

Generating the KDL grammar takes 3 ms. A parse that fails at its last token took 2.72 s against
2.42 s, as both passes ran.

Rejected:

- **Compiling on its own, inside `Parser.parse()`.** Every grammar would get the speed, and also the
  costs below, including grammars whose `>>` functions have side effects and programs that parse once.
- **Generating the error tracking too.** It would keep failed parses fast, at the price of a second
  implementation of every error rule.
- **A `run(tokens, pos)` without `State`, in the interpreted parsers.** It breaks the public
  `ParserFn`, and it removes allocations while the 68 calls per token stay.
- **mypyc.** It needs a wheel per platform for a package that ships none, and the attempt did not
  build, so it has no figure.

## Consequences

- A failed parse takes longer, by the time the compiled pass ran.
- A `>>` function runs in both passes of a failed parse. A function with side effects sees them twice
  there; a parse that succeeds runs it as the interpreted parser does.
- A compiled grammar is a snapshot: a `forward_decl()` defined again afterwards keeps its old meaning
  in the compiled pass. A label given afterwards reaches errors, as they come from the parser.
- Recursion uses about a fifth of the stack, so a document nested too deeply for the interpreted parser
  can parse once compiled: 330 levels of KDL against 60.
- The package uses `exec()`. What it runs is built from the grammar: token types and values enter it
  through `repr()`, and every other object through a name bound to that object. Parsed text never
  reaches it.
- A new combinator works without a form, through `run`, and is only slower. A form that describes
  it wrongly makes the compiled pass accept what the parser rejects, which `tests/test_compile.py`
  looks for by running random grammars through both.
