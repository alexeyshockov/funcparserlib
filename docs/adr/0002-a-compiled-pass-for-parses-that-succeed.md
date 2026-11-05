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
`Parser.check()`, a failure of `commit()` after it read a token, and any exception. Errors are
therefore produced by the
one implementation there is, and the compiled pass has a single obligation: when it succeeds, the
interpreted parser would have given the same value.

Each combinator records what it was built from in `Parser._form`. A parser with no form, such as a
function passed to `Parser()`, is called through `run` from the generated code. CPython's compiler
recurses over nested code, compiling 700 nested `if` statements and raising `RecursionError` at 1,000 on
3.12 to 3.15, and takes 20 nested loops, so a long sequence is split into functions, and code nested
too deeply moves to a function of its own. A grammar that the generated code still cannot
express stays interpreted: one so deep the generator runs out of stack, a `forward_decl()` defined
as itself, and a `many()` of a parser that matches no tokens, whose `ValueError` a first-token test
would skip.

The emitter builds `ast` nodes, and `compile()` takes the module they make. In an expression a `str` is
a name of generated code; a `str` constant is `ast.Constant`. Generated code stays inside what the
compiler takes, so nothing declines on a `SyntaxError`, and a malformed node raises from
`Parser.compile()`. A wrong name still compiles, as `ast.Name` takes any string, and would fall back to
the same result, so `tests/conftest.py` turns a `NameError` raised in a generated frame into an error no
fallback catches, and the random grammars without `check()` or `commit()` assert that generated code
decides the result alone.

Compiling is opt-in. Measured on Python 3.12 on an arm64 laptop, CPU time, best of three:

| Input | Interpreted | Compiled |
| --- | ---: | ---: |
| KDL slice, 453,043 tokens, parsing | 2.41 s | 0.25 s |
| JSON, 945 KB, 327,545 tokens, parsing | 0.287 s | 0.093 s |
| DOT, 342 KB, 96,004 tokens, parsing | 0.288 s | 0.025 s |
| KDL benchmark, 21 MB, tokenizing and parsing, wall clock, one run | 32.45 s | 8.90 s |

Generating the KDL grammar takes 6 ms. A parse that fails at its last token took 2.89 s against 2.56 s,
as both passes ran.

Rejected:

- **Compiling on its own, inside `Parser.parse()`.** Every grammar would get the speed, and also the
  costs below, including grammars whose `>>` functions have side effects and programs that parse once.
- **Generating the error tracking too.** It would keep failed parses fast, at the price of a second
  implementation of every error rule.
- **A `run(tokens, pos)` without `State`, in the interpreted parsers.** It breaks the public
  `ParserFn`, and it removes allocations while the 68 calls per token stay.
- **mypyc, over the library.** It needs a wheel per platform for a package that ships none.
- **mypyc, over the generated code.** The source types mechanically, passes `mypy --strict` and builds
  in 3 s. It takes the KDL slice from 0.254 s to 0.181 s and the 21 MB benchmark from 8.90 s to 8.23 s:
  the code it compiles took 0.123 s and takes 0.061 s, and the rest is `>>` functions, the tree, garbage
  collection and the tokenizer. Integer token types gave nothing more, at 0.188 s. Users would need a
  build step and a wheel per platform and Python version, for 8% end to end. The step can come later,
  as the source `ast.unparse()` gives stays typable.
- **Generating source text.** It generates in 3.3 ms against 5.9 ms for the KDL grammar on 3.12. Every
  expression is then an f-string with precedence to keep by hand, and a malformed line is a `SyntaxError`
  that nesting past CPython's limits also raises, so a mistake leaves the grammar interpreted unseen.
- **Bytecode, written directly.** It is as much concatenation, with stack depth to keep by hand and
  nothing to check it: wrong bytecode can crash the interpreter, where a wrong node raises. It follows
  CPython's instruction set, with 140 opcodes on 3.12, 150 on 3.13 and 154 on 3.14, and it runs in the
  interpreter loop that mypyc removes for the gain measured above.
- **Generated code that differs by Python version.** No syntax or builtin of 3.13 to 3.15 changes how
  this code is written, and the shapes tried did not gain on one version over another. On an arm64
  laptop, CPU time, best of six, JSON of 169,514 tokens and DOT of 78,004 take 48 ms and 13 ms on both
  3.12 and 3.15. The 3.15 JIT takes DOT to 10 ms and leaves JSON at 49 ms; it is experimental and set by
  the user, and the same code gets it.

## Consequences

- A failed parse takes longer, by the time the compiled pass ran.
- A `>>` function runs in both passes of a parse the compiled pass does not finish. That is a failed
  parse, a parse past a `check()` that failed on a branch `|` backtracked from, and a parse past a
  failure of `commit()` around a parsing function, whose reading the compiled pass cannot see. Any
  other parse that succeeds runs it as the interpreted parser does.
- A compiled grammar is a snapshot: a `forward_decl()` defined again afterwards keeps its old meaning
  in the compiled pass. A label given afterwards reaches errors, as they come from the parser.
- Recursion uses about a fifth of the stack, so a document nested too deeply for the interpreted parser
  can parse once compiled: 330 levels of KDL against 60.
- The package uses `exec()`. What it runs is built from the grammar: a token type or value that is
  exactly a `str` enters it as `ast.Constant`, and every other object through a name bound to that
  object, as `ast.Constant` rejects a subclass of `int` or `str`. Parsed text never reaches it.
- `CompiledParser.source` is `ast.unparse()` of the module. Text goes through the tokenizer's indentation
  limit, so generated code is compiled from the nodes, with statements numbered as `ast.unparse()` lays
  them out, which `tests/test_compile.py` compares against parsing the text. The compiled parser keeps the
  module, 1.2 MB for the KDL grammar, as a `forward_decl()` defined again would regenerate other code.
- A new combinator works without a form, through `run`, and is only slower. A form that describes
  it wrongly makes the compiled pass accept what the parser rejects, which `tests/test_compile.py`
  looks for by running random grammars through both.
