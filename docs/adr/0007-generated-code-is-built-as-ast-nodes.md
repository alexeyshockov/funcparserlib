# 0007 — Generated code is built as `ast` nodes

- **Status:** Accepted
- **Date:** 2026-11-06

## Context

ADR 0002 generates Python functions, and the emitter wrote them as text. Statements and indentation had
helpers, but every expression was an f-string: precedence was the caller's to keep, a name was told from
an expression by `str.isidentifier()`, and a `str` constant went through `repr()`.

A mistake in that text did not fail. A malformed line was a `SyntaxError`, which `_functions` took for
nesting past what CPython compiles, so the grammar stayed interpreted. A wrong name was a `NameError` at
parse time, which the `except Exception` around generated code took for an error the parser would raise
again. Both ended in the interpreted parser and the same result: a missing colon and a misspelt name in
`emit_but_not` each passed all 552 tests of the time.

## Decision

The emitter builds `ast` nodes, and `compile()` takes the module they make. In an expression a `str` is a
name of generated code; a `str` constant is `ast.Constant`. Nothing declines on a `SyntaxError`, as
generated code stays inside what the compiler takes: `_MAX_LOOPS` holds the nested loops under the 20
blocks CPython compiles.

`_MAX_DEPTH` stays. Nodes have no limit of indentation, but CPython's compiler recurses over them: on
3.12 to 3.15, 700 nested `if` statements compile and 1,000 raise `RecursionError`. Without the limit a
grammar `test_deeply_nested_options_compile` compiles would stay interpreted.

A wrong name still compiles, as `ast.Name` takes any string, so the tests watch for it:
`tests/conftest.py` turns a `NameError` raised in a generated frame into an error no fallback catches,
and the random grammars without `check()` or `commit()` assert that generated code decides the result
alone.

The change moved no generated code: every grammar compiled by the test suite, 4,400 of them, gave the
same `ast.unparse()` text through both emitters, and 374 more were declined by both.

Rejected:

- **Bytecode, written directly.** It is as much concatenation, with stack depth to keep by hand and
  nothing to check it: wrong bytecode can crash the interpreter, where a wrong node raises. It follows
  CPython's instruction set, with 140 opcodes on 3.12, 150 on 3.13 and 154 on 3.14, and it runs in the
  interpreter loop that mypyc removes for the gain ADR 0002 measures.
- **Generated code that differs by Python version.** No syntax or builtin of 3.13 to 3.15 changes how
  this code is written, and the shapes tried did not gain on one version over another. Measured on `HEAD`
  of 2026-11-05 on an arm64 laptop, CPU time, best of six, JSON of 169,514 tokens and DOT of 78,004:
  48 ms and 13 ms on both 3.12 and 3.15. The 3.15 JIT takes DOT to 10 ms and leaves JSON at 49 ms; it is
  experimental and set by the user, and the same code gets it.

## Consequences

- A malformed node raises from `Parser.compile()`, as `TypeError` or `ValueError`, instead of leaving the
  grammar interpreted.
- Generating takes about twice as long: the KDL grammar from 3.3 ms to 5.9 ms on 3.12 and from 3.7 ms to
  6.6 ms on 3.15, once per `Parser.compile()`. Parsing is unchanged.
- An emitter reads longer than the f-string it replaced, through the helpers at the top of
  `_compile.py`.
- `CompiledParser.source` is `ast.unparse()` of the module, for reading it or for mypyc. Text goes through
  the tokenizer's indentation limit, so generated code is compiled from the nodes, with statements numbered
  as `ast.unparse()` lays them out; `tests/test_compile.py` compares that against parsing the text. The
  compiled parser keeps the module, as a redefined `forward_decl()` would give a regenerated one other code.
- A value the user gives enters through a name unless it is exactly a `str`: `ast.Constant` rejects a
  subclass of `int` or `str`.
