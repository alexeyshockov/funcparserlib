# funcparserlib benchmarks

Micro-benchmarks of tokenizing and parsing: `just bench::micro` runs them, `just bench::compare` compares saved runs,
and `CONVENTIONS.md`'s Benchmarks section has the rules.

They run in the noxfile's `bench` session tagged `bench`, on the newest supported Python; another interpreter's runs by
name, `just tools::nox -s "bench(python='3.14')" -- benchmarks`. Quote a figure with the interpreter it ran on. On
Python 3.12 a recursive parser's time depends on the call-stack depth it starts at (up to 57% here, CPython's slow path
for calls across data-stack chunks, [python/cpython#142183](https://github.com/python/cpython/issues/142183)); on 3.14
and 3.15 it varies by about 1 ms.

- `bench_grammars.py`: whole grammars over a generated document, the JSON and DOT grammars of `tests/` and the
  calculator of `examples/`.
- `bench_combinators.py`: one combinator each, over a long input, so a regression points at its combinator.
- `bench_comparison.py`: the JSON grammar against the same grammar in parsy and pyparsing (`_json_other.py`), text
  in, values out, with funcparserlib as a token grammar and as a text grammar (`_json_text.py`), each interpreted and
  compiled; each case asserts it returns what `json.loads()` does. The stdlib `json` and orjson show the scale of a
  parser written for this one format.
