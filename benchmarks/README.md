# funcparserlib benchmarks

Micro-benchmarks of tokenizing and parsing: `just bench::micro` runs them, `just bench::compare` compares saved runs,
and `CONVENTIONS.md`'s Benchmarks section has the rules.

- `bench_grammars.py`: whole grammars over a generated document, the JSON and DOT grammars of `tests/` and the
  calculator of `examples/`.
- `bench_combinators.py`: one combinator each, over a long input, so a regression points at its combinator.
- `bench_comparison.py`: the JSON grammar against the same grammar in parsy and pyparsing (`_json_other.py`), text
  in, values out; each case asserts it returns what `json.loads()` does.
