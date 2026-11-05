"""Whole grammars over generated documents: the JSON and DOT test grammars and the calculator."""

import json
import random
from typing import Any

from examples.calculator import calculate
from tests import dot
from tests import json as json_grammar


def _json_document(size: int) -> str:
    rnd = random.Random(1)

    def value(depth: int) -> object:
        k = rnd.random()
        if depth > 4 or k < 0.4:
            return rnd.choice(
                [
                    None,
                    True,
                    rnd.randint(-(10**6), 10**6),
                    rnd.random() * 1e3,
                    "".join(rnd.choice('abcxyz é"\\') for _ in range(rnd.randint(0, 12))),
                ]
            )
        if k < 0.7:
            return [value(depth + 1) for _ in range(rnd.randint(0, 6))]
        return {f"k{rnd.randint(0, 99)}": value(depth + 1) for _ in range(rnd.randint(0, 6))}

    items: list[object] = []
    while len(json.dumps(items)) < size:
        items.append(value(0))
    return json.dumps(items)


JSON_TEXT = _json_document(50_000)
JSON_TOKENS = json_grammar.tokenize(JSON_TEXT)
DOT_TEXT = (
    "digraph g { "
    + " ".join(f'n{i} -> n{i + 1} [label="e{i}", weight={i}];' for i in range(400))
    + " }"
)
DOT_TOKENS = dot.tokenize(DOT_TEXT)
EXPRESSION = " + ".join(f"({i} * {i + 1} - {i} / 2)" for i in range(1, 200))


def bench_json_tokenize(benchmark: Any) -> None:
    tokens = benchmark(json_grammar.tokenize, JSON_TEXT)
    assert len(tokens) == len(JSON_TOKENS)


def bench_json_parse(benchmark: Any) -> None:
    assert benchmark(json_grammar.parse, JSON_TOKENS) == json.loads(JSON_TEXT)


def bench_json_parse_compiled(benchmark: Any) -> None:
    compiled = json_grammar._grammar().compile()
    assert benchmark(compiled.parse, JSON_TOKENS) == json.loads(JSON_TEXT)


def bench_dot_parse(benchmark: Any) -> None:
    graph = benchmark(dot.parse, DOT_TOKENS)
    assert len(graph.stmts) == 400


def bench_dot_parse_compiled(benchmark: Any) -> None:
    compiled = dot._grammar().compile()
    graph = benchmark(compiled.parse, DOT_TOKENS)
    assert len(graph.stmts) == 400


def bench_calculator(benchmark: Any) -> None:
    assert benchmark(calculate, EXPRESSION) == eval(EXPRESSION)  # noqa: S307 — digits and operators only
