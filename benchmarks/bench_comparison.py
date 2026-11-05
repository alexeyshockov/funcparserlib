"""funcparserlib against parsy and pyparsing: the same JSON document, text in, Python values out.

The stdlib `json` and orjson are there for scale: they are what parses JSON in practice, written in
C and Rust for this one format.

funcparserlib runs the grammar four ways: a token grammar tokenizes and then parses, a text grammar
reads the text directly, as parsy and pyparsing do, and each of the two also compiled. Each grammar
is built once, before the timing. The `json` group compares the whole job; `bench_json_parse` in
bench_grammars.py is funcparserlib's parsing half alone.
"""

import json
from typing import Any

import orjson
import pytest

from benchmarks._json_other import parsy_loads, pyparsing_loads
from benchmarks._json_text import COMPILED, GRAMMAR
from benchmarks.bench_grammars import JSON_TEXT
from tests import json as json_grammar

EXPECTED = json.loads(JSON_TEXT)
TOKENS_COMPILED = json_grammar._grammar().compile(json_grammar._tokenizer())


@pytest.mark.benchmark(group="json")
def bench_stdlib_json(benchmark: Any) -> None:
    assert benchmark(json.loads, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_orjson(benchmark: Any) -> None:
    assert benchmark(orjson.loads, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_funcparserlib_json(benchmark: Any) -> None:
    assert benchmark(json_grammar.loads, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_funcparserlib_json_compiled(benchmark: Any) -> None:
    assert benchmark(TOKENS_COMPILED.parse_text, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_funcparserlib_text_json(benchmark: Any) -> None:
    assert benchmark(GRAMMAR.parse_text, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_funcparserlib_text_json_compiled(benchmark: Any) -> None:
    assert benchmark(COMPILED.parse_text, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_parsy_json(benchmark: Any) -> None:
    assert benchmark(parsy_loads, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_pyparsing_json(benchmark: Any) -> None:
    assert benchmark(pyparsing_loads, JSON_TEXT) == EXPECTED
