"""funcparserlib against parsy and pyparsing: the same JSON document, text in, Python values out.

funcparserlib tokenizes and then parses; parsy and pyparsing parse the text directly. The `json`
group compares the whole job; `bench_funcparserlib_json_parse` is the parsing half alone.
"""

import json
from typing import Any

import pytest

from benchmarks._json_other import parsy_loads, pyparsing_loads
from benchmarks.bench_grammars import JSON_TEXT
from tests import json as json_grammar

EXPECTED = json.loads(JSON_TEXT)


@pytest.mark.benchmark(group="json")
def bench_funcparserlib_json(benchmark: Any) -> None:
    assert benchmark(json_grammar.loads, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_parsy_json(benchmark: Any) -> None:
    assert benchmark(parsy_loads, JSON_TEXT) == EXPECTED


@pytest.mark.benchmark(group="json")
def bench_pyparsing_json(benchmark: Any) -> None:
    assert benchmark(pyparsing_loads, JSON_TEXT) == EXPECTED
