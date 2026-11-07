"""funcparserlib against parsy, pyparsing and lark: the same JSON document, text in, values out.

The stdlib `json` and orjson are there for scale: they are what parses JSON in practice, written in
C and Rust for this one format.

The `msgpack` group reads a MessagePack document with the `msgpack` package, as its C extension and
as its Python fallback, and with the grammar of `examples/msgpack_format.py`, interpreted and
compiled.

funcparserlib runs the grammar four ways: a token grammar tokenizes and then parses, a grammar
without a tokenizer reads the text directly, as parsy and pyparsing do, and each of the two also
compiled. lark runs as LALR(1), a table-driven parser with its own lexer. Each grammar is built
once, before the timing. The `json` group compares the whole job; `bench_json_parse` in
bench_grammars.py is funcparserlib's parsing half alone.
"""

import json
import random
from typing import Any

import msgpack
import msgpack.fallback
import orjson
import pytest

from benchmarks._json_other import lark_loads, parsy_loads, pyparsing_loads
from benchmarks._json_text import COMPILED, GRAMMAR
from benchmarks.bench_grammars import JSON_TEXT
from examples import msgpack_format
from funcparserlib.parser import tokenized
from tests import json as json_grammar

EXPECTED = json.loads(JSON_TEXT)
TOKENS_COMPILED = tokenized(json_grammar._grammar(), json_grammar._tokenizer()).compile()


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


@pytest.mark.benchmark(group="json")
def bench_lark_json(benchmark: Any) -> None:
    assert benchmark(lark_loads, JSON_TEXT) == EXPECTED


def _records(count: int) -> list[dict[str, Any]]:
    """Nested records from a fixed seed: strings, numbers, booleans, null, arrays and maps."""
    rng = random.Random(1)
    return [
        {
            "id": i,
            "name": f"user {i}",
            "active": i % 3 == 0,
            "score": rng.random() * 100,
            "tags": [f"t{j}" for j in range(rng.randrange(5))],
            "address": {
                "city": "x" * rng.randrange(3, 12),
                "zip": rng.randrange(10**5),
                "geo": [rng.random(), rng.random()],
            },
            "note": None,
        }
        for i in range(count)
    ]


MSGPACK_DATA = msgpack.packb(_records(4000))
MSGPACK_EXPECTED = msgpack.unpackb(MSGPACK_DATA, raw=False, strict_map_key=False, timestamp=0)
MSGPACK_COMPILED = msgpack_format.document.compile()


def _fallback_unpack(data: bytes) -> Any:
    unpacker = msgpack.fallback.Unpacker(None, raw=False, max_buffer_size=0)
    unpacker.feed(data)
    return unpacker.unpack()


@pytest.mark.benchmark(group="msgpack")
def bench_msgpack_c_extension(benchmark: Any) -> None:
    assert msgpack.Unpacker is not msgpack.fallback.Unpacker
    result = benchmark(msgpack.unpackb, MSGPACK_DATA, raw=False, strict_map_key=False, timestamp=0)
    assert result == MSGPACK_EXPECTED


@pytest.mark.benchmark(group="msgpack")
def bench_msgpack_python_fallback(benchmark: Any) -> None:
    assert benchmark(_fallback_unpack, MSGPACK_DATA) == MSGPACK_EXPECTED


@pytest.mark.benchmark(group="msgpack")
def bench_funcparserlib_msgpack(benchmark: Any) -> None:
    assert benchmark(msgpack_format.document.parse_bytes, MSGPACK_DATA) == MSGPACK_EXPECTED


@pytest.mark.benchmark(group="msgpack")
def bench_funcparserlib_msgpack_compiled(benchmark: Any) -> None:
    assert benchmark(MSGPACK_COMPILED.parse_bytes, MSGPACK_DATA) == MSGPACK_EXPECTED
