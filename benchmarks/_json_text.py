"""The JSON grammar of `tests/json.py` as a text grammar, for the comparison in bench_comparison.py.

`NUMBER`, `STRING`, `unescape()` and `number()` are those the parsy and pyparsing grammars use, so
that the comparison measures the parsing.
"""

from typing import Any

from benchmarks._json_other import NUMBER, STRING, number, unescape
from funcparserlib.parser import ForwardDeclParser, Parser, forward_decl, many, maybe
from funcparserlib.text import Text

spaced = Text(skip=r"[ \t\r\n]+")


def _array(values: tuple[Any, list[Any]] | None) -> list[Any]:
    return [] if values is None else [values[0], *values[1]]


def _keyword(word: str, value: object) -> Parser[str, Any]:
    return spaced.regex(rf"{word}\b") >> (lambda _: value)


def _punctuation(symbol: str) -> Parser[str, Any]:
    return -spaced.lit(symbol)


value: ForwardDeclParser[str, Any] = forward_decl()
string = spaced.regex(STRING) >> unescape
member = string + _punctuation(":") + value >> tuple
members = maybe(member + many(_punctuation(",") + member)) >> _array
values = maybe(value + many(_punctuation(",") + value)) >> _array
value.define(
    string
    | spaced.regex(NUMBER) >> number
    | _punctuation("{") + members + _punctuation("}") >> dict
    | _punctuation("[") + values + _punctuation("]")
    | _keyword("true", True)
    | _keyword("false", False)
    | _keyword("null", None)
)
GRAMMAR = value + -spaced.end
COMPILED = GRAMMAR.compile()
