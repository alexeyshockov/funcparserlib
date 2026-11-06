"""The JSON grammar of `tests/json.py`, written with parsy, pyparsing and lark, for comparison.

Each follows its library's own JSON example. `unescape()` decodes strings as `tests/json.py` does,
so that the comparison measures the parsing, and each returns what `json.loads()` returns. The lark
grammar runs as LALR(1), with its contextual lexer and the transformer applied during the parse.
"""

import re
from typing import Any

import lark
import parsy
import pyparsing as pp

NUMBER = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
STRING = r'"(?:[^"\\]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))*"'
_ESCAPE = re.compile(r'\\(?:(["\\/bfnrt])|u([0-9A-Fa-f]{4}))')
_STANDARD = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def unescape(quoted: str) -> str:
    """The text of a JSON string literal, quotes removed and escapes decoded."""

    def sub(m: re.Match[str]) -> str:
        return _STANDARD[m.group(1)] if m.group(1) else chr(int(m.group(2), 16))

    # A `\uXXXX` pair encodes one character outside the BMP; a lone surrogate stays as is.
    decoded = _ESCAPE.sub(sub, quoted[1:-1])
    return decoded.encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass")


def number(text: str) -> int | float:
    """A JSON number as `json.loads()` returns it: an int unless it has a fraction or exponent."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def _parsy_grammar() -> Any:
    whitespace = parsy.regex(r"\s*")

    def lexeme(p: Any) -> Any:
        return p << whitespace

    def symbol(s: str) -> Any:
        return lexeme(parsy.string(s))

    value = parsy.forward_declaration()
    string = lexeme(parsy.regex(STRING)).map(unescape)
    member = parsy.seq(string << symbol(":"), value).map(tuple)
    obj = symbol("{") >> member.sep_by(symbol(",")).map(dict) << symbol("}")
    array = symbol("[") >> value.sep_by(symbol(",")) << symbol("]")
    value.become(
        string
        | lexeme(parsy.regex(NUMBER)).map(number)
        | obj
        | array
        | symbol("true").result(True)
        | symbol("false").result(False)
        | symbol("null").result(None)
    )
    return whitespace >> value


def _pyparsing_grammar() -> Any:
    lbrack, rbrack, lbrace, rbrace, colon = map(pp.Suppress, "[]{}:")
    value = pp.Forward()
    string = pp.Regex(STRING).set_parse_action(lambda t: unescape(t[0]))
    member = pp.Group(string + colon + value)
    obj = (lbrace + pp.Optional(pp.DelimitedList(member)) + rbrace).set_parse_action(
        lambda t: [{m[0]: m[1] for m in t}]
    )
    array = (lbrack + pp.Optional(pp.DelimitedList(value)) + rbrack).set_parse_action(
        lambda t: [list(t)]
    )
    value <<= (
        string
        | pp.Regex(NUMBER).set_parse_action(lambda t: number(t[0]))
        | obj
        | array
        | pp.Keyword("true").set_parse_action(pp.replace_with(True))
        | pp.Keyword("false").set_parse_action(pp.replace_with(False))
        | pp.Keyword("null").set_parse_action(pp.replace_with(None))
    )
    return value


class _LarkValues(lark.Transformer[Any, Any]):
    array = list
    pair = tuple
    object = dict

    def string(self, children: list[str]) -> str:
        return unescape(children[0])

    def number(self, children: list[str]) -> int | float:
        return number(children[0])

    def true(self, _: list[Any]) -> bool:
        return True

    def false(self, _: list[Any]) -> bool:
        return False

    def null(self, _: list[Any]) -> None:
        return None


def _lark_grammar() -> lark.Lark:
    # A `/` ends a regular expression in a lark grammar.
    string = STRING.replace("/", r"\/")
    grammar = rf"""
        ?start: value
        ?value: object
              | array
              | string
              | NUMBER -> number
              | "true" -> true
              | "false" -> false
              | "null" -> null
        array: "[" (value ("," value)*)? "]"
        object: "{{" (pair ("," pair)*)? "}}"
        pair: string ":" value
        string: STRING
        NUMBER: /{NUMBER}/
        STRING: /{string}/
        %ignore /[ \t\r\n]+/
    """
    return lark.Lark(grammar, parser="lalr", transformer=_LarkValues())


PARSY = _parsy_grammar()
PYPARSING = _pyparsing_grammar()
LARK = _lark_grammar()


def parsy_loads(text: str) -> object:
    """`text` parsed by the parsy grammar."""
    return PARSY.parse(text)


def pyparsing_loads(text: str) -> object:
    """`text` parsed by the pyparsing grammar."""
    return PYPARSING.parse_string(text, parse_all=True)[0]


def lark_loads(text: str) -> object:
    """`text` parsed by the lark grammar."""
    return LARK.parse(text)
