"""What a downstream type checker infers for the parsing combinators.

The assertions are static, made by `just types-api`; see this directory's conftest.
"""

from collections.abc import Callable, Sequence
from typing import Any, assert_type

from funcparserlib import binary
from funcparserlib.lexer import Token, make_tokenizer
from funcparserlib.parser import (
    CompiledParser,
    ForwardDeclParser,
    IgnoredParser,
    Parser,
    ParsingResult,
    ParsingSuccess,
    State,
    Tuple2Parser,
    Tuple3Parser,
    Tuple4Parser,
    Tuple5Parser,
    a,
    counted,
    finished,
    forward_decl,
    longest,
    many,
    maybe,
    oneplus,
    prefixed,
    skip,
    switch,
    tok,
)
from funcparserlib.text import Text, literal, match
from funcparserlib.util import expand_tuple_args

x = a("x")
n = tok("number") >> int
_ = assert_type(x, Parser[str, str])
_ = assert_type(n, Parser[Token, int])


def one(_tokens: Sequence[str], s: State) -> ParsingResult[int]:
    return ParsingSuccess(1, s)


# `Parser(f)` takes its types from the parsing function.
_ = assert_type(Parser(one), Parser[str, int])

# `+` builds a tuple type per arity.
_ = assert_type(x + x, Tuple2Parser[str, str, str])
_ = assert_type(x + x + x, Tuple3Parser[str, str, str, str])
_ = assert_type((x + x).parse("xx"), tuple[str, str])

# `-p` and `skip(p)` drop their value from the sequence.
_ = assert_type(-x, IgnoredParser[str])
_ = assert_type(skip(x), IgnoredParser[str])
_ = assert_type(-x + x, Parser[str, str])
_ = assert_type(x + -x, Parser[str, str])

# `|` is a union of both values; `>>` is the function's result.
_ = assert_type(tok("id") | n, Parser[Token, str | int])
_ = assert_type(x >> len, Parser[str, int])


def join(xy: tuple[str, str]) -> str:
    return xy[0] + xy[1]


_ = assert_type((x + x) >> join, Parser[str, str])

_ = assert_type(n.compile(), CompiledParser[Token, int])
_ = assert_type(n.compile().parse([]), int)
_ = assert_type(n.recover(n), Parser[Token, int | None])
_ = assert_type(n.recover(default=0), Parser[Token, int])
_ = assert_type(n.recover(n, default="broken"), Parser[Token, int | str])
_ = assert_type(n.validated(str), Parser[Token, str])
_ = assert_type(n.validated(str, "a number"), Parser[Token, str])
_ = assert_type(longest(x, x + x), Parser[str, str | tuple[str, str]])  # ty: ignore[type-assertion-failure] -- ty infers the fallback overload; basedpyright gates this file
_ = assert_type(n.compile(make_tokenizer([])).parse_text(""), int)
_ = assert_type((x + x).compile().parse("xx"), tuple[str, str])

_ = assert_type(many(n), Parser[Token, list[int]])
_ = assert_type(oneplus(n), Parser[Token, list[int]])
_ = assert_type(maybe(n), Parser[Token, int | None])

decl: ForwardDeclParser[str, str] = forward_decl()

# A sequence with a forward declaration or the end of input is a plain parser: no `define()`.
_ = assert_type(-x + decl, Parser[str, str])
_ = assert_type(decl + -x, Parser[str, str])
_ = assert_type(-x + finished, Parser[str, None])
_ = assert_type(-finished + decl, Parser[str, str])


def add(left: int, right: str) -> float:
    return left + len(right)


_ = assert_type(expand_tuple_args(add), Callable[[tuple[int, str]], float])

# `+ -finished` drops the end-of-input value for every left-hand side; `+ finished` keeps it.
_ = assert_type(x + -finished, Parser[str, str])
_ = assert_type(many(n) + -finished, Parser[Token, list[int]])
_ = assert_type(x + x + -finished, Tuple2Parser[str, str, str])
_ = assert_type(x + x + x + -finished, Tuple3Parser[str, str, str, str])
_ = assert_type(x + x + x + x + -finished, Tuple4Parser[str, str, str, str, str])
_ = assert_type(x + x + x + x + x + -finished, Tuple5Parser[str, str, str, str, str, str])
_ = assert_type(-x + -finished, IgnoredParser[str])
_ = assert_type(-finished + x, Parser[str, str])
_ = assert_type(x + finished, Tuple2Parser[str, str, None])

# Text parsers read `str` and give the text they matched; `end` gives nothing.
text = Text(skip=r"\s+")
_ = assert_type(text.match("a"), Parser[str, str])
_ = assert_type(text.literal("a"), Parser[str, str])
_ = assert_type(text.end, Parser[str, None])
_ = assert_type(text.match("a") + -text.end, Parser[str, str])
_ = assert_type(text.match("a") + text.literal("b"), Tuple2Parser[str, str, str])
_ = assert_type(text.match("a") >> len, Parser[str, int])
_ = assert_type(match("a"), Parser[str, str])
_ = assert_type(literal("a"), Parser[str, str])
_ = assert_type(match("a") + -finished, Parser[str, str])

# `longest()` keeps the type of its parsers.
_ = assert_type(longest(x, x), Parser[str, str])
_ = assert_type(longest(n, n, n), Parser[Token, int])

# Parsing text.
_ = assert_type(x.parse_text("x"), str)
_ = assert_type(text.match("a").parse_text("a"), str)
_ = assert_type((text.match("a") + text.match("b")).parse_text("ab"), tuple[str, str])
_ = assert_type(text.match("a").compile().parse_text("a"), str)
_ = assert_type(text.match("a").compile().try_parse("a"), str | None)
_ = assert_type(n.compile(make_tokenizer([])).try_parse(""), int | None)

# Bytes: `funcparserlib.binary` parsers read `int` bytes, and `prefixed()`, `counted()` and
# `switch()` keep the type of what they read.
_ = assert_type(binary.byte(), Parser[int, int])
_ = assert_type(binary.match(b"a+"), Parser[int, bytes])
_ = assert_type(binary.literal(b"ab"), Parser[int, bytes])
_ = assert_type(binary.fixed(">H"), Parser[int, Any])  # pyright: ignore[reportExplicitAny] -- `struct` values are `Any`
_ = assert_type(binary.previous(), Parser[int, int])
_ = assert_type(binary.previous(0x0F), Parser[int, int])
_ = assert_type(binary.byte() + binary.match(b"a"), Tuple2Parser[int, int, bytes])
_ = assert_type(binary.byte() + -finished, Parser[int, int])
_ = assert_type(many(binary.byte()), Parser[int, list[int]])

_ = assert_type(prefixed(binary.byte()), Parser[int, bytes])
_ = assert_type(prefixed(binary.byte(), many(binary.byte())), Parser[int, list[int]])
_ = assert_type(prefixed(binary.byte(), binary.match(b"a")), Parser[int, bytes])
_ = assert_type(
    prefixed(binary.fixed(">H"), binary.byte() + binary.byte()), Parser[int, tuple[int, int]]
)
_ = assert_type(prefixed(text.match("a") >> len), Parser[str, str])
_ = assert_type(prefixed(text.match("a") >> len, text.match("b")), Parser[str, str])

_ = assert_type(counted(binary.byte(), binary.match(b"a")), Parser[int, list[bytes]])
_ = assert_type(counted(binary.byte(), prefixed(binary.byte())), Parser[int, list[bytes]])
_ = assert_type(counted(text.match("a") >> len, text.match("b")), Parser[str, list[str]])

# The cases of `switch()` usually give different values, so it gives `Any`.
mixed: Parser[int, bytes | int] = switch(binary.byte(), {1: binary.match(b"a"), 2: binary.byte()})
ranges: Parser[int, int] = switch(binary.byte(), {range(1, 4): binary.byte(), 0: binary.byte()})
keys: Parser[str, str] = switch(text.match("a"), {"a": text.match("b")})

# Parsing bytes.
_ = assert_type(binary.byte().parse_bytes(b"a"), int)
_ = assert_type((binary.byte() + binary.match(b"a")).parse_bytes(b"ab"), tuple[int, bytes])
_ = assert_type(prefixed(binary.byte()).parse_bytes(b"\x01a"), bytes)
_ = assert_type(counted(binary.byte(), binary.byte()).parse_bytes(b"\x01a"), list[int])
_ = assert_type(binary.byte().compile().parse_bytes(b"a"), int)
_ = assert_type(binary.byte().compile().try_parse(b"a"), int | None)
_ = assert_type(binary.byte().compile().try_parse(b""), int | None)
_ = assert_type(prefixed(binary.byte()).compile().parse_bytes(b"\x01a"), bytes)
_ = assert_type(prefixed(binary.byte()).compile().try_parse(b"\x01a"), bytes | None)
