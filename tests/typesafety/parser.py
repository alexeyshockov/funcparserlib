"""What a downstream type checker infers for the parsing combinators.

The assertions are static, made by `just types-api`; see this directory's conftest.
"""

from collections.abc import Callable, Sequence
from typing import Any, assert_type

from funcparserlib import binary as b
from funcparserlib import text
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
from funcparserlib.text import Text
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
spaced = Text(skip=r"\s+")
_ = assert_type(spaced.regex("a"), Parser[str, str])
_ = assert_type(spaced.lit("a"), Parser[str, str])
_ = assert_type(spaced.end, Parser[str, None])
_ = assert_type(spaced.regex("a") + -spaced.end, Parser[str, str])
_ = assert_type(spaced.regex("a") + spaced.lit("b"), Tuple2Parser[str, str, str])
_ = assert_type(spaced.regex("a") >> len, Parser[str, int])
_ = assert_type(text.regex("a"), Parser[str, str])
_ = assert_type(text.lit("a"), Parser[str, str])
_ = assert_type(text.regex("a") + -finished, Parser[str, str])

# `longest()` keeps the type of its parsers.
_ = assert_type(longest(x, x), Parser[str, str])
_ = assert_type(longest(n, n, n), Parser[Token, int])

# Parsing text.
_ = assert_type(x.parse_text("x"), str)
_ = assert_type(spaced.regex("a").parse_text("a"), str)
_ = assert_type((spaced.regex("a") + spaced.regex("b")).parse_text("ab"), tuple[str, str])
_ = assert_type(spaced.regex("a").compile().parse_text("a"), str)
_ = assert_type(spaced.regex("a").compile().try_parse("a"), str | None)
_ = assert_type(n.compile(make_tokenizer([])).try_parse(""), int | None)

# Bytes: `funcparserlib.binary` parsers read `int` bytes, and `prefixed()`, `counted()` and
# `switch()` keep the type of what they read.
_ = assert_type(b.byte(), Parser[int, int])
_ = assert_type(b.regex(b"a+"), Parser[int, bytes])
_ = assert_type(b.lit(b"ab"), Parser[int, bytes])
_ = assert_type(b.fixed(">H"), Parser[int, Any])  # pyright: ignore[reportExplicitAny] -- `struct` values are `Any`
_ = assert_type(b.previous(), Parser[int, int])
_ = assert_type(b.previous(0x0F), Parser[int, int])
_ = assert_type(b.byte() + b.regex(b"a"), Tuple2Parser[int, int, bytes])
_ = assert_type(b.byte() + -finished, Parser[int, int])
_ = assert_type(many(b.byte()), Parser[int, list[int]])

_ = assert_type(prefixed(b.byte()), Parser[int, bytes])
_ = assert_type(prefixed(b.byte(), many(b.byte())), Parser[int, list[int]])
_ = assert_type(prefixed(b.byte(), b.regex(b"a")), Parser[int, bytes])
_ = assert_type(prefixed(b.fixed(">H"), b.byte() + b.byte()), Parser[int, tuple[int, int]])
_ = assert_type(prefixed(spaced.regex("a") >> len), Parser[str, str])
_ = assert_type(prefixed(spaced.regex("a") >> len, spaced.regex("b")), Parser[str, str])

_ = assert_type(counted(b.byte(), b.regex(b"a")), Parser[int, list[bytes]])
_ = assert_type(counted(b.byte(), prefixed(b.byte())), Parser[int, list[bytes]])
_ = assert_type(counted(spaced.regex("a") >> len, spaced.regex("b")), Parser[str, list[str]])

# The cases of `switch()` usually give different values, so it gives `Any`.
mixed: Parser[int, bytes | int] = switch(b.byte(), {1: b.regex(b"a"), 2: b.byte()})
ranges: Parser[int, int] = switch(b.byte(), {range(1, 4): b.byte(), 0: b.byte()})
keys: Parser[str, str] = switch(spaced.regex("a"), {"a": spaced.regex("b")})

# Parsing bytes.
_ = assert_type(b.byte().parse_bytes(b"a"), int)
_ = assert_type((b.byte() + b.regex(b"a")).parse_bytes(b"ab"), tuple[int, bytes])
_ = assert_type(prefixed(b.byte()).parse_bytes(b"\x01a"), bytes)
_ = assert_type(counted(b.byte(), b.byte()).parse_bytes(b"\x01a"), list[int])
_ = assert_type(b.byte().compile().parse_bytes(b"a"), int)
_ = assert_type(b.byte().compile().try_parse(b"a"), int | None)
_ = assert_type(b.byte().compile().try_parse(b""), int | None)
_ = assert_type(prefixed(b.byte()).compile().parse_bytes(b"\x01a"), bytes)
_ = assert_type(prefixed(b.byte()).compile().try_parse(b"\x01a"), bytes | None)
