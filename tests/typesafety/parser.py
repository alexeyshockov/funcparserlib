"""What a downstream type checker infers for the parsing combinators.

The assertions are static, made by `just types-api`; see this directory's conftest.
"""

from collections.abc import Callable
from typing import assert_type

from funcparserlib.lexer import Token
from funcparserlib.parser import (
    ForwardDeclParser,
    IgnoredParser,
    Parser,
    Tuple2Parser,
    Tuple3Parser,
    Tuple4Parser,
    Tuple5Parser,
    a,
    finished,
    forward_decl,
    many,
    maybe,
    oneplus,
    skip,
    tok,
)
from funcparserlib.util import expand_tuple_args

x = a("x")
n = tok("number") >> int
_ = assert_type(x, Parser[str, str])
_ = assert_type(n, Parser[Token, int])

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

_ = assert_type(many(n), Parser[Token, list[int]])
_ = assert_type(oneplus(n), Parser[Token, list[int]])
_ = assert_type(maybe(n), Parser[Token, int | None])

decl: ForwardDeclParser[str, str] = forward_decl()

# A sequence with a forward declaration or the end of input is a plain parser: no `define()`.
_ = assert_type(-x + decl, Parser[str, str])
_ = assert_type(decl + -x, Parser[str, str])
_ = assert_type(-x + finished, Parser[str, None])


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
