"""One combinator each, over an input long enough that the combinator dominates the time."""

from typing import Any

from funcparserlib.lexer import Token
from funcparserlib.parser import Parser, a, finished, forward_decl, many, maybe, some, tok

N = 10_000
XS = "x" * N
TOKENS = [Token("name", f"n{i}") for i in range(N)]


def bench_many(benchmark: Any) -> None:
    assert benchmark((many(a("x")) + -finished).parse, XS) == list(XS)


def bench_alternatives(benchmark: Any) -> None:
    letters = "abcdefghijklmnopqrstuvwxyz"
    any_letter = a(letters[0])
    for c in letters[1:]:
        any_letter = any_letter | a(c)
    text = letters[-1] * N
    assert benchmark((many(any_letter) + -finished).parse, text) == list(text)


def bench_sequence(benchmark: Any) -> None:
    triple = a("x") + -a("y") + a("z")
    text = "xyz" * (N // 3)
    assert len(benchmark((many(triple) + -finished).parse, text)) == N // 3


def bench_tok(benchmark: Any) -> None:
    names = many(tok("name")) + -finished
    assert len(benchmark(names.parse, TOKENS)) == N


def bench_some_with_map(benchmark: Any) -> None:
    digits = many(some(str.isdigit) >> int) + -finished
    assert sum(benchmark(digits.parse, "7" * N)) == 7 * N


def bench_maybe(benchmark: Any) -> None:
    pairs = many(a("x") + maybe(a("y"))) + -finished
    assert len(benchmark(pairs.parse, "xy" * (N // 2))) == N // 2


def bench_forward_decl_recursion(benchmark: Any) -> None:
    nested: Parser[str, Any] = forward_decl()
    nested.define(-a("(") + maybe(nested) + -a(")"))  # type: ignore[attr-defined]
    depth = 300
    assert benchmark((nested + -finished).parse, "(" * depth + ")" * depth) is None
