from typing import Any

import pytest

from funcparserlib.parser import (
    NoParseError,
    Parser,
    a,
    anything_but,
    many,
    maybe,
    oneplus,
    pure,
    some,
)
from funcparserlib.util import expand_tuple_args


def test_map_is_rshift() -> None:
    assert a("x").map(str.upper).parse("x") == "X"


def test_starmap_spreads_the_tuple() -> None:
    expr = (a("1") + a("2")).starmap(lambda x, y: int(x) + int(y))
    assert expr.parse("12") == 3


def test_when_accepts_what_the_predicate_accepts() -> None:
    digit = some(str.isdigit).named("digit").when(lambda d: d != "0")
    assert digit.parse("7") == "7"
    with pytest.raises(NoParseError, match=r"got unexpected token: '0', expected: when\(\.\.\.\)"):
        digit.parse("0")


def test_when_reports_the_inner_failure() -> None:
    digit = some(str.isdigit).named("digit").when(lambda d: d != "0")
    with pytest.raises(NoParseError, match="got unexpected token: 'x', expected: digit"):
        digit.parse("x")


def test_but_not_rejects_what_the_other_parser_accepts() -> None:
    name = some(str.isalpha).named("name").but_not(a("if"))
    assert name.parse(["x"]) == "x"
    with pytest.raises(NoParseError, match=r"got unexpected token: 'if', expected: name \(but not"):
        name.parse(["if"])


def test_anything_but_takes_one_other_token() -> None:
    expr = a("a") + anything_but(a("x"))
    assert expr.parse("ay") == ("a", "y")
    with pytest.raises(NoParseError, match="got unexpected token: 'x', expected: anything but"):
        expr.parse("ax")


def test_anything_but_fails_at_end_of_input() -> None:
    with pytest.raises(NoParseError, match="got unexpected end of input, expected: anything but"):
        (a("a") + ~a("x")).parse("a")


@pytest.mark.parametrize(
    "expr",
    [many(maybe(a("x"))), many(pure(1)), oneplus(maybe(a("x"))), many(many(a("x")))],
    ids=["many(maybe)", "many(pure)", "oneplus(maybe)", "many(many)"],
)
def test_many_over_a_parser_that_matches_nothing_raises(expr: Parser[str, Any]) -> None:
    with pytest.raises(ValueError, match="matched no tokens inside many"):
        expr.parse("xxy")


def test_expand_tuple_args_for_a_function() -> None:
    expr = (a("1") + a("2")) >> expand_tuple_args(lambda x, y: x + y)
    assert expr.parse("12") == "12"


def test_expand_tuple_args_for_a_method_keeps_self() -> None:
    class Pair:
        @expand_tuple_args
        def join(self, x: str, y: str) -> str:
            return f"{x}-{y}"

    pair = Pair()
    # The overloads type a function; checkers do not bind `self` through the returned callable.
    assert pair.join(("a", "b")) == "a-b"  # type: ignore[call-arg, arg-type]  # ty: ignore[too-many-positional-arguments]
