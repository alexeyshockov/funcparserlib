from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from funcparserlib.parser import (
    NoParseError,
    Parser,
    a,
    anything_but,
    finished,
    many,
    maybe,
    oneplus,
    pure,
    some,
)
from funcparserlib.util import expand_tuple_args, pretty_tree


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


# Properties of the combinators, over inputs of single characters.

CHARS = st.characters(codec="ascii", categories=["L", "N", "P"])


@given(st.lists(st.tuples(CHARS, st.booleans()), min_size=1, max_size=8))
def test_a_sequence_is_the_tuple_of_what_it_keeps(parts: list[tuple[str, bool]]) -> None:
    assume(any(keep for _, keep in parts))
    parsers = [a(c) if keep else -a(c) for c, keep in parts]
    expr = parsers[0]
    for p in parsers[1:]:
        expr = expr + p  # type: ignore[operator]
    kept = tuple(c for c, keep in parts if keep)
    expected = kept if len(kept) > 1 else kept[0]
    assert expr.parse("".join(c for c, _ in parts)) == expected  # type: ignore[union-attr]


@given(st.integers(min_value=0, max_value=50), st.text(alphabet="yz"))
def test_many_takes_every_repetition_and_leaves_the_rest(n: int, rest: str) -> None:
    expr = many(a("x")) + many(some(lambda _: True))
    assert expr.parse("x" * n + rest) == (["x"] * n, list(rest))


@given(st.integers(min_value=0, max_value=50))
def test_oneplus_is_many_without_the_empty_case(n: int) -> None:
    if n:
        assert oneplus(a("x")).parse("x" * n) == ["x"] * n
    else:
        with pytest.raises(NoParseError):
            oneplus(a("x")).parse("")


@given(CHARS)
def test_alternatives_prefer_the_left_one(c: str) -> None:
    left_wins = (a(c) >> (lambda _: "left")) | (some(lambda _: True) >> (lambda _: "right"))
    assert left_wins.parse(c) == "left"


@given(st.text(alphabet="xyz", min_size=1, max_size=10), st.text(alphabet="xyz", max_size=12))
def test_the_error_names_the_first_token_that_differs(grammar: str, text: str) -> None:
    assume(text != grammar)
    expr = a(grammar[0])
    for c in grammar[1:]:
        expr = expr + a(c)  # type: ignore[assignment]
    expr = expr + finished  # type: ignore[assignment]
    k = next((i for i, (g, t) in enumerate(zip(grammar, text, strict=False)) if g != t), None)
    if k is None:
        k = min(len(grammar), len(text))
    with pytest.raises(NoParseError) as e:
        expr.parse(text)
    expected = (
        "got unexpected end of input" if k >= len(text) else f"got unexpected token: {text[k]!r}"
    )
    assert e.value.msg.startswith(expected)


@given(st.tuples(st.integers(), st.text(), st.booleans()))
def test_expand_tuple_args_passes_the_tuple_as_arguments(args: tuple[int, str, bool]) -> None:
    assert expand_tuple_args(lambda x, y, z: (z, y, x))(args) == args[::-1]


TREES = st.recursive(st.integers(), lambda kids: st.lists(kids, max_size=4), max_leaves=20)


@given(TREES)
def test_pretty_tree_draws_a_line_per_node(tree: object) -> None:
    def kids(x: object) -> list[object]:
        return x if isinstance(x, list) else []

    def count(x: object) -> int:
        return 1 + sum(count(k) for k in kids(x))

    drawn = pretty_tree(tree, kids, lambda x: "[]" if isinstance(x, list) else str(x))
    assert len(drawn.splitlines()) == count(tree)
