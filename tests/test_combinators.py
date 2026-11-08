import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, TypeAlias

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from funcparserlib.lexer import Token
from funcparserlib.parser import (
    NoParseError,
    Parser,
    ParsingFailure,
    ParsingResult,
    ParsingSuccess,
    State,
    _alternatives,
    a,
    anything_but,
    finished,
    forward_decl,
    many,
    maybe,
    oneplus,
    parser,
    pure,
    separated,
    some,
    tok,
)
from funcparserlib.util import expand_tuple_args, pretty_tree
from tests.rules import kept

P: TypeAlias = Parser[Any, Any]


def test_map_is_rshift() -> None:
    assert a("x").map(str.upper).parse("x") == "X"


def test_starmap_spreads_the_tuple() -> None:
    expr = (a("1") + a("2")).starmap(lambda x, y: int(x) + int(y))
    assert expr.parse("12") == 3


def test_when_accepts_what_the_predicate_accepts() -> None:
    digit = some(str.isdigit).named("digit").when(lambda d: d != "0")
    assert digit.parse("7") == "7"
    with pytest.raises(NoParseError, match=r"got unexpected token: '0', expected: digit$"):
        digit.parse("0")


def test_when_reports_the_inner_failure() -> None:
    digit = some(str.isdigit).named("digit").when(lambda d: d != "0")
    with pytest.raises(NoParseError, match="got unexpected token: 'x', expected: digit"):
        digit.parse("x")


@pytest.mark.parametrize(
    ("p", "text", "expected"),
    [
        ((a("x") + a("y")).when(lambda _: False), "xy", "when(...)"),
        ((a("x") + a("y")).but_not(a("x")), "xy", "but_not(...)"),
        (anything_but(a("x") + a("y")), "xy", "anything else"),
        ((a("x") >> str.upper).when(lambda _: False), "x", "'x'"),
        (tok("number").when(lambda _: False), [Token("number", "1")], "number"),
        (some(str.isdigit).but_not(a("1") | a("2")), "1", "some(...) (but not ...)"),
    ],
    ids=["when, sequence", "but_not, sequence", "~, sequence", "when, map", "when, tok", "but_not"],
)
def test_a_predicate_combinator_names_what_its_parser_parses_never_its_shape(
    p: Parser[Any, Any], text: Any, expected: str
) -> None:
    with pytest.raises(NoParseError) as e:
        p.parse(text)
    assert e.value.expected == (expected,)


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
        (a("a") + anything_but(a("x"))).parse("a")


@pytest.mark.parametrize(
    "expr",
    [many(maybe(a("x"))), many(pure(1)), oneplus(maybe(a("x"))), many(many(a("x")))],
    ids=["many(maybe)", "many(pure)", "oneplus(maybe)", "many(many)"],
)
def test_many_over_a_parser_that_matches_nothing_raises(expr: Parser[str, Any]) -> None:
    with pytest.raises(ValueError, match="matched no tokens inside many"):
        expr.parse("xxy")


def test_expand_tuple_args_for_a_function() -> None:
    # The lambda takes its types from the tuple it is given, which mypy does not see.
    add = expand_tuple_args(lambda x, y: x + y)  # type: ignore[operator]
    expr = (a("1") + a("2")) >> add
    assert expr.parse("12") == "12"


def test_expand_tuple_args_for_a_method_keeps_self() -> None:
    class Pair:
        @expand_tuple_args
        def join(self, x: str, y: str) -> str:  # type: ignore[misc]
            return f"{x}-{y}"

    pair = Pair()
    # The overloads type a function; checkers do not bind `self` through the returned callable.
    assert pair.join(("a", "b")) == "a-b"  # type: ignore[call-arg, misc]  # ty: ignore[too-many-positional-arguments]


# Results compose as the parsers that made them.


@pytest.mark.parametrize(
    ("first", "second", "text"),
    [
        (a("x"), a("y"), "xy"),
        (a("x") + a("y"), a("z"), "xyz"),
        (-a("x"), a("y"), "xy"),
        (a("x"), -a("y"), "xy"),
        (a("x"), a("y"), "xz"),
        (a("x"), a("y"), "wy"),
    ],
    ids=[
        "two values",
        "a tuple grows",
        "ignored left",
        "ignored right",
        "failure right",
        "failure left",
    ],
)
def test_the_sum_of_results_is_the_result_of_the_sum_of_parsers(
    first: P, second: P, text: str
) -> None:
    left = first.run(text, State(0, 0))
    right = second.run(text, left.state)
    assert left + right == (first + second).run(text, State(0, 0))


# Properties of the combinators, over inputs of single characters.

CHARS = st.characters(codec="ascii", categories=["L", "N", "P"])


@given(st.lists(st.tuples(CHARS, st.booleans()), min_size=1, max_size=8))
def test_a_sequence_is_the_tuple_of_what_it_keeps(parts: list[tuple[str, bool]]) -> None:
    assume(any(keep for _, keep in parts))
    parsers = [a(c) if keep else -a(c) for c, keep in parts]
    expr = parsers[0]
    for p in parsers[1:]:
        expr = expr + p
    kept = tuple(c for c, keep in parts if keep)
    expected = kept if len(kept) > 1 else kept[0]
    assert expr.parse("".join(c for c, _ in parts)) == expected


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


@given(st.integers(min_value=0, max_value=20))
def test_separated_gives_each_item_once(n: int) -> None:
    assert separated(a("x"), a(",")).parse(",".join("x" * n)) == ["x"] * n


@given(st.integers(min_value=1, max_value=20))
def test_separated_keeps_a_tuple_as_one_item(n: int) -> None:
    pair = a("x") + a("y")
    assert separated(pair, a(",")).parse(",".join(["xy"] * n)) == [("x", "y")] * n


def test_separated_leaves_a_trailing_separator_unless_told_to_read_it() -> None:
    assert (separated(a("x"), a(",")) + a(",")).parse("x,x,") == (["x", "x"], ",")
    assert (separated(a("x"), a(","), trailing=True) + -finished).parse("x,x,") == ["x", "x"]


def test_separated_reads_only_one_trailing_separator() -> None:
    with pytest.raises(NoParseError):
        (separated(a("x"), a(","), trailing=True) + -finished).parse("x,,")


def test_separated_gives_back_a_separator_when_the_item_after_it_fails_inside() -> None:
    items = separated(a("x") + a("y"), a(","))
    assert (items + a(",") + a("x")).parse("xy,x") == ([("x", "y")], ",", "x")


def test_separated_reads_no_lone_separator() -> None:
    with pytest.raises(NoParseError):
        (separated(a("x"), a(","), trailing=True) + -finished).parse(",")


@given(CHARS)
def test_alternatives_prefer_the_left_one(c: str) -> None:
    left_wins = (a(c) >> (lambda _: "left")) | (some(lambda _: True) >> (lambda _: "right"))
    assert left_wins.parse(c) == "left"


@given(st.text(alphabet="xyz", min_size=1, max_size=10), st.text(alphabet="xyz", max_size=12))
def test_the_error_names_the_first_token_that_differs(grammar: str, text: str) -> None:
    assume(text != grammar)
    expr = a(grammar[0])
    for c in grammar[1:]:
        expr = expr + a(c)
    expr = expr + finished
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


# A parsing function written for 1.0: a `(value, state)` tuple, or a raised `NoParseError`.


def legacy_digit(tokens: Any, s: State) -> Any:
    if s.pos < len(tokens) and tokens[s.pos].isdigit():
        return tokens[s.pos], State(s.pos + 1, max(s.pos + 1, s.max))
    raise NoParseError("got unexpected token", s)


def test_a_legacy_parsing_function_parses_and_backtracks() -> None:
    expr = many(Parser(legacy_digit) | a("x"))
    with pytest.warns(DeprecationWarning, match="legacy_digit follows the 1.0 protocol"):
        assert expr.parse("1x2") == ["1", "x", "2"]


def test_a_legacy_parsing_function_warns_once_per_parser() -> None:
    digits = many(Parser(legacy_digit))
    with pytest.warns(DeprecationWarning, match="follows the 1.0 protocol") as record:
        digits.parse("123")
    assert len(record) == 1


def test_a_legacy_parsing_function_keeps_what_its_parsers_expected() -> None:
    def legacy_pair(tokens: Any, s: State) -> Any:
        v1, s1 = a("x").run(tokens, s)
        v2, s2 = a("y").run(tokens, s1)
        return (v1, v2), s2

    with (
        pytest.warns(DeprecationWarning, match="follows the 1.0 protocol"),
        pytest.raises(NoParseError) as e,
    ):
        (Parser(legacy_pair) | a("z")).parse("xw")
    assert e.value.expected == ("'y'",)


def test_a_parsing_function_returning_a_result_does_not_warn() -> None:
    @parser("digit")
    def digit(tokens: Sequence[str], s: State) -> ParsingResult[str]:
        if s.pos < len(tokens) and tokens[s.pos].isdigit():
            return ParsingSuccess(tokens[s.pos], s.reset_pos(s.pos + 1))
        return ParsingFailure(s.expect(digit))

    assert many(digit).parse("12") == ["1", "2"]


def test_a_parsing_function_of_parser_raises_as_it_is() -> None:
    @parser("digit")
    def digit(tokens: Sequence[str], s: State) -> ParsingResult[str]:
        raise NoParseError("no digits here", s)

    with pytest.raises(NoParseError, match=r"^no digits here$"):
        (digit | a("x")).parse("x")


def test_a_legacy_parsing_function_warns_once_per_function_by_default() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for _ in range(3):
            Parser(legacy_digit).parse("1")
    assert len(record) == 1


@pytest.mark.interpreted_only
def test_a_skipped_parser_still_converts_its_value() -> None:
    seen: list[str] = []
    expr = -(a("x") >> seen.append) + a("y")
    assert expr.parse("xy") == "y"
    assert seen == ["x"]


def either(p: Parser[str, Any], q: Parser[str, Any]) -> Parser[str, Any]:
    """The choice the `parser()` docs show."""

    @parser(None)
    def tried(tokens: Sequence[str], s: State) -> ParsingResult[Any]:
        res = p.run(tokens, s)
        if not isinstance(res, ParsingFailure) or res.committed:
            return res
        st = res.state
        return q.run(tokens, replace(s, max=st.max, expected=st.expected))

    return tried


@pytest.mark.parametrize("text", ["xy", "xz", "xw", "x", "w", ""])
def test_a_parsing_function_that_backtracks_reads_as_the_choice_does(text: str) -> None:
    def outcome(p: Parser[str, Any]) -> object:
        try:
            return p.parse(text)
        except NoParseError as e:
            return e.msg

    x, y, z = a("x"), a("y"), a("z")
    assert outcome(either(x + y, x + z)) == outcome((x + y) | (x + z))


def test_a_parsing_function_that_backtracks_drops_the_failed_branch_rules() -> None:
    x = a("x").validated(kept(lambda _: False, "bad x"))
    assert either(x + a("y"), a("x") + a("z")).parse("xz") == ("x", "z")


def test_a_parsing_function_that_reads_run_sees_a_later_name() -> None:
    p = a("x") + a("y")

    @parser(None)
    def reads(tokens: Sequence[str], s: State) -> ParsingResult[Any]:
        return p.run(tokens, s)

    p.named("pair")
    with pytest.raises(NoParseError, match="expected: pair"):
        reads.parse("z")


def test_a_parsing_function_that_reads_run_sees_a_later_definition() -> None:
    declared = forward_decl()

    @parser(None)
    def reads(tokens: Sequence[str], s: State) -> ParsingResult[Any]:
        return declared.run(tokens, s)

    declared.define(a("x"))
    assert reads.parse("x") == "x"


def repeated(step: Callable[[P], P], levels: int = 20) -> P:
    """`step` applied `levels` times. Growth that doubles exhausts memory at about 33 levels."""
    grammar: P = tok("a")
    for _ in range(levels):
        grammar = step(grammar)
    return grammar


@pytest.mark.parametrize(
    "step",
    [
        lambda g: g + g,
        lambda g: g | g,
        lambda g: g | (g + tok("b")),
        lambda g: oneplus(g + tok("b")),
        lambda g: maybe(g) + g,
        lambda g: many(g) + g,
        lambda g: g.but_not(g),
    ],
    ids=["+", "|", "| +", "oneplus", "maybe", "many", "but_not"],
)
def test_a_composed_name_stays_bounded(step: Callable[[P], P]) -> None:
    name = repeated(step).name
    assert name is not None
    assert len(name) <= 200


def test_the_name_of_a_nested_maybe_stays_bounded() -> None:
    name = repeated(maybe, 60).name
    assert name is not None
    assert len(name) <= 200


def test_a_short_composed_name_is_whole() -> None:
    assert (tok("a") + tok("b")).name == "(a, b)"


def test_a_name_given_to_a_parser_is_whole() -> None:
    name = "x" * 300
    assert tok("a").named(name).name == name
    assert (tok("a").named(name) >> str).name == name


def test_a_parser_repeated_in_a_choice_counts_once() -> None:
    x, y = a("x"), a("y")
    assert _alternatives(x | y | x | (y | x)) == (x, y)


def test_a_choice_nested_in_itself_counts_each_parser_once() -> None:
    x = tok("a")
    grammar: P = x
    for _ in range(20):
        grammar = grammar | grammar
    assert _alternatives(grammar) == (x,)


@pytest.mark.parametrize(
    ("text", "value"), [("x", "x"), ("y", "y"), ("z", "z")], ids=["first", "repeated", "last"]
)
def test_a_repeated_alternative_changes_no_value(text: str, value: str) -> None:
    x, y, z = a("x"), a("y"), a("z")
    assert (x | y | x | z | y).parse(text) == value


def test_a_repeated_alternative_changes_no_error() -> None:
    x, y, z = a("x"), a("y"), a("z")
    repeated_choice = x | y | x | z | y
    with pytest.raises(NoParseError) as got:
        repeated_choice.parse("q")
    with pytest.raises(NoParseError) as expected:
        (x | y | z).parse("q")
    assert str(got.value) == str(expected.value)


def test_expand_tuple_args_of_a_built_in() -> None:
    built_in: Any = max
    assert expand_tuple_args(built_in)((1, 3, 2)) == 3


def test_expand_tuple_args_of_a_function_with_a_parameter_called_cls() -> None:
    def named(cls: str, name: str) -> str:
        return cls + name

    assert expand_tuple_args(named)(("a", "b")) == "ab"
    assert expand_tuple_args(lambda cls, name: cls + name)(("a", "b")) == "ab"  # type: ignore[operator]
