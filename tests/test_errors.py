import functools
import operator
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from funcparserlib import parser as parser_module
from funcparserlib.lexer import Token, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    NoParseError,
    ParseErrors,
    Parser,
    ParsingFailure,
    State,
    TokenDescription,
    a,
    commit,
    finished,
    forward_decl,
    many,
    maybe,
    parser,
    some,
    tok,
)

x, y, z = a("x"), a("y"), a("z")


def error(p: Parser[Any, Any], tokens: Any, **kwargs: Any) -> NoParseError:
    with pytest.raises(NoParseError) as e:
        p.parse(tokens, **kwargs)
    return e.value


# What was expected: every parser that failed at the furthest position.


@pytest.mark.parametrize(
    ("p", "text", "expected"),
    [
        ((x + y) | (x + z), "xw", ("'y'", "'z'")),
        (many(x) + -finished, "xxw", ("'x'", "end of input")),
        (maybe(x) + y, "w", ("'x'", "'y'")),
        (x | y | z, "w", ("'x'", "'y'", "'z'")),
    ],
    ids=["two branches, same prefix", "many before the end", "maybe before", "alternatives"],
)
def test_the_error_expects_every_parser_that_failed_at_the_furthest_token(
    p: Parser[str, Any], text: str, expected: tuple[str, ...]
) -> None:
    e = error(p, text)
    assert e.expected == expected
    assert str(e).endswith("expected: " + " or ".join(expected))


def test_a_sequence_is_expected_as_its_first_parser_not_by_its_shape() -> None:
    digit = some(str.isdigit).named("digit")
    expr = forward_decl()
    expr.define(digit | (-a("(") + expr + -a(")")))
    assert str(error(expr + -finished, "((x))")) == (
        "got unexpected token: 'x', expected: digit or '('"
    )


def test_a_label_names_a_parser_that_failed_at_its_start() -> None:
    letter = (x | y | z).named("letter")
    assert error(letter + -finished, "w").expected == ("letter",)


def test_a_label_does_not_hide_a_failure_past_its_start() -> None:
    pair = (x + y).named("pair")
    assert error(pair, "xz").expected == ("'y'",)


def test_a_parser_named_none_is_no_label() -> None:
    assert error((x + y).named(None) + z, "w").expected == ("'x'",)


def test_the_error_holds_the_token_it_stopped_at() -> None:
    e = error(x + y, "xz")
    assert (e.token, e.state.max) == ("z", 1)
    assert error(x + y, "x").token is None


# A nested parse() keeps its own error.


def test_an_error_from_a_nested_parse_is_reported_as_it_was_raised() -> None:
    inner = many(a("i")) + -finished
    outer = x + (some(lambda _: True) >> inner.parse) + -finished
    assert str(error(outer, ["x", "iiw"])) == (
        "got unexpected token: 'w', expected: 'i' or end of input"
    )


# commit()


CALL = a("f") + commit(a("(") + x + a(")"))
WORD = many(a("f"))


def test_a_committed_failure_is_not_backtracked_from() -> None:
    assert str(error(CALL | WORD, "f(y)")) == "got unexpected token: 'y', expected: 'x'"


def test_a_failure_before_the_committed_parser_reads_a_token_is_not_committed() -> None:
    assert (CALL | WORD).parse("ff") == ["f", "f"]


def test_many_and_maybe_fail_with_a_committed_failure() -> None:
    assert error(many(CALL) + -finished, "f(x)f(y)").expected == ("'x'",)
    assert error(maybe(CALL) + WORD, "f(y)").expected == ("'x'",)


def test_a_committed_failure_is_final_inside_a_negation() -> None:
    assert error(~commit(x + y), "xz").expected == ("'y'",)
    assert error(a("x").but_not(commit(x + y)), "xz").expected == ("'y'",)


GRAMMARS = st.sampled_from(["fx", "f(x)", "fy", "f(y)", "f(", "", "(x)"])


@given(st.lists(GRAMMARS, max_size=4).map("".join))
def test_commit_only_turns_a_success_into_a_failure(text: str) -> None:
    # commit(p) gives p's value as one item, as parentheses do.
    plain = many(a("f") + (a("(") + x + a(")")) | a("f") + y) + -finished
    committed = many(a("f") + commit(a("(") + x + a(")")) | a("f") + y) + -finished
    try:
        value = committed.parse(text)
    except NoParseError:
        return
    assert plain.parse(text) == value


# context()


def test_the_error_names_its_contexts_innermost_first() -> None:
    args = (a("(") + many(x) + a(")")).context("args")
    call = (a("f") + args).context("call")
    e = error(call, "f(xy)")
    assert e.context == ("args", "call")
    assert str(e) == "got unexpected token: 'y', expected: 'x' or ')' in args in call"


def test_a_context_ends_where_its_parser_ends() -> None:
    p = (x + y).context("pair") + z
    e = error(p, "xyw")
    assert e.context == ()
    assert str(e) == "got unexpected token: 'w', expected: 'z'"


# check()

DIGIT = some(str.isdigit) >> int
SMALL = DIGIT.check(lambda n: n < 5, "too big")


def test_failed_checks_are_raised_together_in_order() -> None:
    with pytest.raises(ParseErrors) as e:
        (many(SMALL) + -finished).parse("1928")
    assert [(str(err), err.state.pos) for err in e.value.exceptions] == [
        ("too big", 1),
        ("too big", 3),
    ]


def test_a_failed_parse_joins_the_failed_checks() -> None:
    with pytest.raises(ParseErrors) as e:
        (many(SMALL) + -finished).parse("19x")
    assert [str(err) for err in e.value.exceptions] == [
        "too big",
        "got unexpected token: 'x', expected: some(...) or end of input",
    ]


def test_a_check_on_a_branch_backtracked_from_is_forgotten() -> None:
    assert ((SMALL + a("!")) | (DIGIT + a("?"))).parse("9?") == (9, "?")


def test_except_star_handles_failed_checks_and_a_single_error_alike() -> None:
    caught: list[str] = []
    for text in ("9", "x"):
        try:
            (SMALL + -finished).parse(text)
        except* NoParseError as group:
            caught.extend(str(e) for e in group.exceptions)
    assert caught == ["too big", "got unexpected token: 'x', expected: some(...)"]


@given(st.text(alphabet="0123456789x", max_size=8))
def test_a_check_that_holds_changes_nothing(text: str) -> None:
    plain = many(DIGIT) + -finished
    checked = many(DIGIT.check(lambda _: True, "never")) + -finished
    try:
        value = plain.parse(text)
    except NoParseError as e:
        expected = str(e)
    else:
        assert checked.parse(text) == value
        return
    assert str(error(checked, text)) == expected


# describe= and render()

TOKENIZE = make_tokenizer(
    [TokenSpec("n", r"\d+"), TokenSpec("op", r"[+*]"), TokenSpec("sp", r"\s+")]
)
SOURCE = "1 + 2\n+ * 3"
TOKENS = [t for t in TOKENIZE(SOURCE) if t.type != "sp"]
SUM = tok("n") + many(tok("op") + tok("n")) + -finished


def test_a_token_with_a_position_is_rendered_under_its_source_line() -> None:
    e = error(SUM, TOKENS)
    assert e.description == TokenDescription("'*'", (2, 3), (2, 3))
    assert e.render(SOURCE) == (
        "2,3-2,3: got unexpected token: '*', expected: n\n2 | + * 3\n      ^"
    )


def test_the_end_of_input_is_rendered_after_the_last_token() -> None:
    assert error(SUM, TOKENS[:2]).render(SOURCE) == (
        "got unexpected end of input, expected: n\n1 | 1 + 2\n       ^"
    )


def test_render_without_a_source_or_a_position_is_the_message() -> None:
    e = error(SUM, TOKENS)
    assert e.render() == str(e)
    assert error(x, "y").render("y") == str(error(x, "y"))


def test_describe_shows_a_foreign_token() -> None:
    class Foreign:
        def __init__(self, kind: str, line: int) -> None:
            self.kind, self.line = kind, line

    def describe(t: Foreign) -> TokenDescription:
        return TokenDescription(t.kind, (t.line, 1), (t.line, 3))

    kind = some(lambda t: t.kind == "word").named("word")
    e = error(kind, [Foreign("number", 2)], describe=describe)
    assert str(e) == "2,1-2,3: got unexpected token: number, expected: word"


def test_the_default_description_reads_a_value_without_a_position() -> None:
    class Valued:
        value = "v"

    assert str(error(x, [Valued()])) == "got unexpected token: 'v', expected: 'x'"


def test_a_token_from_the_lexer_keeps_its_message() -> None:
    e = error(tok("op"), [Token("n", "1", (1, 1), (1, 1))])
    assert str(e) == "1,1-1,1: got unexpected token: '1', expected: op"


def test_a_no_parse_error_is_built_as_in_1_0() -> None:
    e = NoParseError("message", State(0, 0))
    assert (str(e), e.token, e.expected, e.context) == ("message", None, (), ())


# `finished` is shared by every grammar.


def test_naming_finished_leaves_finished_unchanged() -> None:
    eof = finished.named("EOF")
    assert error(x + eof, "xy").expected == ("EOF",)
    assert error(x + finished, "xy").expected == ("end of input",)
    assert finished.name == "end of input"


def test_a_forward_decl_defined_as_a_later_forward_decl_parses() -> None:
    alias, expr = forward_decl(), forward_decl()
    alias.define(expr)
    expr.define(x + y)
    assert alias.parse("xy") == ("x", "y")


def test_a_forward_decl_defined_as_itself_fails_at_define() -> None:
    expr = forward_decl()
    with pytest.raises(ValueError, match="defined as itself"):
        expr.define(expr)


# A token test under `>>` or `-p` fails as the parsers it is built from would, names included.


def test_a_token_test_named_after_use_is_named_in_its_errors() -> None:
    digit = some(str.isdigit)
    number = digit >> int
    digit.named("digit")
    assert error(number, "x").expected == ("digit",)


def test_a_parser_over_a_token_test_named_after_use_is_named_in_its_errors() -> None:
    word = tok("word")
    upper = word >> str.upper
    word.named("a word")
    assert error(-upper, [Token("op", "=")]).expected == ("a word",)


def test_a_skipped_token_test_names_the_token() -> None:
    e = error(tok("word") + -tok("op", ":"), [Token("word", "w"), Token("op", "=")])
    assert e.expected == ("':'",)


# The optimised parsers, fused token tests and `|` dispatched by the next token, give what the
# plain combinators give: the parsers built with `debug` on, which are neither fused nor dispatched.

WRAPPERS = ["label", "map", "skip", "maybe", "commit", "context", "not", "but_not"]
TERMS = st.recursive(
    st.tuples(st.just("a"), st.sampled_from("xyz")),
    lambda kids: st.one_of(
        st.tuples(st.sampled_from(WRAPPERS), kids),
        st.tuples(st.just("seq"), kids, kids),
        st.tuples(st.just("or"), st.lists(kids, min_size=2, max_size=5)),
    ),
    max_leaves=10,
)
GRAMMARS = st.tuples(st.just("or"), st.lists(TERMS, min_size=3, max_size=5))


WRAP: dict[str, Callable[[Parser[str, Any]], Parser[str, Any]]] = {
    "label": lambda p: p.named("label"),
    "map": lambda p: p >> repr,
    "skip": lambda p: -p + a("z"),
    "maybe": maybe,
    "commit": commit,
    "context": lambda p: p.context("ctx"),
    "not": lambda p: ~p,
    "but_not": lambda p: p.but_not(a("y")),
}


def build(g: Any) -> Parser[str, Any]:
    kind, *args = g
    if kind == "a":
        return a(args[0])
    if kind == "seq":
        return build(args[0]) + build(args[1])
    if kind == "or":
        return functools.reduce(operator.or_, [build(p) for p in args[0]])
    return WRAP[kind](build(args[0]))


def outcome(p: Parser[str, Any], text: str) -> object:
    try:
        return "value", (p + -finished).parse(text)
    except NoParseError as e:
        return "error", e.msg, e.expected, e.context, e.state.max


@given(GRAMMARS, st.text("xyz", max_size=6))
def test_the_optimised_parsers_parse_as_the_plain_ones(grammar: Any, text: str) -> None:
    plain_debug, parser_module.debug = parser_module.debug, True
    try:
        plain = build(grammar)
    finally:
        parser_module.debug = plain_debug
    assert outcome(build(grammar), text) == outcome(plain, text)


def nested() -> Parser[str, Any]:
    """Lists of `n` and `m`, and a list of a list, chosen among by the next token."""
    value: Any = forward_decl().named("value")
    items = -a("[") + maybe(value) + -a("]")
    pair = -a("(") + value + -a(",") + value + -a(")")
    value.define(items | pair | a("n") | a("m"))
    return value


@given(st.text("[](,)nmx", max_size=12))
def test_a_recursive_dispatched_grammar_parses_as_the_plain_one(text: str) -> None:
    plain_debug, parser_module.debug = parser_module.debug, True
    try:
        plain = nested()
    finally:
        parser_module.debug = plain_debug
    assert outcome(nested(), text) == outcome(plain, text)


class CountedReads(list[str]):
    """Tokens that count how many times a parser read one."""

    reads = 0

    def __getitem__(self, i: Any) -> Any:
        self.reads += 1
        return super().__getitem__(i)


@pytest.mark.parametrize("failing", ["x", "]x"])
def test_a_failure_deep_in_a_dispatched_grammar_takes_time_linear_in_the_depth(
    failing: str,
) -> None:
    depth = 20
    tokens = CountedReads("[" * depth + failing)
    with pytest.raises(NoParseError):
        nested().parse(tokens)
    assert tokens.reads < 50 * depth


def test_a_failure_before_any_token_in_nested_choices_takes_time_linear_in_the_nesting() -> None:
    depth = 20
    p: Parser[str, Any] = x | y | z
    for level in range(depth):
        p = declared(p) | a(f"p{level}") | a(f"q{level}")
    tokens = CountedReads("w")
    with pytest.raises(NoParseError):
        p.parse(tokens)
    assert tokens.reads < 50 * depth


def test_the_message_of_dispatched_alternatives_is_the_last_alternatives() -> None:
    @parser("number")
    def number(tokens: Any, s: State) -> Any:
        return ParsingFailure(s.expect(number), "got no number")

    assert error(y | z | x + number, "xw").msg == "got no number: 'w', expected: number"
    assert error(x + number | y | z, "xw").msg == "got unexpected token: 'w', expected: number"


def declared(p: Parser[str, Any]) -> Parser[str, Any]:
    """`p` behind a forward declaration: `|` cannot tell which tokens it starts with."""
    declaration: Any = forward_decl()
    declaration.define(p)
    return declaration


def test_a_labelled_choice_among_dispatched_alternatives_is_named_by_its_label() -> None:
    word = (a("p") | a("q") | a("r")).named("word")
    p = declared(x) | word | declared(z) | y
    assert error(p, "w").expected == ("'x'", "word", "'z'", "'y'")


def test_a_choice_named_after_use_among_dispatched_alternatives_is_named_by_its_label() -> None:
    word = a("p") | a("q") | a("r")
    p = declared(x) | word | declared(z) | y
    word.named("word")
    assert error(p, "w").expected == ("'x'", "word", "'z'", "'y'")


def test_a_chain_of_alternatives_builds_one_dispatch_table(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    table = parser_module._dispatch_table
    monkeypatch.setattr(
        parser_module, "_dispatch_table", lambda alts: calls.append(alts) or table(alts)
    )
    p = functools.reduce(operator.or_, [a(c) for c in "abcdefgh"])
    assert calls == []
    assert [p.parse("h"), p.parse("a")] == ["h", "a"]
    assert len(calls) == 1
