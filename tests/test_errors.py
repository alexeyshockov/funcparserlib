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
    ForwardDeclParser,
    Invalid,
    NoParseError,
    ParseErrors,
    Parser,
    ParsingFailure,
    ParsingResult,
    ParsingSuccess,
    State,
    TokenDescription,
    a,
    anything_but,
    commit,
    counted,
    finished,
    forward_decl,
    many,
    maybe,
    parser,
    prefixed,
    pure,
    regex,
    some,
    switch,
    tok,
)
from tests.rules import kept

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
    assert error(anything_but(commit(x + y)), "xz").expected == ("'y'",)
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


def test_the_error_names_the_contexts_every_expected_parser_is_in() -> None:
    p = (x.context("first") | y.context("second")).context("inner").context("outer")
    e = error(p, "z")
    assert e.context == ("inner", "outer")
    assert str(e) == "got unexpected token: 'z', expected: 'x' or 'y' in inner in outer"


def test_a_context_ends_where_its_parser_ends() -> None:
    p = (x + y).context("pair") + z
    e = error(p, "xyw")
    assert e.context == ()
    assert str(e) == "got unexpected token: 'w', expected: 'z'"


# validated() as a rule on a value

DIGIT = some(str.isdigit) >> int
SMALL = DIGIT.validated(kept(lambda n: n < 5, "too big"))


def test_broken_rules_are_raised_together_in_order() -> None:
    with pytest.raises(ParseErrors) as e:
        (many(SMALL) + -finished).parse("1928")
    assert [(str(err), err.state.pos) for err in e.value.exceptions] == [
        ("too big", 1),
        ("too big", 3),
    ]


def test_a_failed_parse_joins_the_broken_rules() -> None:
    with pytest.raises(ParseErrors) as e:
        (many(SMALL) + -finished).parse("19x")
    assert [str(err) for err in e.value.exceptions] == [
        "too big",
        "got unexpected token: 'x', expected: some(...) or end of input",
    ]


@pytest.mark.interpreted_only
def test_a_rule_on_a_branch_backtracked_from_is_forgotten() -> None:
    assert ((SMALL + a("!")) | (DIGIT + a("?"))).parse("9?") == (9, "?")


def test_except_star_handles_broken_rules_and_a_single_error_alike() -> None:
    caught: list[str] = []
    for source in ("9", "x"):
        try:
            (SMALL + -finished).parse(source)
        except* NoParseError as group:
            caught.extend(str(e) for e in group.exceptions)
    assert caught == ["too big", "got unexpected token: 'x', expected: some(...)"]


@given(st.text(alphabet="0123456789x", max_size=8))
def test_a_rule_that_holds_changes_nothing(text: str) -> None:
    plain = many(DIGIT) + -finished
    checked = many(DIGIT.validated(kept(lambda _: True, "never"))) + -finished
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


def test_a_token_over_several_lines_is_marked_to_the_end_of_its_first() -> None:
    tokenize = make_tokenizer(
        [TokenSpec("n", r"\d+"), TokenSpec("s", r'"[^"]*"'), TokenSpec("sp", r"\s+")]
    )
    source = '1 "ab\ncd"'
    e = error(SUM, [t for t in tokenize(source) if t.type != "sp"])
    assert e.render(source).split("\n")[1:] == ['1 | 1 "ab', "      ^^^"]


def test_a_line_is_rendered_as_it_is_without_its_carriage_return() -> None:
    source = "1 +  \r\n+ * 3"
    tokens = [t for t in TOKENIZE(source.replace("\r", " ")) if t.type != "sp"]
    assert error(SUM, tokens[:2]).render(source).split("\n")[1:] == ["1 | 1 +  ", "       ^"]


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

    kind: Parser[Foreign, Foreign] = some(lambda t: t.kind == "word").named("word")
    e = error(kind, [Foreign("number", 2)], describe=describe)
    assert str(e) == "2,1-2,3: got unexpected token: number, expected: word"


def test_the_default_description_reads_a_value_without_a_position() -> None:
    class Valued:
        value = "v"

    assert str(error(x, [Valued()])) == "got unexpected token: 'v', expected: 'x'"


def test_a_token_with_a_start_and_no_end_has_no_position() -> None:
    class Started:
        value = "v"
        start = (1, 1)
        end = None

    assert error(x, [Started()]).description == TokenDescription("'v'")


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
    "not": anything_but,
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
    value: ForwardDeclParser[str, Any] = forward_decl().named("value")
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
    declaration: ForwardDeclParser[str, Any] = forward_decl()
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


def test_a_token_that_cannot_be_hashed_fails_dispatched_alternatives_as_any_token() -> None:
    p = (x | y | z) + -finished
    assert str(error(p, [["x"]])) == "got unexpected token: ['x'], expected: 'x' or 'y' or 'z'"


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


def test_a_recovered_error_shows_a_token_as_describe_says() -> None:
    line = (a("x") + a("=") + a("1")).recover(until=a(";"))
    grammar = many(line + -a(";")) + -finished
    with pytest.raises(ParseErrors) as e:
        grammar.parse(list("x=?;"), describe=lambda t: TokenDescription(f"<{t}>"))
    assert [error.msg for error in e.value.exceptions] == [
        "got unexpected token: <?>, expected: '1'"
    ]


# validated()


def odd(digit: str) -> int:
    """The number of an odd digit; an even one is rejected."""
    if int(digit) % 2 == 0:
        raise Invalid(f"{digit} is even")
    return int(digit)


def noted(digit: str) -> str:
    """The digit, with an error for `2` that keeps it."""
    if digit == "2":
        raise Invalid("2 is noted", keep=True)
    return digit


digit = some(str.isdigit)
Compiled = Callable[[Parser[Any, Any]], Any]
both = pytest.mark.parametrize("build", [lambda p: p, lambda p: p.compile()], ids=["run", "gen"])


def messages(p: Any, tokens: Any) -> list[str]:
    """The messages of the errors of a parse that raises `ParseErrors`."""
    with pytest.raises(ParseErrors) as e:
        p.parse(tokens)
    return [error.msg for error in e.value.exceptions]


@both
def test_validated_gives_what_the_function_returns(build: Compiled) -> None:
    assert build(many(digit.validated(odd)) + -finished).parse(list("13")) == [1, 3]


@both
def test_validated_reports_every_rejected_value(build: Compiled) -> None:
    p = build(many(digit.validated(odd)) + -finished)
    assert messages(p, list("1234")) == ["2 is even", "4 is even"]


def test_no_function_after_a_rejected_value_runs() -> None:
    seen: list[Any] = []
    p = digit.validated(odd) + digit >> seen.append
    assert messages(p, list("23")) == ["2 is even"]
    assert seen == []


def test_a_rejection_in_a_list_reaches_no_function_over_the_list() -> None:
    seen: list[Any] = []
    p = many(digit.validated(odd)) >> seen.append
    assert messages(p, list("123")) == ["2 is even"]
    assert seen == []


@pytest.mark.interpreted_only
def test_a_kept_value_goes_on_to_what_comes_after() -> None:
    seen: list[Any] = []
    p = digit.validated(noted).validated(odd) >> seen.append
    assert messages(p, ["2"]) == ["2 is noted", "2 is even"]
    assert seen == []
    q = digit.validated(noted) >> seen.append
    assert messages(q, ["2"]) == ["2 is noted"]
    assert seen == ["2"]


def test_a_function_does_not_run_over_a_rejected_value() -> None:
    seen: list[Any] = []
    p = digit.validated(odd).validated(seen.append)
    assert messages(p, ["2"]) == ["2 is even"]
    assert seen == []


def test_a_rejected_value_is_no_reason_to_try_an_alternative() -> None:
    assert messages(digit.validated(odd) | a("2"), ["2"]) == ["2 is even"]


def test_a_part_that_stopped_at_a_rejected_value_is_no_reason_to_try_an_alternative() -> None:
    p = counted(digit.validated(odd), a("x")) | (a("2") + a("x") + a("x"))
    assert messages(p + -finished, list("2xx")) == ["2 is even"]


def test_when_still_decides_among_alternatives() -> None:
    p = digit.when(lambda d: d == "1") | a("2")
    assert p.parse(["2"]) == "2"


def test_a_rejection_on_a_branch_that_fails_later_is_forgotten() -> None:
    p = (digit.validated(odd) + a("!")) | (a("2") + a("?"))
    assert p.parse(["2", "?"]) == ("2", "?")


def test_a_nested_grammar_rejects_with_its_own_error() -> None:
    inner = many(a("i")) + -finished
    anything: Parser[str, str] = some(lambda _: True)
    p = anything.validated(inner.parse) + -finished
    assert messages(p, ["iw"]) == ["got unexpected token: 'w', expected: 'i' or end of input"]


def test_a_nested_grammar_rejects_with_every_error_it_has() -> None:
    inner = many(digit.validated(odd)) + -finished
    anything: Parser[str, str] = some(lambda _: True)
    p = anything.validated(inner.parse) + -finished
    assert messages(p, ["24"]) == ["2 is even", "4 is even"]


def test_a_label_names_what_a_nested_grammar_expected() -> None:
    inner = many(a("i")) + -finished
    anything: Parser[str, str] = some(lambda _: True)
    p = anything.validated(inner.parse, "a word of i") + -finished
    with pytest.raises(ParseErrors) as e:
        p.parse(["iw"], describe=lambda t: TokenDescription(f"<{t}>"))
    [error] = e.value.exceptions
    assert error.msg == "got unexpected token: <iw>, expected: a word of i"
    assert isinstance(error.__cause__, NoParseError)


def test_a_nested_error_covers_every_token_of_the_value() -> None:
    inner = many(a("i")) + -finished
    pair: Parser[str, str] = some(str.isalpha) + some(str.isalpha) >> "".join
    (labelled,) = (
        pair.validated(inner.parse, "a word of i").run(list("iw"), State(0, 0)).state.problems
    )
    (unlabelled,) = pair.validated(inner.parse).run(list("iw"), State(0, 0)).state.problems
    assert (labelled.start, labelled.end, labelled.rejected) == (0, 2, True)
    assert (unlabelled.start, unlabelled.end, unlabelled.rejected) == (0, 2, True)
    assert not unlabelled.unexpected


def test_no_function_runs_over_a_value_a_nested_grammar_rejected() -> None:
    seen: list[Any] = []
    inner = many(a("i")) + -finished
    anything: Parser[str, str] = some(lambda _: True)
    p = anything.validated(inner.parse) >> seen.append
    assert messages(p, ["iw"]) == ["got unexpected token: 'w', expected: 'i' or end of input"]
    assert seen == []


def test_the_error_of_a_nested_grammar_is_the_cause() -> None:
    inner = many(a("i")) + -finished
    anything: Parser[str, str] = some(lambda _: True)
    with pytest.raises(ParseErrors) as e:
        anything.validated(inner.parse).parse(["iw"])
    assert isinstance(e.value.exceptions[0].__cause__, NoParseError)


def test_an_error_inside_prefixed_covers_its_tokens_in_the_whole_input() -> None:
    length = some(str.isdigit) >> int
    p = a("x") + prefixed(length, many(digit.validated(odd)))
    problems = p.run(list("x3124"), State(0, 0)).state.problems
    assert [(e.message, e.start, e.end) for e in problems] == [
        ("2 is even", 3, 4),
        ("4 is even", 4, 5),
    ]


def test_the_exception_of_the_function_is_the_cause() -> None:
    with pytest.raises(ParseErrors) as e:
        digit.validated(odd).parse(["2"])
    assert isinstance(e.value.exceptions[0].__cause__, Invalid)


def test_any_other_exception_is_a_bug_and_propagates() -> None:
    known: dict[str, int] = {}
    with pytest.raises(KeyError):
        digit.validated(known.__getitem__).parse(["2"])


@pytest.mark.parametrize(
    "needs_value",
    [
        lambda n: n.bind(a),
        lambda n: counted(n, a("x")),
        prefixed,
        lambda n: switch(n, {1: a("x")}),
        lambda n: n.when(lambda v: v > 0),
        lambda n: n.but_not(a("x")),
    ],
    ids=["bind", "counted", "prefixed", "switch", "when", "but_not"],
)
def test_what_needs_a_value_adds_nothing_to_a_rejection(needs_value: Any) -> None:
    assert messages(needs_value(digit.validated(odd)), ["2", "x"]) == ["2 is even"]


@pytest.mark.parametrize(
    ("p", "tokens"),
    [
        (digit.validated(lambda d: (d, d)) + digit, list("12")),
        ((digit + digit).validated(lambda pair: pair[0]) + digit, list("123")),
        ((-a("x")).validated(lambda _: "made") + digit, list("x1")),
        ((digit + digit).validated(list) + digit, list("123")),
        (digit.validated(lambda _: None) + digit, list("12")),
        (many(a("1").validated(int)) + digit, list("12")),
    ],
    ids=["from-token", "from-tuple", "from-ignored", "to-list", "to-none", "in-many"],
)
def test_generated_code_gives_the_value_validated_makes(p: Parser[str, Any], tokens: Any) -> None:
    assert p.compile().parse(tokens) == p.parse(tokens)


def test_a_failure_after_a_rejected_value_stays_a_failure_in_bind() -> None:
    p = (digit.validated(odd) + a("x")).bind(pure)
    assert messages(p, list("2y")) == ["2 is even", "got unexpected token: 'y', expected: 'x'"]
    assert (p | (digit + a("y"))).parse(list("2y")) == ("2", "y")


# recover()


def test_recover_goes_on_from_until_after_a_part_that_stopped() -> None:
    row = counted(digit.validated(odd) >> int, a("x")) + -a(";")
    p = many(row.recover(until=a("1"))) + -finished
    assert messages(p, list("2xx;1x;")) == ["2 is even"]


def test_recover_without_until_leaves_a_part_that_stopped() -> None:
    row = counted(digit.validated(odd), a("x")).recover(default=list[str]())
    assert isinstance(row.run(list("2x"), State(0, 0)), ParsingFailure)


@both
def test_recover_gives_a_rejected_part_its_default(build: Compiled) -> None:
    item = (digit.validated(odd) + (digit >> int) >> sum).recover(default=-1)
    p = many(item) >> list
    res = p.run(list("2314"), State(0, 0))
    assert res.value == [-1, 5]
    assert [e.message for e in res.state.problems] == ["2 is even"]
    assert messages(build(p + -finished), list("2314")) == ["2 is even"]


@pytest.mark.interpreted_only
def test_a_function_runs_above_a_recovered_part() -> None:
    seen: list[Any] = []
    p = digit.validated(odd).recover(default=0) + digit >> seen.append
    assert messages(p, list("23")) == ["2 is even"]
    assert seen == [(0, "3")]


def test_recover_inside_recover_closes_the_inner_part_only() -> None:
    inner = digit.validated(odd).recover(default="inner")
    outer = ((inner + a("x")) >> (lambda p: ("built", p))).recover(default="outer")
    res = outer.run(["2", "x"], State(0, 0))
    assert res.value == ("built", ("inner", "x"))


def test_the_error_recover_goes_on_after_covers_the_token_it_stopped_at() -> None:
    line = (x + a("=") + a("1")).recover(until=a(";"))
    (stopped,) = line.context("line").run(list("x=?;"), State(0, 0)).state.problems
    assert (stopped.start, stopped.end, stopped.context) == (2, 3, ("line",))
    (at_end,) = line.run(list("x="), State(0, 0)).state.problems
    assert (at_end.message, at_end.start, at_end.end) == (
        "got unexpected end of input, expected: '1'",
        2,
        3,
    )
    (in_text,) = line.run("x=?;", State(0, 0)).state.problems
    assert (in_text.start, in_text.end) == (2, 3)


def test_the_error_recover_goes_on_after_has_the_token_and_its_place() -> None:
    tokenize = make_tokenizer([TokenSpec("space", r"\s+"), TokenSpec("op", r"\S")], skip=["space"])
    line = a(Token("op", "x")) + a(Token("op", "=")) + a(Token("op", "1"))
    semicolon = a(Token("op", ";"))
    recovered = line.recover(until=semicolon).context("line")
    with pytest.raises(ParseErrors) as e:
        (recovered + -semicolon + -finished).parse(list(tokenize("x = ? ;")))
    (stopped,) = e.value.exceptions
    assert str(stopped) == "1,5-1,5: got unexpected token: '?', expected: '1' in line"
    assert stopped.token == Token("op", "?")
    assert stopped.description == TokenDescription("'?'", (1, 5), (1, 5))
    assert stopped.context == ("line",)
    assert stopped.state == State(2, 2)
    with pytest.raises(ParseErrors) as e:
        (recovered + -finished).parse(list(tokenize("x =")))
    (at_end,) = e.value.exceptions
    assert str(at_end) == "1,4-1,4: got unexpected end of input, expected: '1' in line"
    assert at_end.token is None


@both
def test_the_error_recover_goes_on_after_reads_as_the_error_of_the_parse(build: Compiled) -> None:
    line = ((x + a("=") + a("1")).context("assignment") | (y + a("?")).context("query")).context(
        "line"
    )
    with pytest.raises(NoParseError) as fatal:
        build(line + -a(";") + -finished).parse(list("x=?;"))
    with pytest.raises(ParseErrors) as recovered:
        build(line.recover(until=a(";")) + -a(";") + -finished).parse(list("x=?;"))
    assert [str(e) for e in recovered.value.exceptions] == [str(fatal.value)]
    assert str(fatal.value) == "got unexpected token: '?', expected: '1' in assignment in line"


def test_recover_without_until_leaves_a_failure_alone() -> None:
    p = (a("x") + a("y")).recover(default=0)
    assert isinstance(p.run(list("xz"), State(0, 0)), ParsingFailure)


def test_recover_changes_nothing_in_a_part_that_is_not_broken() -> None:
    assert digit.validated(noted).recover(default=0).run(["1"], State(0, 0)).value == "1"


def test_a_kept_value_is_no_broken_part() -> None:
    res = digit.validated(noted).recover(default=0).run(["2"], State(0, 0))
    assert res.value == "2"


# State.rejected_since()


def test_a_parsing_function_can_tell_a_rejected_part() -> None:
    @parser("doubled")
    def doubled(tokens: Any, s: State) -> ParsingResult[int]:
        res = digit.validated(odd).run(tokens, s)
        if isinstance(res, ParsingFailure) or res.state.rejected_since(s):
            return res
        return ParsingSuccess(res.value * 2, res.state)

    assert doubled.parse(["3"]) == 6
    assert messages(doubled, ["2"]) == ["2 is even"]


# Compiled


def test_try_parse_gives_none_for_a_rejected_value() -> None:
    p = regex(r"\d").validated(odd).compile()
    assert p.try_parse("3") == 3
    assert p.try_parse("2") is None
