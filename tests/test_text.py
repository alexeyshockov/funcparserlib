"""`Text` grammars, `longest()` and `CompiledParser.try_parse()`: compiled code parses as the
parser.
"""

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st

from funcparserlib.lexer import LexerError, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    NoParseError,
    ParseErrors,
    Parser,
    ParsingSuccess,
    _Undecided,
    a,
    commit,
    finished,
    forward_decl,
    many,
    maybe,
    oneplus,
    pure,
    some,
    tok,
)
from funcparserlib.parser import longest as typed_longest
from funcparserlib.text import Text

type P = Parser[Any, Any]
type Runner = Callable[[P, str], Any]


def longest(*parsers: P) -> P:
    """`longest()` of parsers with values of different types, which its signature does not take."""
    return typed_longest(*parsers)


skipping = Text(skip=r"\s+")
plain = Text()
comments = Text(skip=r"(?:\s|#[^\n]*)+")


def interpreted(grammar: P, text: str) -> Any:
    return grammar.parse_text(text)


def compiled(grammar: P, text: str) -> Any:
    return grammar.compile().parse_text(text)


@pytest.fixture(params=[interpreted, compiled], ids=["interpreted", "compiled"])
def run(request: pytest.FixtureRequest) -> Runner:
    """Both ways of parsing text: the grammar itself and its generated code."""
    return request.param  # type: ignore[no-any-return]


def message(run: Runner, grammar: P, text: str) -> str:
    """The text of the error a parse raises, with its `ParseErrors` joined by newlines."""
    with pytest.raises((NoParseError, ParseErrors)) as caught:
        run(grammar, text)
    if isinstance(caught.value, ParseErrors):
        return "\n".join(str(e) for e in caught.value.exceptions)
    return str(caught.value)


def shape(value: object) -> object:
    """A value with the types of its parts: a `tuple` is not a `list`, what `+` builds is a
    `_Tuple`.
    """
    if isinstance(value, (tuple, list)):
        return (type(value), [shape(item) for item in value])
    return (type(value), value)


def outcome(parse: Callable[[str], Any], text: str) -> tuple[object, ...]:
    """What a parse gave, in a form that compares values, errors and what they carry."""
    try:
        return ("value", shape(parse(text)))
    except ParseErrors as group:
        return (
            "errors",
            [(str(e), e.expected, e.context, e.description) for e in group.exceptions],
        )
    except NoParseError as e:
        return ("error", str(e), e.expected, e.context, e.description, e.state.max)
    except (ValueError, NotImplementedError, AttributeError) as e:
        return (type(e), str(e))


# The grammars of the property: Text parsers that skip, Text parsers that do not, in one grammar.


def same(value: object) -> object:
    return value


def wrap(value: object) -> tuple[str, object]:
    return ("wrapped", value)


def short(value: object) -> bool:
    return len(repr(value)) % 3 != 0


def recursive(body: P) -> P:
    declared: P = forward_decl()
    declared.define(skipping.token("a") + maybe(body + declared))
    return declared


leaves: st.SearchStrategy[P] = st.sampled_from(
    [
        skipping.token("a"),
        skipping.token("a+"),
        skipping.token("[ab]"),
        skipping.literal("ab"),
        skipping.token("a|(?=b)"),
        skipping.literal("b"),
        plain.token("[ab]"),
        plain.token(" +"),
        plain.token("[^a]+"),
        plain.token("(?i:A)b?"),
        plain.token(r"\w+"),
        plain.literal("a"),
        plain.literal("\n"),
        skipping.end,
        plain.end,
        -skipping.token("b"),
        skipping.token("a") >> str.upper,
        pure("p"),
    ]
)


def plain_combined(kids: st.SearchStrategy[P]) -> st.SearchStrategy[P]:
    """What generated code decides on its own."""
    pairs = st.tuples(kids, kids)
    return st.one_of(
        pairs.map(lambda k: k[0] + k[1]),
        pairs.map(lambda k: k[0] | k[1]),
        pairs.map(lambda k: longest(k[0], k[1])),
        pairs.map(lambda k: longest(k[0], k[1], k[0])),
        kids.map(lambda k: k + k),
        pairs.map(lambda k: k[0].but_not(k[1])),
        kids.map(lambda k: k >> same),
        kids.map(lambda k: k >> wrap),
        kids.map(lambda k: -k),
        kids.map(many),
        kids.map(maybe),
        kids.map(oneplus),
        kids.map(lambda k: k.when(short)),
        kids.map(lambda k: k.context("label")),
        kids.map(lambda k: k.named("name")),
        kids.map(recursive),
        kids.map(lambda k: k + -skipping.end),
    )


def combined(kids: st.SearchStrategy[P]) -> st.SearchStrategy[P]:
    return st.one_of(
        plain_combined(kids), kids.map(commit), kids.map(lambda k: k.check(short, "too long"))
    )


grammars = st.recursive(leaves, combined, max_leaves=10)
plain_grammars = st.recursive(leaves, plain_combined, max_leaves=10)
texts = st.text(alphabet="ab \n", max_size=8)


def loose(value: Any) -> Any:
    """`value` as a checker sees it where a test passes what the signatures do not take."""
    return value


def raised(parse: Callable[[str], Any], text: str) -> type[BaseException] | None:
    try:
        parse(text)
    except Exception as e:  # noqa: BLE001 — which exception is the thing compared
        return type(e)
    return None


@settings(max_examples=150, deadline=None)
@given(grammars, st.lists(texts, min_size=4, max_size=4))
def test_a_compiled_text_grammar_parses_as_its_parser(grammar: P, documents: list[str]) -> None:
    code = grammar.compile()
    assume(not code._reads_tokens)
    for text in documents:
        expected = outcome(grammar.parse_text, text)
        event(str(expected[0]))
        assert outcome(code.parse_text, text) == expected


@settings(max_examples=150, deadline=None)
@given(plain_grammars, st.lists(texts, min_size=4, max_size=4))
def test_generated_code_decides_a_grammar_without_check_or_commit(
    grammar: P, documents: list[str]
) -> None:
    code = grammar.compile()
    assume(not code._reads_tokens and code._fast_text is not None)
    fast = code._fast_text
    assert fast is not None
    for text in documents:
        expected = outcome(grammar.parse_text, text)
        if not isinstance(expected[0], str):
            with pytest.raises(_Undecided):
                fast(text)
            continue
        res = fast(text)
        if expected[0] == "value":
            assert res is not None
            assert ("value", shape(res[0])) == expected
        else:
            assert res is None


@settings(max_examples=150, deadline=None)
@given(grammars, st.lists(texts, min_size=4, max_size=4))
def test_try_parse_gives_the_value_or_none_where_parse_text_raises(
    grammar: P, documents: list[str]
) -> None:
    code = grammar.compile()
    assume(not code._reads_tokens)
    for text in documents:
        expected = outcome(grammar.parse_text, text)
        if expected[0] == "value":
            assert shape(code.try_parse(text)) == expected[1]
        elif expected[0] in ("error", "errors"):
            assert code.try_parse(text) is None
        else:
            assert raised(code.try_parse, text) is expected[0]


# longest()


def test_longest_keeps_the_longest_match(run: Runner) -> None:
    grammar = longest(skipping.literal("a"), skipping.literal("ab"), skipping.literal("abc"))
    assert run(grammar + -skipping.end, "ab") == "ab"
    assert run(grammar, "abc") == "abc"


def test_longest_keeps_the_first_of_matches_that_tie(run: Runner) -> None:
    first = skipping.token("a+") >> (lambda s: ("first", s))
    second = skipping.literal("aa") >> (lambda s: ("second", s))
    assert run(longest(first, second), "aa") == ("first", "aa")
    assert run(longest(second, first), "aa") == ("second", "aa")


@pytest.mark.parametrize("flip", [False, True])
def test_longest_does_not_depend_on_the_order_of_its_parsers(run: Runner, flip: bool) -> None:
    parsers = [skipping.literal("x"), skipping.literal("x") + skipping.literal("y")]
    grammar = longest(*reversed(parsers) if flip else parsers)
    assert run(grammar, "x y") == ("x", "y")


def test_longest_ends_after_what_the_longest_skips(run: Runner) -> None:
    grammar = longest(skipping.literal("a"), skipping.literal("a") + skipping.literal("b"))
    assert run(grammar + skipping.end, "a b  ") == ("a", "b", None)


def test_a_failed_longest_names_every_alternative(run: Runner) -> None:
    grammar = longest(
        skipping.literal("a"), skipping.literal("b"), skipping.token("c+").named("cs")
    )
    assert (
        message(run, grammar, " d")
        == "1,2-1,2: got unexpected token: 'd', expected: 'a' or 'b' or cs"
    )


def test_a_failed_longest_names_what_the_alternative_that_got_furthest_expected(
    run: Runner,
) -> None:
    grammar = longest(skipping.literal("a") + skipping.literal("b"), skipping.literal("c"))
    assert message(run, grammar, "ax") == "1,2-1,2: got unexpected token: 'x', expected: 'b'"


def test_a_committed_failure_in_longest_is_final(run: Runner) -> None:
    grammar = longest(commit(skipping.literal("a") + skipping.literal("b")), skipping.literal("a"))
    assert message(run, grammar, "ac") == "1,2-1,2: got unexpected token: 'c', expected: 'b'"
    assert run(grammar, "ab") == ("a", "b")


def test_a_committed_failure_after_a_longest_is_final(run: Runner) -> None:
    grammar = longest(skipping.literal("a"), skipping.literal("a") + skipping.literal("b"))
    grammar = commit(grammar + skipping.literal("c")) | skipping.literal("a")
    assert message(run, grammar, "abx") == "1,3-1,3: got unexpected token: 'x', expected: 'c'"


def test_a_check_on_an_alternative_that_lost_is_forgotten(run: Runner) -> None:
    lost = skipping.token("a").check(lambda _: False, "lost")
    assert run(longest(lost, skipping.token("ab")), "ab") == "ab"
    assert run(longest(skipping.token("ab"), lost), "ab") == "ab"


def test_a_check_on_the_alternative_that_won_is_reported(run: Runner) -> None:
    won = skipping.token("ab").check(lambda _: False, "won")
    lost = skipping.token("a").check(lambda _: False, "lost")
    assert message(run, longest(lost, won), "ab") == "1,1-1,2: won"


def test_the_function_after_arrows_of_each_alternative_runs() -> None:
    seen: list[str] = []
    grammar = longest(
        skipping.literal("a") >> (lambda s: seen.append("short") or s),
        skipping.literal("ab") >> (lambda s: seen.append("long") or s),
    )
    assert grammar.parse_text("ab") == "ab"
    assert seen == ["short", "long"]
    seen.clear()
    assert grammar.compile().parse_text("ab") == "ab"
    assert seen == ["short", "long"]


@pytest.mark.parametrize("count", [0, 1])
def test_longest_of_fewer_than_two_parsers_raises(count: int) -> None:
    with pytest.raises(ValueError, match="at least two parsers"):
        longest(*[skipping.literal("a")][:count])


def test_longest_of_tokens_keeps_the_longest_match() -> None:
    tokenize = make_tokenizer([TokenSpec("a", "a"), TokenSpec("b", "b")])
    grammar = longest(tok("a"), tok("a") + tok("b")) + -finished
    assert grammar.parse(list(tokenize("ab"))) == ("a", "b")
    assert grammar.compile(tokenize).parse_text("ab") == ("a", "b")
    assert grammar.compile(tokenize).parse_text("a") == "a"


def test_longest_of_characters_keeps_the_longest_match() -> None:
    grammar = longest(a("x"), a("x") + a("y"), a("x") + a("y") + a("z"))
    assert grammar.parse("xyz") == ("x", "y", "z")
    assert grammar.parse("xy") == ("x", "y")
    assert grammar.compile().parse("xy") == ("x", "y")
    assert grammar.parse_text("x") == "x"


def test_longest_of_tokens_names_every_alternative() -> None:
    tokenize = make_tokenizer([TokenSpec("a", "a"), TokenSpec("b", "b"), TokenSpec("c", "c")])
    grammar = longest(tok("a"), tok("b"))
    with pytest.raises(NoParseError, match="expected: a or b"):
        grammar.compile(tokenize).parse_text("c")


# Skip


def test_leading_and_trailing_skip_is_read(run: Runner) -> None:
    grammar = skipping.token(r"\d+") + -skipping.end
    assert run(grammar, " \n 12 \t\n") == "12"


def test_text_of_only_skip_is_the_end_of_input(run: Runner) -> None:
    assert run(skipping.end, " \n\t ") is None
    assert run(skipping.end, "") is None
    assert message(run, skipping.token("a"), " \n ") == (
        "1,1-1,1: got unexpected end of input, expected: /a/"
    )


def test_skip_before_end_of_input_is_read(run: Runner) -> None:
    assert run(skipping.literal("a") + skipping.end, "a   \n") == ("a", None)
    assert message(run, skipping.literal("a") + skipping.end, "a   b") == (
        "1,5-1,5: got unexpected token: 'b', expected: end of input"
    )


def test_a_skip_that_matches_comments_skips_them_between_tokens(run: Runner) -> None:
    grammar = many(comments.token(r"\d+")) + -comments.end
    assert run(grammar, "# one\n1 # two\n 2\n#three\n3 # four") == ["1", "2", "3"]
    assert run(grammar, "# only a comment") == []


def test_skip_matches_as_many_times_as_it_can(run: Runner) -> None:
    repeated = Text(skip=r"\s|#[^\n]*")
    grammar = repeated.token("a") + repeated.token("b")
    assert run(grammar, "a b") == ("a", "b")
    assert run(grammar, "a  # c\n  b") == ("a", "b")


def test_skip_of_a_verbose_pattern_ending_in_a_comment_repeats(run: Runner) -> None:
    verbose = Text(skip=r"\s  # whitespace", flags=re.VERBOSE)
    assert run(verbose.token("a") + verbose.token("b"), "a  \n b") == ("a", "b")


def test_a_text_without_skip_reads_whitespace_as_text(run: Runner) -> None:
    assert message(run, plain.token("a") + plain.token("b"), "a b") == (
        "1,2-1,2: got unexpected token: ' ', expected: /b/"
    )
    assert (
        message(run, plain.token("a"), " a") == "1,1-1,1: got unexpected token: ' ', expected: /a/"
    )
    assert run(plain.token("a") + plain.token(" ") + plain.end, "a ") == ("a", " ", None)


def quoted(text: Text) -> P:
    """A string written as a grammar: a quote, characters, a quote."""
    chars = many(text.token(r'[^"\\]+') | text.token(r"\\."))
    return (-text.literal('"') + chars + -text.literal('"')) >> "".join


def test_a_string_grammar_of_a_text_without_skip_keeps_its_whitespace(run: Runner) -> None:
    assert run(quoted(plain), '" a  b\\n\\" "') == ' a  b\\n\\" '


def test_a_string_grammar_of_a_skipping_text_loses_the_whitespace_after_its_quote(
    run: Runner,
) -> None:
    assert run(quoted(skipping), '"  a  b "') == "a  b "


def test_the_end_skips_what_a_text_without_skip_left(run: Runner) -> None:
    string = -plain.literal('"') + plain.token('[^"]+') + -plain.literal('"') + -skipping.end
    assert run(string, '"a b"') == "a b"
    assert run(string, '"a b"  ') == "a b"


def test_an_error_inside_a_string_grammar_names_what_it_expected(run: Runner) -> None:
    grammar = quoted(plain).named("string") + -plain.end
    assert message(run, grammar, '"ab') == (
        "1,4-1,4: got unexpected end of input, expected: /[^\"\\\\]+/ or /\\\\./ or '\"'"
    )


@pytest.mark.parametrize("pattern", [r"\s*", r"", r"(?:)", r"x?"])
def test_a_skip_that_matches_the_empty_string_raises(pattern: str) -> None:
    with pytest.raises(ValueError, match="matches the empty string"):
        Text(skip=pattern)


@pytest.mark.parametrize("pattern", [r"a*", r"", r"x?"])
def test_a_token_that_matches_the_empty_string_raises(pattern: str) -> None:
    with pytest.raises(ValueError, match="matches the empty string"):
        skipping.token(pattern)


def test_an_empty_literal_raises() -> None:
    with pytest.raises(ValueError, match="literal value is empty"):
        skipping.literal("")


def test_a_literal_is_text() -> None:
    with pytest.raises(TypeError, match="literal value must be a str, not int") as caught:
        skipping.literal(loose(5))
    assert caught.value.__notes__


def test_flags_apply_to_the_skip_and_to_a_token() -> None:
    loud = Text(skip=r"x+", flags=re.IGNORECASE)
    grammar = loud.token("a", re.IGNORECASE) + loud.token("b")
    assert grammar.parse_text("XxAXXb") == ("A", "b")
    assert grammar.compile().parse_text("XxAXXb") == ("A", "b")


def test_a_token_pattern_that_matches_the_empty_string_at_a_position_fails_there(
    run: Runner,
) -> None:
    grammar = skipping.token(r"a|(?=b)")
    assert run(grammar, "a") == "a"
    assert message(run, grammar, "b") == "1,1-1,1: got unexpected token: 'b', expected: /a|(?=b)/"


def test_what_the_alternatives_of_longest_expected_is_in_the_error_after_it(run: Runner) -> None:
    grammar = longest(skipping.literal("a"), skipping.literal("a") + skipping.literal("b"))
    assert message(run, grammar + skipping.literal("c"), "ax") == (
        "1,2-1,2: got unexpected token: 'x', expected: 'b' or 'c'"
    )


@pytest.mark.parametrize(
    ("grammar", "text", "value"),
    [
        (skipping.end, "", None),
        (skipping.end, " \n ", None),
        (many(skipping.token("a")) + -skipping.end, "  ", []),
        (many(skipping.token("a")) + -skipping.end, "", []),
        (maybe(skipping.token("a")) + -skipping.end, "\n", None),
        (skipping.token("a") + -skipping.end, " a \n", "a"),
        (skipping.token("a") + skipping.end, "a", ("a", None)),
        (skipping.token("a") | skipping.end, "  ", None),
    ],
)
def test_the_generated_code_accepts_what_the_parser_accepts(
    grammar: P, text: str, value: object
) -> None:
    code = grammar.compile()
    assert code._fast_text is not None
    result = code._fast_text(text)
    assert result is not None
    assert result[0] == value == grammar.parse_text(text)


# Errors


def pair() -> P:
    return (
        skipping.token(r"\d+").named("number")
        + skipping.literal("+")
        + skipping.token(r"\d+").named("number")
    )


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("1 + x", "1,5-1,5"),
        ("1 +\n  x", "2,3-2,3"),
        ("1 +\r\n x", "2,2-2,2"),
        ("1 +\t x", "1,6-1,6"),
        ("1 +\n\n\n\tx", "4,2-4,2"),
        ("\n\n1 + x", "3,5-3,5"),
    ],
)
def test_an_error_has_the_line_and_the_column_of_the_text(
    run: Runner, text: str, where: str
) -> None:
    assert (
        message(run, pair() + -skipping.end, text)
        == f"{where}: got unexpected token: 'x', expected: number"
    )


def test_a_tab_is_one_column(run: Runner) -> None:
    assert (
        message(run, pair(), "\t1 +\t\tx") == "1,7-1,7: got unexpected token: 'x', expected: number"
    )


def test_the_end_of_the_input_has_a_position(run: Runner) -> None:
    assert message(run, pair(), "1 +") == "1,4-1,4: got unexpected end of input, expected: number"
    assert message(run, pair(), "1 +  \n ") == (
        "1,4-1,4: got unexpected end of input, expected: number"
    )
    assert message(run, pair(), "") == "1,1-1,1: got unexpected end of input, expected: number"


def test_an_error_shows_the_word_it_stopped_at(run: Runner) -> None:
    assert message(run, pair(), "1 + abc+def ghi") == (
        "1,5-1,5: got unexpected token: 'abc+def', expected: number"
    )
    assert message(run, pair(), "1 + " + "x" * 30) == (
        f"1,5-1,5: got unexpected token: '{'x' * 20}', expected: number"
    )


def test_an_error_renders_the_line_it_is_on(run: Runner) -> None:
    with pytest.raises(NoParseError) as caught:
        run(pair(), "1 +\n  x + 3")
    assert caught.value.render("1 +\n  x + 3") == "\n".join(
        ["2,3-2,3: got unexpected token: 'x', expected: number", "2 |   x + 3", "      ^"]
    )


def test_an_error_renders_a_line_that_ends_with_a_carriage_return(run: Runner) -> None:
    source = "1 +\r\n x"
    with pytest.raises(NoParseError) as caught:
        run(pair(), source)
    assert caught.value.render(source).splitlines()[1:] == ["2 |  x", "     ^"]


def test_parse_errors_render_each_error(run: Runner) -> None:
    digit = plain.token(r"\d").check(lambda v: int(v) < 5, "too big") + -plain.literal(",")
    source = "9,1,\n7,"
    grammar = many(digit | -plain.literal("\n")) + -plain.end
    with pytest.raises(ParseErrors) as caught:
        run(grammar, source)
    assert caught.value.render(source) == "\n".join(
        ["1,1-1,1: too big", "1 | 9,1,", "    ^", "", "2,1-2,1: too big", "2 | 7,", "    ^"]
    )


def test_a_failed_check_names_the_position_of_its_token(run: Runner) -> None:
    grammar = many(skipping.token(r"\d+").check(lambda v: int(v) < 5, "too big")) + -skipping.end
    assert message(run, grammar, "1 2 \n 9 3  7") == "2,2-2,2: too big\n2,7-2,7: too big"


def test_a_failed_check_spans_the_token_without_the_text_skipped_after_it(run: Runner) -> None:
    grammar = skipping.token(r"[a-z]+").check(lambda _: False, "bad") + -skipping.end
    with pytest.raises(ParseErrors) as caught:
        run(grammar, "ab   ")
    (error,) = caught.value.exceptions
    assert error.description is not None
    assert (error.description.start, error.description.end) == ((1, 1), (1, 2))


def test_a_failed_check_names_the_token_it_checked(run: Runner) -> None:
    grammar = skipping.token(r"[a-z]+").check(lambda _: False, "bad") + skipping.literal("+")
    with pytest.raises(ParseErrors) as caught:
        run(grammar, "ab+cd")
    (error,) = caught.value.exceptions
    assert error.description is not None
    assert error.description.text == "'ab'"


def test_a_check_error_and_the_error_of_the_parse_come_together(run: Runner) -> None:
    grammar = many(skipping.token(r"\d+").check(lambda v: int(v) < 5, "too big")) + -skipping.end
    assert (
        message(run, grammar, "9 x")
        == "1,1-1,1: too big\n1,3-1,3: got unexpected token: 'x', expected: /\\d+/ or end of input"
    )


def test_a_label_replaces_what_a_parser_expected(run: Runner) -> None:
    named = skipping.token(r"\d+").named("number")
    assert message(run, named, "x") == "1,1-1,1: got unexpected token: 'x', expected: number"
    assert (
        message(run, skipping.token(r"\d+"), "x")
        == "1,1-1,1: got unexpected token: 'x', expected: /\\d+/"
    )
    assert (
        message(run, skipping.literal("if"), "x")
        == "1,1-1,1: got unexpected token: 'x', expected: 'if'"
    )
    assert (
        message(run, skipping.end, "x")
        == "1,1-1,1: got unexpected token: 'x', expected: end of input"
    )


def test_a_context_names_where_the_parser_was(run: Runner) -> None:
    named = skipping.token(r"\d+").named("number")
    grammar = (skipping.literal("(") + named + skipping.literal(")")).context("group")
    assert (
        message(run, grammar, "( x")
        == "1,3-1,3: got unexpected token: 'x', expected: number in group"
    )


def test_an_error_names_everything_expected_where_the_parsing_stopped(run: Runner) -> None:
    grammar = skipping.literal("a") | skipping.literal("b").named("bee") | skipping.end
    assert message(run, grammar, " \n c") == (
        "2,2-2,2: got unexpected token: 'c', expected: 'a' or bee or end of input"
    )


def test_an_error_from_a_nested_parse_is_unchanged(run: Runner) -> None:
    inner = skipping.token("a") + skipping.token("b")

    def run_inner(text: Any, s: Any) -> Any:
        return inner.run(text, s)

    grammar = Parser(run_inner) + skipping.end
    assert message(run, grammar, "a\nc") == message(run, inner, "a\nc")


def test_a_failed_check_on_a_branch_that_was_backtracked_from_is_forgotten(run: Runner) -> None:
    bad = skipping.token("a").check(lambda _: False, "bad") + skipping.token("b")
    assert run(bad | skipping.token("a") + skipping.token("c"), "a c") == ("a", "c")


# Token grammars compiled


@pytest.fixture
def tokenize() -> Any:
    return make_tokenizer(
        [TokenSpec("space", r"\s+"), TokenSpec("num", r"\d+"), TokenSpec("plus", r"\+")],
        skip=["space"],
    )


def sum_of_two() -> P:
    return (tok("num") >> int) + -tok("plus") + (tok("num") >> int) + -finished


def test_text_no_token_spec_matches_is_explained_by_what_the_grammar_expected(
    tokenize: Any,
) -> None:
    code = sum_of_two().compile(tokenize)
    with pytest.raises(NoParseError) as caught:
        code.parse_text("1 + $")
    assert str(caught.value) == "1,5-1,5: got unexpected token: '$', expected: num"
    assert isinstance(caught.value.__cause__, LexerError)
    assert caught.value.expected == ("num",)


def test_the_explanation_of_text_on_a_later_line(tokenize: Any) -> None:
    with pytest.raises(NoParseError, match=r"^3,3-3,3: got unexpected token: '\$', expected: num$"):
        sum_of_two().compile(tokenize).parse_text("1 +\n\n  $")


def test_a_mistake_among_the_tokens_before_the_bad_text_is_the_error(tokenize: Any) -> None:
    code = sum_of_two().compile(tokenize)
    with pytest.raises(NoParseError) as caught:
        code.parse_text("1 1 $")
    assert str(caught.value) == "1,3-1,3: got unexpected token: '1', expected: plus"
    assert isinstance(caught.value.__cause__, LexerError)


def test_the_tokenizer_error_is_raised_when_the_grammar_accepts_the_tokens_before_it(
    tokenize: Any,
) -> None:
    with pytest.raises(LexerError, match="cannot tokenize data"):
        many(tok("num")).compile(tokenize).parse_text("1 2 $")


def test_bad_text_at_the_start_is_explained(tokenize: Any) -> None:
    with pytest.raises(NoParseError, match=r"^1,1-1,1: got unexpected token: '\$', expected: num$"):
        sum_of_two().compile(tokenize).parse_text("$")


def test_a_grammar_of_tokens_parses_text_as_the_interpreted_parser_parses_the_tokens(
    tokenize: Any,
) -> None:
    code = sum_of_two().compile(tokenize)
    for text in ["1+2", " 1 + 2 ", "1 + ", "1 2", "", "+"]:
        assert outcome(code.parse_text, text) == outcome(
            lambda t: sum_of_two().parse(list(tokenize(t))), text
        )


def test_the_text_of_a_token_grammar_is_explained_when_the_grammar_is_not_generated(
    tokenize: Any,
) -> None:
    opaque = Parser(lambda toks, s: sum_of_two().run(toks, s))
    code = opaque.compile(tokenize)
    assert code._fast_text is None
    with pytest.raises(NoParseError, match="got unexpected token: '\\$', expected: num"):
        code.parse_text("1 + $")


def test_try_parse_of_a_token_grammar_gives_the_value_or_none(tokenize: Any) -> None:
    code = sum_of_two().compile(tokenize)
    assert code.try_parse("1 + 2") == (1, 2)
    assert code.try_parse("1 +") is None
    assert code.try_parse("1 + 2 3") is None
    assert code.try_parse("1 + $") is None
    assert code.try_parse("") is None


def test_try_parse_of_a_grammar_that_accepts_the_tokens_before_bad_text_is_none(
    tokenize: Any,
) -> None:
    assert many(tok("num")).compile(tokenize).try_parse("1 2 $") is None
    assert many(tok("num")).compile(tokenize).try_parse("1 2") == ["1", "2"]


def test_try_parse_keeps_a_valid_parse_whose_other_branch_had_a_failing_check(
    tokenize: Any,
) -> None:
    rejected = tok("num").check(lambda _: False, "rejected") + tok("num")
    grammar = (rejected | tok("num") + -tok("plus")) + tok("num") + -finished
    code = grammar.compile(tokenize)
    assert code.try_parse("1 + 2") == ("1", "2")
    assert code.parse_text("1 + 2") == ("1", "2")


def test_try_parse_rejects_a_parse_whose_check_failed(tokenize: Any) -> None:
    grammar = tok("num").check(lambda _: False, "rejected") + -finished
    code = grammar.compile(tokenize)
    assert code.try_parse("1") is None
    with pytest.raises(ParseErrors):
        code.parse_text("1")


def test_try_parse_of_a_commit_around_a_parsing_function(tokenize: Any) -> None:
    def read_plus(toks: Sequence[Any], s: Any) -> Any:
        return ParsingSuccess(toks[s.pos].value, s.reset_pos(s.pos + 1))

    grammar = (
        tok("num") + commit(Parser(read_plus) + tok("num")) | tok("num") + tok("plus")
    ) + -finished
    code = grammar.compile(tokenize)
    assert code.try_parse("1 + 2") == code.parse_text("1 + 2") == ("1", ("+", "2"))
    assert code.try_parse("1 + +") is None
    assert code.try_parse("1 +") is None


def test_try_parse_gives_none_for_nesting_too_deep_for_the_parser() -> None:
    nested: P = forward_decl()
    nested.define(skipping.literal("(") + maybe(nested) + skipping.literal(")"))
    code = nested.compile()
    deep = "(" * 5000 + ")" * 5000
    assert code.try_parse(deep) is None
    assert code.try_parse("(())") is not None
    with pytest.raises(RecursionError):
        code.parse_text(deep)


def test_try_parse_without_a_tokenizer_raises_for_a_grammar_of_tokens() -> None:
    with pytest.raises(ValueError, match="without a tokenizer cannot parse text"):
        tok("num").compile().try_parse("1")


def test_parse_text_without_a_tokenizer_raises_for_a_grammar_of_tokens() -> None:
    with pytest.raises(ValueError, match="without a tokenizer cannot parse text"):
        tok("num").compile().parse_text("1")


def test_a_text_grammar_needs_no_tokenizer() -> None:
    code = (skipping.token(r"\d+") >> int).compile()
    assert code.parse_text(" 12 ") == 12
    assert code.try_parse("x") is None


# Characters and Text in one grammar


def test_a_grammar_of_characters_parses_text(run: Runner) -> None:
    assert run(a("x") + a("y"), "xy") == ("x", "y")


def test_a_grammar_of_characters_reports_where_it_stopped() -> None:
    grammar = a("x") + a("\n") + a("y")
    with pytest.raises(NoParseError) as caught:
        grammar.parse_text("x\nz")
    assert str(caught.value) == "2,1-2,1: got unexpected token: 'z', expected: 'y'"


def test_characters_and_a_text_in_one_grammar_agree_when_compiled(run: Runner) -> None:
    grammar = some(lambda c: c == "a") + skipping.literal("b") + skipping.end
    assert run(grammar, "ab ") == ("a", "b", None)
    assert message(run, grammar, "ax") == "1,2-1,2: got unexpected token: 'x', expected: 'b'"


def test_a_character_is_not_skipped_over_by_a_text(run: Runner) -> None:
    grammar = a("x") + a(" ") + skipping.literal("b")
    assert run(grammar, "x b") == ("x", " ", "b")
    assert run(skipping.literal("b") + a("c"), "b c") == ("b", "c")


def test_a_text_and_tok_in_one_grammar_cannot_read_the_text(run: Runner) -> None:
    read_text: P = skipping.token("a")
    with pytest.raises(AttributeError, match="has no attribute 'type'"):
        run(read_text + tok("b"), "ab")


# A broken fast path falls back to the interpreted parser with the same result, so these look at it.


@pytest.mark.parametrize(
    ("grammar", "text", "value"),
    [
        (plain.token("a+"), "aa", "aa"),
        (plain.literal("ab"), "ab", "ab"),
        (skipping.token("a") + skipping.literal("b") + -skipping.end, " a b ", ("a", "b")),
        (longest(plain.literal("a"), plain.token("a+")), "aa", "aa"),
        (commit(plain.literal("a")) | plain.literal("b"), "b", "b"),
        (a("x") + plain.literal("y"), "xy", ("x", "y")),
    ],
)
def test_the_generated_code_decides_what_it_can(grammar: P, text: str, value: object) -> None:
    fast = grammar.compile()._fast_text
    assert fast is not None
    res = fast(text)
    assert res is not None
    assert res[0] == value


def test_the_generated_code_rejects_without_the_parser() -> None:
    fast = (plain.literal("a") + plain.token("b")).compile()._fast_text
    assert fast is not None
    assert fast("ac") is None


def test_text_may_start_with_skip_before_a_first_parser_without_it(run: Runner) -> None:
    grammar = plain.literal("a") + skipping.literal("b")
    assert run(grammar, "  ab") == ("a", "b")


def test_leading_skip_is_no_token_a_commit_read(run: Runner) -> None:
    grammar = commit(skipping.literal("a")) | skipping.literal("b")
    assert run(grammar, " b") == "b"


def test_a_label_applies_past_leading_skip(run: Runner) -> None:
    named = (skipping.literal("(") | skipping.literal("[")).named("expr")
    assert message(run, named, " ?") == "1,2-1,2: got unexpected token: '?', expected: expr"


def test_a_failed_when_is_at_the_token_past_leading_skip(run: Runner) -> None:
    name = skipping.token(r"\w+").when(lambda s: s != "if").named("name")
    assert message(run, name, "\n\n  if") == "3,3-3,3: got unexpected token: 'if', expected: name"


def test_a_commit_of_the_end_is_no_token_read(run: Runner) -> None:
    grammar = plain.literal("a") + (commit(skipping.end) | plain.literal(" b"))
    assert run(grammar, "a b") == ("a", " b")


def test_a_failed_check_spans_no_comment_after_the_token(run: Runner) -> None:
    grammar = comments.token(r"\d+").check(lambda _: False, "bad") + -comments.end
    assert message(run, grammar, "1 # note\n") == "1,1-1,1: bad"


def test_a_failed_check_at_the_end_of_the_text_has_a_position(run: Runner) -> None:
    grammar = many(skipping.literal("a")).check(bool, "at least one") + -skipping.end
    assert message(run, grammar, "") == "1,1-1,1: at least one"


def test_a_failed_parse_gives_the_character_as_its_token() -> None:
    with pytest.raises(NoParseError) as caught:
        skipping.literal("a").parse_text("?x")
    assert caught.value.token == "?x"  # noqa: S105 — a parsed word


def test_try_parse_runs_a_conversion_in_two_passes_at_most() -> None:
    seen: list[str] = []
    grammar = (plain.token(r"\d") >> seen.append).check(lambda _: False, "bad")
    assert grammar.compile().try_parse("0") is None
    assert len(seen) == 2


def test_a_skip_with_a_global_flag_raises_with_a_hint() -> None:
    with pytest.raises(ValueError, match="cannot repeat"):
        Text(skip=r"(?i)x+")


def test_a_commit_past_a_skipping_end_is_final(run: Runner) -> None:
    grammar = many(commit(skipping.end + plain.literal("aa")) | plain.literal("a"))
    assert message(run, grammar, "a\n") == "1,2-1,2: got unexpected end of input, expected: 'aa'"


# `Parser.recover()`: every error of a document in one run.

blank = Text(skip=r"[ \t]+")
line_end = blank.token(r"\n+").named("newline")
setting = (
    blank.token("[a-z]+").named("key")
    + -blank.literal("=")
    + (blank.token(r"\d+") >> int).named("number")
).recover(until=line_end)
config = many(setting + -maybe(line_end)) + -blank.end


def test_recover_reports_every_failure(run: Runner) -> None:
    assert message(run, config, "a = 1\nb = ?\nc 2\nd = 4\n") == (
        "2,5-2,5: got unexpected token: '?', expected: number\n"
        "3,3-3,3: got unexpected token: '2', expected: '='"
    )


def test_recover_changes_nothing_in_a_valid_document(run: Runner) -> None:
    assert run(config, "a = 1\nb = 2\n") == [("a", 1), ("b", 2)]


def test_recover_reports_a_failure_at_the_end_of_the_text(run: Runner) -> None:
    assert message(run, config, "a =") == ("1,4-1,4: got unexpected end of input, expected: number")


def test_recover_gives_its_default_for_what_it_skipped() -> None:
    grammar = many(
        (skipping.literal("a") + skipping.literal("b")).recover(skipping.literal(";"), ("?", "?"))
        + -skipping.literal(";")
    )
    with pytest.raises(ParseErrors) as caught:
        grammar.parse_text("a b; a c;")
    assert len(caught.value.exceptions) == 1


def test_recover_fails_as_usual_at_its_first_token(run: Runner) -> None:
    assert run(many(setting), "") == []
    assert message(run, setting + -blank.end, "=") == (
        "1,1-1,1: got unexpected token: '=', expected: key"
    )


def test_recover_keeps_a_check_error_before_the_failure(run: Runner) -> None:
    number = blank.token(r"\d+").check(lambda _: False, "bad")
    line = (number + blank.literal("x")).recover(until=line_end)
    assert message(run, many(line + -maybe(line_end)) + -blank.end, "1 y\n") == (
        "1,1-1,1: bad\n1,3-1,3: got unexpected token: 'y', expected: 'x'"
    )


def test_recover_catches_a_commit_failure(run: Runner) -> None:
    line = (blank.literal("a") + commit(blank.literal("b"))).recover(until=line_end)
    assert message(run, many(line + -maybe(line_end)) + -blank.end, "a c\n") == (
        "1,3-1,3: got unexpected token: 'c', expected: 'b'"
    )


def test_recover_with_tokens() -> None:
    spec = make_tokenizer(
        [TokenSpec("sp", " +"), TokenSpec("a", "a"), TokenSpec("b", "b"), TokenSpec("end", ";")],
        skip={"sp"},
    )
    grammar = many((tok("a") + tok("b")).recover(until=tok("end")) + -tok("end")) + -finished
    assert grammar.parse(list(spec("a b;"))) == [("a", "b")]
    with pytest.raises(ParseErrors) as caught:
        grammar.compile(spec).parse_text("a a; a b; b;")
    assert len(caught.value.exceptions) == 2


def test_recover_in_try_parse_rejects() -> None:
    assert config.compile().try_parse("a = ?\n") is None
    assert config.compile().try_parse("a = 1\n") == [("a", 1)]


# First-character tests


@pytest.mark.parametrize(
    ("grammar", "text", "value"),
    [
        pytest.param(plain.token("a+") | plain.token("[ab]+"), "ab", "a", id="overlap, first"),
        pytest.param(plain.token("b+") | plain.token("[ab]+"), "ab", "ab", id="overlap, second"),
        pytest.param(longest(plain.token("a+"), plain.token("[ab]+")), "ab", "ab", id="longest"),
        pytest.param(plain.token("(?i:a)") | pure("x"), "A", "A", id="scoped flag"),
        pytest.param(plain.token("a", re.IGNORECASE) | pure("x"), "A", "A", id="flag"),
        pytest.param(plain.token("[^a]+") | pure("x"), "", "x", id="negated class, end of text"),
        pytest.param(plain.token("[^a]+") | pure("x"), "a", "x", id="negated class, excluded"),
        pytest.param(plain.token("[^a]+") | pure("x"), "b", "b", id="negated class, included"),
        pytest.param(maybe(plain.token("a")) + plain.token("b"), "b", (None, "b"), id="nullable"),
    ],
)
def test_a_first_character_test_keeps_what_the_alternatives_match(
    grammar: P, text: str, value: Any
) -> None:
    assert grammar.parse_text(text) == value
    assert grammar.compile().parse_text(text) == value


@pytest.mark.parametrize("call", ["parse_text", "try_parse"])
def test_bytes_given_to_a_grammar_of_text_raise_as_in_its_parser(call: str) -> None:
    code = (plain.token("a") | pure("x")).compile()
    with pytest.raises(TypeError, match="string pattern"):
        getattr(code, call)(loose(b"a"))


def test_alternatives_read_the_first_character_once(tmp_path: Path) -> None:
    source = tmp_path / "generated.py"
    (plain.token("a") | plain.token("b") | plain.token("c")).compile(source_file=source)
    assert source.read_text().count("T[p0:p0 + 1]") == 1
