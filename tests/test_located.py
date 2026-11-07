"""Where a value was read from, and the parts of a text a grammar reads."""

import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from funcparserlib.lexer import TokenSpec, make_tokenizer
from funcparserlib.parser import (
    Invalid,
    Located,
    NoParseError,
    ParseErrors,
    Parser,
    TokenDescription,
    commit,
    finished,
    forward_decl,
    lit,
    many,
    maybe,
    prefixed,
    regex,
    skipping,
    some,
    tok,
    tokenized,
)
from funcparserlib.util import Lines

P = Parser[Any, Any]
Run = Callable[[P, str], Any]

spaced = skipping(r"\s+|#[^\n]*")
word = spaced.regex(r"[a-z]+").named("word")
tokenize = make_tokenizer(
    [TokenSpec("space", r"\s+"), TokenSpec("name", r"[a-z]+"), TokenSpec("op", r"[=;]")],
    skip={"space"},
)


@pytest.fixture(params=["interpreted", "compiled"])
def run(request: pytest.FixtureRequest) -> Run:
    """Parse a text with a grammar, as it is and compiled."""
    if request.param == "interpreted":
        return lambda grammar, text: grammar.parse_text(text)
    return lambda grammar, text: grammar.compile().parse_text(text)


def test_a_value_knows_the_offsets_it_was_read_from(run: Run) -> None:
    assert run(regex("a+").located() + regex("b+").located(), "aabbb") == (
        Located("aa", 0, 2),
        Located("bbb", 2, 5),
    )


def test_a_span_ends_before_the_skip_after_it(run: Run) -> None:
    text = "foo  # one\n  bar  # two\n"
    first, second = run(word.located() + word.located() + -spaced.end, text)
    assert text[first.start : first.end] == "foo"
    assert text[second.start : second.end] == "bar"


def test_a_rule_spans_all_it_read_without_the_skip_after_it(run: Run) -> None:
    text = "foo bar  # end\n"
    pair = run((word + word).located() + -spaced.end, text)
    assert pair == Located(("foo", "bar"), 0, 7)


quoted = spaced.regex(r'"[^"]*"')


def test_a_span_keeps_what_the_skip_pattern_would_match_inside_the_value(run: Run) -> None:
    assert run(quoted.located() + -spaced.end, '"a #b"  # c\n') == Located('"a #b"', 0, 6)


def test_a_failed_part_leaves_the_end_of_the_span_where_it_was(run: Run) -> None:
    rule = (word + maybe(lit("!")) + many(lit("?"))).located()
    assert run(rule + -spaced.end, "foo  # c\n") == Located(("foo", None, []), 0, 3)


def test_the_end_of_the_text_leaves_the_end_of_the_span_where_it_was(run: Run) -> None:
    assert run((word + -spaced.end).located(), "foo  # c\n") == Located("foo", 0, 3)
    assert run((regex("a") + -spaced.end).located(), "a  ") == Located("a", 0, 1)


def test_a_label_or_a_commit_around_a_part_keeps_the_end_of_the_span(run: Run) -> None:
    assert run(word.context("a word").located() + -spaced.end, "foo  ") == Located("foo", 0, 3)
    assert run(commit(word).located() + -spaced.end, "foo  ") == Located("foo", 0, 3)
    assert run((word + commit(maybe(lit("!")))).located() + -spaced.end, "foo  ") == Located(
        ("foo", None), 0, 3
    )


def test_a_checked_or_recovered_part_keeps_the_end_of_the_span(run: Run) -> None:
    assert run(word.validated(str).located() + -spaced.end, "foo  ") == Located("foo", 0, 3)
    assert run(word.recover(default="").located() + -spaced.end, "foo  ") == Located("foo", 0, 3)
    rule = (word + maybe(lit("!")).recover()).located()
    assert run(rule + -spaced.end, "foo  ") == Located(("foo", None), 0, 3)


def test_a_recovered_rejection_keeps_the_end_of_the_span() -> None:
    def refuse(value: str) -> str:
        raise Invalid("no words")

    seen: list[Located[str]] = []
    rule = (word.validated(refuse).recover(default="x").located() >> seen.append) + -spaced.end
    with pytest.raises(ParseErrors):
        rule.parse_text("foo  ")
    assert seen[0] == Located("x", 0, 3)


def test_a_prefixed_region_keeps_the_end_of_the_span(run: Run) -> None:
    rule = prefixed(regex(r"\d") >> int, word).located()
    assert run(rule, "5foo  ") == Located("foo", 0, 4)


def test_a_span_that_read_nothing_after_a_skip_stays_empty(run: Run) -> None:
    grammar = word + maybe(lit("!")).located() + -spaced.end
    assert run(grammar, "foo  ") == ("foo", Located(None, 5, 5))


def test_a_rejected_value_is_shown_whole() -> None:
    def refuse(value: str) -> str:
        raise Invalid("no strings")

    with pytest.raises(ParseErrors) as caught:
        quoted.validated(refuse).parse_text('"a #b"  # c')
    description = caught.value.exceptions[0].description
    assert description is not None
    assert description.text == repr('"a #b"')


def test_located_values_nest(run: Run) -> None:
    pair = (word.located() + word.located()).located()
    assert run(pair, "a  b") == Located((Located("a", 0, 1), Located("b", 3, 4)), 0, 4)


def test_a_located_grammar_still_skips_what_the_text_starts_with(run: Run) -> None:
    assert run(word.located() + -spaced.end, "  # c\n  foo ") == Located("foo", 8, 11)


def test_located_reads_what_its_parser_reads(run: Run) -> None:
    grammar = (lit("a") + lit("b")).located() | lit("ac").located()
    assert run(grammar, "ac") == Located("ac", 0, 2)


def test_located_works_in_a_forward_declared_rule(run: Run) -> None:
    item: Any = forward_decl()
    item.define((-spaced.lit("(") + item + -spaced.lit(")") | word).located())
    assert run(item, "( ( a ) ) ") == Located(Located(Located("a", 4, 5), 2, 7), 0, 9)


def test_a_failure_of_a_located_parser_says_what_its_parser_expected(run: Run) -> None:
    with pytest.raises(NoParseError, match="expected: word"):
        run(word.located(), "1")


def test_a_rejected_value_has_no_location() -> None:
    def rejected(_: str) -> str:
        raise Invalid("not this one")

    called = []
    grammar = regex("a").validated(rejected).located() >> called.append
    with pytest.raises(ParseErrors, match="1 parsing error"):
        grammar.parse_text("a")
    assert called == []


def test_a_value_of_tokens_knows_their_indexes() -> None:
    tokens = list(tokenize("a = b;\nc = d;"))
    assign = (tok("name") + -tok("op", "=") + tok("name")).located() + -tok("op", ";")
    grammar = many(assign) + -finished
    expected = [Located(("a", "b"), 0, 3), Located(("c", "d"), 4, 7)]
    assert grammar.parse(tokens) == expected
    assert grammar.compile().parse(tokens) == expected
    in_text = [Located(("a", "b"), 0, 5), Located(("c", "d"), 7, 12)]
    assert tokenized(grammar, tokenize).parse_text("a = b;\nc = d;") == in_text
    assert tokenized(grammar, tokenize).compile().parse_text("a = b;\nc = d;") == in_text
    assert tokens[expected[1].start].start == (2, 1)


def test_an_alternative_of_a_located_token_reads_the_token_when_compiled() -> None:
    tokens = list(tokenize("a"))
    grammar = maybe(tok("name").located())
    assert grammar.parse(tokens) == Located("a", 0, 1)
    assert grammar.compile().parse(tokens) == Located("a", 0, 1)


def test_a_position_inside_a_prefixed_region_is_relative_to_it() -> None:
    length = regex(r"\d") >> int
    assert prefixed(length, regex("a+").located()).parse_text("2aa") == Located("aa", 0, 2)


def test_a_position_in_bytes_is_an_offset() -> None:
    byte = some(lambda v: v == 97)
    assert many(byte.located()).parse_bytes(b"aa") == [Located(97, 0, 1), Located(97, 1, 2)]


# `re` says what the matches of a pattern are, as a grammar of the same pattern has to.

date = regex(r"\d{4}") + -lit("-") + regex(r"\d\d")
pattern = re.compile(r"(\d{4})-(\d\d)")
texts = st.lists(st.sampled_from(["2026-11", "1999-0", "-", "12", "é", "\n", " ", "20"])).map(
    "".join
)


@given(texts)
def test_matches_are_those_of_re(text: str) -> None:
    expected = [Located(m.groups(), m.start(), m.end()) for m in pattern.finditer(text)]
    assert list(date.scan_text(text)) == expected
    assert list(date.compile().scan_text(text)) == expected


@given(texts)
def test_replace_gives_what_re_gives(text: str) -> None:
    expected = pattern.sub(lambda m: f"<{m[1]}/{m[2]}>", text)

    def render(m: Located[tuple[str, str]]) -> str:
        return f"<{m.value[0]}/{m.value[1]}>"

    assert date.replace_text(text, render) == expected
    assert date.compile().replace_text(text, render) == expected


def test_matches_stop_where_the_program_stops() -> None:
    seen: list[str] = []
    grammar = regex("a") >> (lambda value: seen.append(value) or value)
    assert next(grammar.scan_text("a a a")) == Located("a", 0, 1)
    assert seen == ["a"]


def test_matches_do_not_overlap() -> None:
    assert [m.start for m in regex("aa").scan_text("aaaaa")] == [0, 2]


def test_the_skip_after_a_match_stays_in_the_text() -> None:
    assert word.replace_text("foo  # c\nbar", lambda m: m.value.upper()) == "FOO  # c\nBAR"


def test_a_replaced_match_leaves_none_of_its_text() -> None:
    assert quoted.replace_text('x "a #b" y "c" z', lambda m: "S") == "x S y S z"
    assert quoted.compile().replace_text('x "a #b" y', lambda m: "S") == "x S y"
    assert [m.end for m in quoted.scan_text('x "a #b"  # c\n')] == [8]


def test_text_without_a_match_gives_no_matches() -> None:
    assert list(word.scan_text("123")) == []
    assert word.replace_text("123", lambda m: "x") == "123"
    assert list(word.scan_text("")) == []


def test_a_failed_commit_raises_with_its_place() -> None:
    link = commit(lit("[[") + regex("[a-z]+") + -lit("]]"))
    assert [m.value for m in link.scan_text("x [[a]] y")] == [("[[", "a")]
    with pytest.raises(NoParseError) as caught:
        list(link.scan_text("ok\n[[a"))
    assert str(caught.value) == "2,4-2,4: got unexpected end of input, expected: ']]'"


def test_a_match_before_a_failed_commit_comes_first() -> None:
    link = commit(lit("[[") + regex("[a-z]+") + -lit("]]"))
    found = link.scan_text("[[a]] [[b")
    assert next(found) == Located(("[[", "a"), 0, 5)
    with pytest.raises(NoParseError):
        next(found)


def test_replace_gives_nothing_when_a_part_is_wrong() -> None:
    link = commit(lit("[[") + regex("[a-z]+") + -lit("]]"))
    with pytest.raises(NoParseError):
        link.replace_text("[[a]] [[b", lambda m: m.value[1])


def test_a_rejected_value_raises() -> None:
    def month(value: str) -> int:
        if int(value) > 12:
            raise Invalid("a month")
        return int(value)

    grammar = regex(r"\d\d").validated(month)
    assert [m.value for m in grammar.scan_text("11 12")] == [11, 12]
    with pytest.raises(ParseErrors) as caught:
        list(grammar.scan_text("11 13"))
    assert caught.value.exceptions[0].msg == "1,4-1,5: a month"


def test_a_match_that_reads_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="matched no text at offset 0") as caught:
        list(maybe(lit("a")).scan_text("xa"))
    assert "maybe()" in caught.value.__notes__[0]


def test_matches_refuse_what_is_not_text() -> None:
    with pytest.raises(TypeError, match="bytes cannot be parsed as text"):
        word.scan_text(b"a")  # ty: ignore[invalid-argument-type]


def test_matches_refuse_a_grammar_of_tokens_at_the_call() -> None:
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse text") as caught:
        tok("name").scan_text("a")
    assert "tokenized()" in caught.value.__notes__[0]


def test_a_grammar_of_tokens_bound_to_its_tokenizer_matches_in_a_text() -> None:
    grammar = tokenized(tok("name") + -tok("op", "=") + tok("name"), tokenize)
    text = "say a = b, then c=d."
    expected = [Located(("a", "b"), 4, 9), Located(("c", "d"), 16, 19)]
    assert list(grammar.scan_text(text)) == expected
    assert list(grammar.compile().scan_text(text)) == expected
    assert grammar.replace_text(text, lambda m: "=".join(reversed(m.value))) == "say b=a, then d=c."


# `Token` positions and the text say what a place is.


def expected_place(text: str, offset: int) -> tuple[int, int]:
    return text.count("\n", 0, offset) + 1, offset - text.rfind("\n", 0, offset)


lined = st.text("ab\n", max_size=12)


@given(lined)
def test_a_place_counts_lines_and_columns_from_one(text: str) -> None:
    lines = Lines(text)
    for offset in range(len(text) + 1):
        assert lines.place(offset) == expected_place(text, offset)


@given(lined)
def test_an_offset_and_its_place_are_the_same_position(text: str) -> None:
    lines = Lines(text)
    for offset in range(len(text) + 1):
        assert lines.offset(lines.place(offset)) == offset


def test_a_token_is_where_lines_say_it_is() -> None:
    text = "a = b;\n\n  cd = e;"
    lines = Lines(text)
    for token in tokenize(text):
        assert token.start is not None
        assert token.end is not None
        start, end = lines.offset(token.start), lines.offset(token.end) + 1
        assert text[start:end] == token.value


@pytest.mark.parametrize("offset", [-1, 4])
def test_an_offset_outside_the_text_is_an_error(offset: int) -> None:
    with pytest.raises(ValueError, match=f"offset {offset} is outside the text"):
        Lines("a\nb").place(offset)


@pytest.mark.parametrize("place", [(0, 1), (3, 1), (1, 0), (1, 3), (2, 3)])
def test_a_place_outside_the_text_is_an_error(place: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="is outside the text"):
        Lines("a\nb").offset(place)


def test_a_round_of_many_that_fails_leaves_the_end_of_the_span_where_it_was(run: Run) -> None:
    pairs = many(word + spaced.lit("=")).located()
    assert run(pairs + word + -spaced.end, "a = b =  c # end\n") == (
        Located([("a", "="), ("b", "=")], 0, 7),
        "c",
    )


def test_a_span_of_a_rule_ends_where_the_rule_read_last(run: Run) -> None:
    expr: Any = forward_decl()
    expr.define(word + maybe(spaced.lit("(") + expr + spaced.lit(")")))
    nested = ("a", ("(", ("b", ("(", ("c", None), ")")), ")"))
    assert run(expr.located() + -spaced.end, "a ( b ( c ) )  # end\n") == Located(nested, 0, 13)


def test_spans_of_text_are_generated_code() -> None:
    grammar = many(word.located()) + -spaced.end
    compiled = grammar.compile()
    assert compiled._fast_text is not None
    res = compiled._fast_text("a  bb # c\n")
    assert res is not None
    assert res[0] == [Located("a", 0, 1), Located("bb", 3, 5)]


# A scan of tokens.

pair = tok("name") + -tok("op", "=") + tok("name")


@pytest.mark.parametrize("compiled", [False, True])
def test_a_scan_of_tokens_gives_their_indexes(compiled: bool) -> None:
    tokens = list(tokenize("x a = b ; c = d e"))
    grammar = pair.compile() if compiled else pair
    assert list(grammar.scan(tokens)) == [Located(("a", "b"), 1, 4), Located(("c", "d"), 5, 8)]


class Counted(list[Any]):
    """A list that records the indexes read from it."""

    read: list[int]

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, int):
            self.read.append(i)
        return super().__getitem__(i)

    def __iter__(self) -> Iterator[Any]:
        self.read.append(len(self))
        return super().__iter__()


def test_a_scan_of_tokens_reads_them_only_as_far_as_its_matches() -> None:
    tokens = Counted(tokenize("a = b ; " * 1000))
    tokens.read = []
    assert next(pair.scan(tokens)) == Located(("a", "b"), 0, 3)
    assert max(tokens.read) < 5


def test_a_span_function_gives_the_place_of_a_match_in_its_text() -> None:
    text = "x a = b ; c = d e"
    lines = Lines(text)

    def span(tokens: Any, start: int, end: int) -> tuple[int, int]:
        last = tokens[end - 1]
        return lines.offset(tokens[start].start), lines.offset(last.start) + len(last.value)

    found = [(m.value, text[m.start : m.end]) for m in pair.scan(list(tokenize(text)), span=span)]
    assert found == [(("a", "b"), "a = b"), (("c", "d"), "c = d")]


def test_a_failed_commit_in_a_scan_of_tokens_raises_with_the_description() -> None:
    tokens = list(tokenize("a = b ; c = ;"))
    with pytest.raises(NoParseError) as caught:
        list(commit(pair).scan(tokens, describe=lambda t: TokenDescription(f"<{t.value}>")))
    assert "<;>" in str(caught.value)


def test_a_scan_of_tokens_refuses_a_match_that_reads_none() -> None:
    with pytest.raises(ValueError, match="matched no tokens at index 0"):
        list(maybe(tok("name")).scan(list(tokenize("= a"))))


@pytest.mark.parametrize("text", ["foo bar", b"foo bar"])
def test_a_scan_of_tokens_refuses_text(text: Any) -> None:
    with pytest.raises(TypeError, match="no sequence of tokens to scan") as caught:
        list(word.scan(text))
    assert "scan_text()" in caught.value.__notes__[0]


def test_a_span_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(ValueError, match="span gave an end of 0, before its start of 4"):
        list(pair.scan(list(tokenize("a = b")), span=lambda tokens, start, end: (4, 0)))
