"""A token grammar bound to its tokenizer, as a parser of text."""

import gc
import weakref
from collections.abc import Callable
from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from funcparserlib.lexer import LexerError, Token, Tokenizer, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    Invalid,
    Located,
    NoParseError,
    ParseErrors,
    Parser,
    State,
    _Chain,
    _TextTokens,
    anything_but,
    finished,
    lit,
    many,
    maybe,
    skipping,
    some,
    tok,
    tokenized,
)

P = Parser[Any, Any]
Run = Callable[[P, str], Any]

tokenize = make_tokenizer(
    [
        TokenSpec("space", r"\s+"),
        TokenSpec("comment", r"#[^\n]*"),
        TokenSpec("int", r"\d+"),
        TokenSpec("name", r"[a-z]+"),
        TokenSpec("op", r"[=+;]"),
    ],
    skip={"space", "comment"},
)
number = tok("int") >> int
total = number + many(-tok("op", "+") + number)
assign = tok("name") + -tok("op", "=") + number
spaced = skipping(r"\s+")


@pytest.fixture(params=["interpreted", "compiled"])
def run(request: pytest.FixtureRequest) -> Run:
    """Parse a text with a grammar, as it is and compiled."""
    if request.param == "interpreted":
        return lambda grammar, text: grammar.parse_text(text)
    return lambda grammar, text: grammar.compile().parse_text(text)


def lexed(text: str) -> list[Token]:
    """The tokens of `text`, ending with what the tokenizer stopped at, as a parser reads them."""
    tokens: list[Token] = []
    try:
        tokens.extend(tokenize(text))
    except LexerError:
        pass
    return tokens


texts = st.text(alphabet="ab1 =+;#\n?", max_size=30)


@given(texts)
def test_from_the_start_of_a_text_the_tokens_are_those_of_the_tokenizer(text: str) -> None:
    read = list(_TextTokens(_Chain(text, tokenize), 0))
    expected = lexed(text)
    assert read[: len(expected)] == expected
    assert [(t.start, t.end) for t in read[: len(expected)]] == [(t.start, t.end) for t in expected]
    try:
        list(tokenize(text))
    except LexerError:
        assert len(read) == len(expected) + 1
        assert read[-1].type == "text no token spec matches"
    else:
        assert len(read) == len(expected)


@given(texts, st.integers(0, 30))
def test_tokens_from_an_offset_are_those_of_the_rest_of_the_text(text: str, offset: int) -> None:
    offset = min(offset, len(text))
    chain = _Chain(text, tokenize)
    list(_TextTokens(chain, 0))
    from_offset = [t.value for t in _TextTokens(chain, offset)]
    rest = [t.value for t in lexed(text[offset:])]
    assert from_offset[: len(rest)] == rest


def test_a_grammar_of_tokens_finds_its_matches_and_steps_over_text_it_cannot_read() -> None:
    text = "say x = 1 and y=22? then ??? z = 3"
    assert [(m.value, m.start, m.end) for m in tokenized(assign, tokenize).scan(text)] == [
        (("x", 1), 4, 9),
        (("y", 22), 14, 18),
        (("z", 3), 29, 34),
    ]


@given(texts)
def test_matches_are_those_of_the_grammar_at_each_offset(text: str) -> None:
    grammar = tokenized(assign, tokenize)
    for m in grammar.scan(text):
        assert grammar.parse_text(text[m.start : m.end]) == m.value
        assert not text[m.start].isspace()
        assert not text[m.end - 1].isspace()


def test_a_match_does_not_start_at_what_the_tokenizer_skips() -> None:
    matches = tokenized(total, tokenize).scan("  1 + 2 # c\n 3")
    assert [(m.value, m.start, m.end) for m in matches] == [((1, [2]), 2, 7), ((3, []), 13, 14)]


def test_a_bound_grammar_is_part_of_a_grammar_of_text(run: Run) -> None:
    calc = spaced.lit("calc") + tokenized(total, tokenize) + spaced.lit(";")
    assert run(calc + -spaced.end, "calc 1 + 2 # sum\n ;") == ("calc", (1, [2]), ";")


def test_text_goes_on_after_the_skip_of_its_last_token(run: Run) -> None:
    grammar = tokenized(number, tokenize).located() + spaced.regex("[A-Z]+")
    assert run(grammar, "12  # c\n  AB") == (Located(12, 0, 2), "AB")


def test_a_bound_part_does_not_read_what_comes_before_skips() -> None:
    with pytest.raises(NoParseError):
        (lit("calc") + tokenized(number, tokenize)).parse_text("calc 1")
    assert (spaced.lit("calc") + tokenized(number, tokenize)).parse_text("calc 1") == ("calc", 1)


def test_spans_inside_a_bound_grammar_are_offsets_of_the_text(run: Run) -> None:
    grammar = tokenized(many(assign.located() + -tok("op", ";")) + finished, tokenize)
    assert run(grammar, "a = 1;\n  bb=22 ;") == (
        [Located(("a", 1), 0, 5), Located(("bb", 22), 9, 14)],
        None,
    )


def test_a_span_of_a_bound_part_ends_before_its_skip(run: Run) -> None:
    grammar = (spaced.lit("x") + tokenized(total, tokenize)).located() + spaced.lit(";")
    assert run(grammar, "x 1 + 2  # c\n;") == (Located(("x", (1, [2])), 0, 7), ";")


def test_an_error_in_a_bound_part_has_its_place_in_the_text() -> None:
    calc = spaced.lit("calc") + tokenized(total, tokenize) + spaced.lit(";")
    with pytest.raises(NoParseError) as caught:
        calc.parse_text("calc 1 +\n  ;")
    assert str(caught.value) == "2,3-2,3: got unexpected token: ';', expected: int"


def test_a_rejected_value_in_a_bound_part_has_its_place_in_the_text() -> None:
    def small(n: int) -> int:
        if n > 9:
            raise Invalid("too big")
        return n

    calc = spaced.lit("calc") + tokenized(number.validated(small), tokenize)
    with pytest.raises(ParseErrors) as caught:
        calc.parse_text("calc  42")
    assert str(caught.value.exceptions[0]) == "1,7-1,8: too big"


def test_parse_text_of_a_bound_grammar_raises_the_errors_of_its_tokens() -> None:
    grammar = tokenized(total + finished, tokenize)
    with pytest.raises(NoParseError) as caught:
        grammar.parse_text("1 + ?")
    assert str(caught.value) == "1,5-1,5: got unexpected token: '?', expected: int"
    assert isinstance(caught.value.__cause__, LexerError)
    with pytest.raises(LexerError):
        tokenized(total, tokenize).parse_text("1 ?")


def test_a_bound_grammar_takes_only_text() -> None:
    with pytest.raises(TypeError, match="list cannot be parsed as text"):
        tokenized(total, tokenize).run(cast("Any", ["1"]), State(0, 0))
    with pytest.raises(TypeError, match="bytes cannot be parsed as text"):
        tokenized(total, tokenize).parse_bytes(b"1")


def test_tokenizing_reads_each_offset_once_over_all_the_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[int] = []
    read = Tokenizer._read

    def counted(self: Tokenizer, s: str, i: int) -> tuple[str, int] | None:
        reads.append(i)
        return read(self, s, i)

    monkeypatch.setattr(Tokenizer, "_read", counted)
    text = "word x = 1 # prose\n" * 200
    assert len(list(tokenized(assign, tokenize).scan(text))) == 200
    assert len(reads) == len(set(reads))
    assert len(reads) <= len(text)


def test_maybe_of_a_bound_part_that_reads_nothing_keeps_the_skip_before_it(run: Run) -> None:
    grammar = (spaced.lit("x") + maybe(tokenized(number, tokenize))).located() + -spaced.end
    assert run(grammar, "x  ") == Located(("x", None), 0, 1)


def test_a_bound_part_that_reads_to_the_end_ends_past_the_skip_there(run: Run) -> None:
    assert run(tokenized(total, tokenize) + finished, "1 + 2  # c") == (1, [2], None)


def test_a_token_of_a_value_says_it_cannot_parse_text_as_its_compiled_parser_does() -> None:
    plus = tok("op", "+")
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse text"):
        plus.parse_text("+")
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse text"):
        plus.compile().parse_text("+")
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse bytes"):
        plus.parse_bytes(b"+")


class Text(str):
    """A text a test can hold a weak reference to."""

    __slots__ = ("__weakref__",)


def test_a_parse_keeps_no_text_once_it_returns() -> None:
    grammar = spaced.lit("calc") + tokenized(total, tokenize)
    text = Text("calc 1 + 2")
    assert grammar.parse_text(text) == ("calc", (1, [2]))
    assert [m.value for m in grammar.scan(text)] == [("calc", (1, [2]))]
    assert grammar.compile().parse_text(text) == ("calc", (1, [2]))
    ref = weakref.ref(text)
    del text
    gc.collect()
    assert ref() is None


def test_the_first_match_reads_the_text_only_as_far_as_it(monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[int] = []
    read = Tokenizer._read

    def counted(self: Tokenizer, s: str, i: int) -> tuple[str, int] | None:
        reads.append(i)
        return read(self, s, i)

    monkeypatch.setattr(Tokenizer, "_read", counted)
    name = some(lambda t: t.type == "name") >> (lambda t: t.value)
    statement = name + -tok("op", "=") + (number | tok("name"))
    statements = statement + many(-tok("op", ";") + statement) + maybe(-tok("op", ";"))
    text = "say a = 1; b = c;\n" + "and then c = 3; " * 10_000
    first = next(tokenized(statements, tokenize).scan(text))
    assert first == Located(("a", 1, [("b", "c")]), 4, 17)
    assert max(reads) < 40


def no_length(self: _TextTokens) -> int:
    """`len()` of the tokens of a text, which a parser asks for only to report an error."""
    msg = "a parser read the length of the tokens of a text"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("grammar", "text", "value"),
    [
        (some(lambda t: t.type == "int"), "x 1", Token("int", "1")),
        ((tok("int") + tok("name")) | (tok("op") + tok("int")) | assign, "a = 1", ("a", 1)),
        (tok("name") + anything_but(tok("op", ";")), "a 1;", ("a", Token("int", "1"))),
        (tok("name") + -tok("op", "=") + number.recover(tok("op", ";")), "a = 1;", ("a", 1)),
        (assign + finished, "a = 1", ("a", 1, None)),
    ],
)
def test_a_match_tells_the_end_of_its_tokens_without_reading_their_length(
    monkeypatch: pytest.MonkeyPatch, grammar: P, text: str, value: object
) -> None:
    monkeypatch.setattr(_TextTokens, "__len__", no_length)
    m = next(tokenized(grammar, tokenize).scan(text))
    assert m.value == value
