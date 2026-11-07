"""A token grammar bound to its tokenizer, as a parser of text."""

import gc
import weakref
from collections.abc import Callable
from typing import Any, cast

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from funcparserlib import binary as b
from funcparserlib._compile import (
    _Compiler,
    _form,
    _least_fixed_point,
    _scanned_nodes,
    _starts_of,
    _target,
)
from funcparserlib.lexer import LexerError, Token, Tokenizer, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    Invalid,
    Located,
    NoParseError,
    ParseErrors,
    Parser,
    ParsingSuccess,
    State,
    _Chain,
    _TextTokens,
    a,
    anything_but,
    commit,
    counted,
    finished,
    forward_decl,
    lit,
    longest,
    many,
    maybe,
    oneplus,
    prefixed,
    pure,
    skipping,
    some,
    switch,
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
    assert [(m.value, m.start, m.end) for m in tokenized(assign, tokenize).scan_text(text)] == [
        (("x", 1), 4, 9),
        (("y", 22), 14, 18),
        (("z", 3), 29, 34),
    ]


@given(texts)
def test_matches_are_those_of_the_grammar_at_each_offset(text: str) -> None:
    grammar = tokenized(assign, tokenize)
    for m in grammar.scan_text(text):
        assert grammar.parse_text(text[m.start : m.end]) == m.value
        assert not text[m.start].isspace()
        assert not text[m.end - 1].isspace()


def test_a_match_does_not_start_at_what_the_tokenizer_skips() -> None:
    matches = tokenized(total, tokenize).scan_text("  1 + 2 # c\n 3")
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


@pytest.mark.interpreted_only
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
    assert len(list(tokenized(assign, tokenize).scan_text(text))) == 200
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
    assert [m.value for m in grammar.scan_text(text)] == [("calc", (1, [2]))]
    assert grammar.compile().parse_text(text) == ("calc", (1, [2]))
    ref = weakref.ref(text)
    del text
    gc.collect()
    assert ref() is None


@pytest.mark.interpreted_only
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
    first = next(tokenized(statements, tokenize).scan_text(text))
    assert first == Located(("a", 1, [("b", "c")]), 4, 17)
    assert max(reads) < 40


def no_length(self: _TextTokens) -> int:
    """`len()` of the tokens of a text, which a parser asks for only to report an error."""
    msg = "a parser read the length of the tokens of a text"
    raise AssertionError(msg)


@pytest.mark.interpreted_only
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
    m = next(tokenized(grammar, tokenize).scan_text(text))
    assert m.value == value


def test_a_scan_that_reaches_a_tok_it_cannot_see_says_it_cannot_parse_text() -> None:
    hidden = Parser(lambda tokens, s: tok("name").run(tokens, s))
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse text"):
        list(hidden.scan_text("a"))
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse text"):
        hidden.parse_text("a")


@pytest.mark.interpreted_only
def test_a_parse_of_another_text_inside_a_scan_keeps_the_tokens_the_scan_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[tuple[int, int]] = []
    read = Tokenizer._read

    def counted(self: Tokenizer, s: str, i: int) -> tuple[str, int] | None:
        reads.append((id(s), i))
        return read(self, s, i)

    monkeypatch.setattr(Tokenizer, "_read", counted)
    inner = tokenized(number, tokenize) + finished
    text = "a = 1 " * 50
    outer = tokenized(
        tok("name") + -tok("op", "=") + (number >> (lambda n: inner.parse_text(str(n)))), tokenize
    )
    assert len(list(outer.scan_text(text))) == 50
    of_text = [i for s, i in reads if s == id(text)]
    assert len(of_text) == len(set(of_text))


def test_a_grammar_that_starts_with_a_bound_part_reads_the_skip_the_text_starts_with(
    run: Run,
) -> None:
    assert run(tokenized(number, tokenize) >> (lambda n: n * 2), "  # c\n 21") == 42
    assert run(tokenized(number, tokenize) + spaced.lit(";"), " 1 ;") == (1, ";")


words = make_tokenizer(
    [
        TokenSpec("space", r"\s+"),
        TokenSpec("comment", r"#[^\n]*"),
        TokenSpec("string", r'"[^"]*"'),
        TokenSpec("name", r"[a-z]+"),
        TokenSpec("op", r"="),
        TokenSpec("int", r"\d+"),
    ],
    skip={"space", "comment"},
)
setting = tok("name") + -tok("op") + tok("int")


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("x\n# was b = 2\nc = 3\n", [(("c", "3"), 14, 19)]),
        ('say "a = 1" then c = 3', [(("c", "3"), 17, 22)]),
        ("elif = 1", [(("elif", "1"), 0, 8)]),
        ("a = 1 ?? ! b = 2", [(("a", "1"), 0, 5), (("b", "2"), 11, 16)]),
    ],
)
def test_a_scan_of_a_bound_grammar_tries_the_tokens_of_the_text(
    compiled: bool, text: str, found: list[tuple[object, int, int]]
) -> None:
    grammar = tokenized(setting, words)
    scanner = grammar.compile() if compiled else grammar
    assert [(m.value, m.start, m.end) for m in scanner.scan_text(text)] == found


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "wrap",
    [
        lambda g: g >> (lambda v: v),
        lambda g: g.located() >> (lambda m: m.value),
        lambda g: g + -spaced.end,
    ],
)
def test_a_grammar_that_starts_with_a_bound_part_scans_the_tokens_of_the_text(
    compiled: bool, wrap: Callable[[P], P]
) -> None:
    grammar = wrap(tokenized(setting, words))
    scanner = grammar.compile() if compiled else grammar
    text = 'say "a = 1" then # b = 2\nc = 3'
    assert [(m.value, m.start) for m in scanner.scan_text(text)] == [(("c", "3"), 25)]


def test_replace_text_of_a_bound_grammar_leaves_strings_and_comments_as_they_are() -> None:
    grammar = tokenized(setting, words)
    text = 'a = 1 "b = 2" # c = 3\nd = 4'
    assert grammar.replace_text(text, lambda m: "=".join(m.value)) == 'a=1 "b = 2" # c = 3\nd=4'


def test_a_broken_construct_in_a_scan_of_a_bound_grammar_has_its_line_and_column() -> None:
    grammar = tokenized(tok("name") + commit(-tok("op") + tok("int")), words)
    with pytest.raises(NoParseError) as caught:
        list(grammar.scan_text("a = 1\nb = c"))
    assert str(caught.value).startswith("2,5-2,5: got unexpected token: 'c'")


def test_a_compiled_scan_of_a_bound_grammar_reads_only_as_far_as_its_first_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[int] = []
    read = Tokenizer._read

    def counted(self: Tokenizer, s: str, i: int) -> tuple[str, int] | None:
        reads.append(i)
        return read(self, s, i)

    compiled = tokenized(setting, words).compile()
    monkeypatch.setattr(Tokenizer, "_read", counted)
    assert next(compiled.scan_text("c = 3 " * 10_000)).value == ("c", "3")
    assert max(reads) < 20


def test_a_grammar_that_can_start_with_text_is_tried_at_each_offset() -> None:
    grammar = tokenized(setting, words) | spaced.lit("q")
    found = [(m.value, m.start) for m in grammar.scan_text('say "a = 1" q')]
    assert found == [(("a", "1"), 5), ("q", 12)]


bound = tokenized(number, tokenize)
shared = maybe(bound)
mark = spaced.lit("?")


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("grammar", "found"),
    [
        ((shared + bound) | (shared + mark), [((None, "?"), 3)]),
        (tokenized(many(number), tokenize) + mark, [(([], "?"), 3)]),
        (bound | counted(pure(1), mark), [(["?"], 3), (12, 5)]),
        (bound | prefixed(pure(1), mark), [("?", 3), (12, 5)]),
        (bound | switch(pure(0), {0: mark}), [("?", 3), (12, 5)]),
        (longest(maybe(bound) + mark, bound), [((None, "?"), 3), (12, 5)]),
        (many(bound) + mark, [(([], "?"), 3)]),
        (bound | (pure(0).bind(lambda _: mark)), [("?", 3), (12, 5)]),
    ],
)
def test_a_start_after_a_part_that_can_read_nothing_is_tried_at_each_offset(
    compiled: bool, grammar: P, found: list[tuple[object, int]]
) -> None:
    scanner = grammar.compile() if compiled else grammar
    assert [(m.value, m.start) for m in scanner.scan_text("ab ? 12")] == found


def test_a_recursive_grammar_with_a_start_of_text_is_tried_at_each_offset() -> None:
    nested = forward_decl()
    nested.define(-spaced.lit("(") + maybe(nested) + -spaced.lit(")") | bound)
    assert [(m.value, m.start) for m in nested.scan_text("ab (1) 2")] == [(1, 3), (2, 7)]


def nested(inner: P) -> P:
    """A recursive parser: `inner`, or a mark and itself."""
    rec = forward_decl()
    rec.define(inner | (mark + rec))
    return rec


Step = Callable[[P, P, bool], P]
steps: dict[str, Step] = {
    "bound": lambda _l, _r, real: bound if real else some(lambda _: False),
    "nothing": lambda _l, _r, real: tokenized(many(number), tokenize) if real else pure([]),
    "mark": lambda _l, _r, _real: mark,
    "pure": lambda _l, _r, _real: pure(None),
    "end": lambda _l, _r, _real: spaced.end,
    "seq": lambda left, right, _real: left + right,
    "alt": lambda left, right, _real: left | right,
    "longest": lambda left, right, _real: longest(left, right),
    "but_not": lambda left, right, _real: left.but_not(right),
    "maybe": lambda left, _r, _real: maybe(left),
    "many": lambda left, _r, _real: many(left),
    "oneplus": lambda left, _r, _real: oneplus(left),
    "counted": lambda left, _r, _real: counted(pure(1), left),
    "prefixed": lambda left, _r, _real: prefixed(pure(1), left),
    "switch": lambda left, _r, _real: switch(pure(0), {0: left}),
    "bind": lambda left, _r, _real: pure(0).bind(lambda _: left),
    "located": lambda left, _r, _real: left.located(),
    "nested": lambda left, _r, _real: nested(left),
}
Recipe = list[tuple[str, int, int]]
recipes = st.lists(
    st.tuples(st.sampled_from(sorted(steps)), st.integers(0, 3), st.integers(0, 3)),
    min_size=1,
    max_size=12,
)


def built(recipe: Recipe, *, bound_parts: bool) -> list[P]:
    """The parsers of `recipe`, each step on the parsers made before it, counted back from the last,
    so that steps share them.

    Without `bound_parts`, a bound part reads nothing where it can, and fails where it cannot.
    """
    made: list[P] = []
    for name, i, j in recipe:
        left = made[-1 - i % len(made)] if made else pure(None)
        right = made[-1 - j % len(made)] if made else pure(None)
        made.append(steps[name](left, right, bound_parts))
    return made


def reads(p: P, text: str, offset: int) -> bool | None:
    """Whether `p` reads something of `text` at `offset`, `None` where it fails."""
    try:
        res = p.run(cast("Any", text), State(offset, offset))
    except ValueError:
        return None
    return res.state.pos > offset if isinstance(res, ParsingSuccess) else None


@settings(max_examples=500, deadline=None)
@given(recipes, st.text(alphabet="1 ?a", max_size=8))
# A part reused after it read nothing, a bound part that reads nothing, a count that reads nothing.
@example(
    [("bound", 0, 0), ("maybe", 0, 0), ("mark", 0, 0), ("seq", 1, 2), ("seq", 2, 1), ("alt", 1, 0)],
    "a ?",
)
@example([("nothing", 0, 0), ("mark", 0, 0), ("seq", 1, 0)], "a ?")
@example([("bound", 0, 0), ("mark", 0, 0), ("counted", 0, 0), ("alt", 2, 0)], "a ?")
def test_the_starts_of_each_part_of_a_grammar_are_what_it_reads_first(
    recipe: Recipe, text: str
) -> None:
    real, rest = built(recipe, bound_parts=True), built(recipe, bound_parts=False)
    for part, without in zip(real, rest, strict=True):
        known = _least_fixed_point(_scanned_nodes(part), (frozenset(), False), _starts_of)
        starts, nullable = known[id(_target(part))]
        for offset in range(len(text)):
            # A part that reads nothing only where it says it can, and starts only where it says.
            assert nullable or reads(part, text, offset) is not False, offset
            if None not in starts:
                assert not reads(without, text, offset), offset


@pytest.mark.interpreted_only
def test_a_scan_inside_a_parse_of_the_same_text_reads_its_tokens_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[int] = []
    read = Tokenizer._read

    def counted(self: Tokenizer, s: str, i: int) -> tuple[str, int] | None:
        reads.append(i)
        return read(self, s, i)

    monkeypatch.setattr(Tokenizer, "_read", counted)
    text = "a = 1 " * 20
    inner = tokenized(assign, tokenize)
    pairs = tokenized(many(tok("name") + -tok("op") + number), tokenize)
    assert (pairs >> (lambda _: len(list(inner.scan_text(text))))).parse_text(text) == 20
    assert len(reads) == len(set(reads))


@pytest.mark.interpreted_only
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("text", ["1 + 2 + 3", "  # c\n 1 + 2"])
@pytest.mark.parametrize(
    "wrap", [lambda g: g >> (lambda v: v), lambda g: g + -spaced.end], ids=[">>", "+ end"]
)
def test_a_parse_of_a_grammar_that_starts_with_a_bound_part_reads_each_offset_once(
    monkeypatch: pytest.MonkeyPatch, compiled: bool, text: str, wrap: Callable[[P], P]
) -> None:
    reads: list[int] = []
    read = Tokenizer._read

    def counted(self: Tokenizer, s: str, i: int) -> tuple[str, int] | None:
        reads.append(i)
        return read(self, s, i)

    grammar = wrap(tokenized(total, tokenize))
    parser = grammar.compile() if compiled else grammar
    monkeypatch.setattr(Tokenizer, "_read", counted)
    parser.parse_text(text)
    assert reads
    assert len(reads) == len(set(reads))


def test_the_starts_of_a_scan_know_every_form_the_compiler_generates() -> None:
    def nothing(_tokens: Any, s: State) -> Any:
        return pure(None).run(_tokens, s)

    one = spaced.lit("a")
    forms = [
        one + one,
        one | one,
        one.but_not(one),
        b.byte(),
        commit(one),
        one.context("c"),
        counted(pure(1), one),
        finished,
        b.fixed(">H"),
        -one,
        one.located(),
        longest(one, one),
        many(one),
        one >> str.upper,
        oneplus(one),
        Parser(nothing),
        prefixed(pure(1)),
        b.previous(),
        pure(None),
        (one + one).starmap(lambda x, y: x + y),
        switch(pure(0), {0: one}),
        a("a"),
        lit("a"),
        spaced.end,
        bound,
        one.when(bool),
    ]
    helpers = {"call", "conversion", "function_call", "groups", "loop", "returning"}
    generated = {n.removeprefix("emit_") for n in dir(_Compiler) if n.startswith("emit_")}
    assert {_form(p)[0] for p in forms} == generated - helpers
    for p in forms:
        _least_fixed_point(_scanned_nodes(p), (frozenset(), False), _starts_of)


@pytest.mark.parametrize("compiled", [False, True])
def test_the_parser_but_not_rules_out_is_not_a_start_of_a_scan(compiled: bool) -> None:
    grammar = tokenized(tok("name"), words).but_not(lit("say"))
    scanner = grammar.compile() if compiled else grammar
    assert [(m.value, m.start) for m in scanner.scan_text('say "ab" c')] == [("c", 9)]


@pytest.mark.parametrize("compiled", [False, True])
def test_the_end_of_a_text_after_its_skip_is_a_start_of_a_scan(compiled: bool) -> None:
    grammar = maybe(bound) + -spaced.end
    scanner = grammar.compile() if compiled else grammar
    assert [(m.value, m.start) for m in scanner.scan_text("12 ?  ")] == [(None, 4)]
