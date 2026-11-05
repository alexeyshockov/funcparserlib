import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from funcparserlib.lexer import LexerError, TokenSpec, _combined, make_tokenizer


@pytest.mark.parametrize("pattern", [r"a*", r"a?", r"", r"$"])
def test_a_spec_that_matches_the_empty_string_is_rejected(pattern: str) -> None:
    with pytest.raises(ValueError, match="matches the empty string"):
        make_tokenizer([TokenSpec("t", pattern)])


def test_a_legacy_tuple_spec_that_matches_the_empty_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="'t' matches the empty string"):
        make_tokenizer([("t", (r"\s*",))])


def test_an_empty_match_in_context_raises_instead_of_looping() -> None:
    tokenize = make_tokenizer([TokenSpec("boundary", r"\b"), TokenSpec("word", r"\w+")])
    with pytest.raises(ValueError, match=r"'boundary' matched the empty string at 1,1"):
        list(tokenize("ab"))


@pytest.mark.parametrize("separator", ["\x0c", "\r", "\x0b", "\x1c", "\x85", "\u2028"])
def test_the_error_names_the_line_the_position_counts(separator: str) -> None:
    tokenize = make_tokenizer(
        [TokenSpec("word", r"\w+"), TokenSpec("sep", separator), TokenSpec("nl", "\n")]
    )
    with pytest.raises(LexerError) as e:
        list(tokenize(f"a{separator}b\nc?"))
    assert e.value.place == (2, 2)
    assert e.value.msg == "c?"


def test_the_error_line_drops_a_crlf_ending() -> None:
    tokenize = make_tokenizer([TokenSpec("word", r"\w+"), TokenSpec("nl", r"\r\n")])
    with pytest.raises(LexerError) as e:
        list(tokenize("a\r\nb?\r\nc"))
    assert e.value.msg == "b?"


# Properties: any text over the specs' alphabet tokenizes, and each token knows where it is.

SPECS = [
    TokenSpec("space", r"[ \t]+"),
    TokenSpec("nl", r"\r?\n"),
    TokenSpec("word", r"[a-z]+"),
    TokenSpec("number", r"[0-9]+"),
    TokenSpec("op", r"[-+*/()]"),
]
TEXT = st.text(alphabet="abz019 \t\n\r+-*/()").filter(lambda s: "\r" not in s.replace("\r\n", ""))


def place(text: str, i: int) -> tuple[int, int]:
    """The 1-based (line, column) of `text[i]`, counting lines at "\\n"."""
    return text.count("\n", 0, i) + 1, i - text.rfind("\n", 0, i)


@given(TEXT)
def test_the_tokens_are_the_text(text: str) -> None:
    assert "".join(t.value for t in make_tokenizer(SPECS)(text)) == text


@given(TEXT)
def test_a_token_starts_where_its_text_starts_and_ends_on_its_last_character(text: str) -> None:
    i = 0
    for t in make_tokenizer(SPECS)(text):
        j = i + len(t.value)
        assert t.start == place(text, i)
        line, column = place(text, j)
        assert t.end == (line, column - 1)
        i = j


@given(TEXT, TEXT)
def test_the_error_is_at_the_first_character_no_spec_matches(before: str, after: str) -> None:
    text = f"{before}?{after}"
    with pytest.raises(LexerError) as e:
        list(make_tokenizer(SPECS)(text))
    assert e.value.place == place(text, len(before))


# The first spec that matches at a position makes the token, whatever the specs' groups and flags.

OVERLAPPING = [
    TokenSpec("kw", r"if|in"),
    TokenSpec("name", r"(?P<letters>[a-z]+)"),
    TokenSpec("quoted", r"(['\"])[a-z]*\1"),
    TokenSpec("upper", r"[a-z]+  # letters, any case", re.VERBOSE | re.IGNORECASE),
    TokenSpec("space", r"\s+"),
    TokenSpec("other", r"."),
]


def first_match(specs: list[TokenSpec], text: str) -> list[tuple[str, str]]:
    """The tokens of trying each spec in turn at each position."""
    compiled = [(spec.type, re.compile(spec.pattern, spec.flags)) for spec in specs]
    tokens, i = [], 0
    while i < len(text):
        token_type, m = next((t, m) for t, r in compiled if (m := r.match(text, i)))
        tokens.append((token_type, m.group()))
        i = m.end()
    return tokens


@pytest.mark.parametrize(
    "specs", [OVERLAPPING, OVERLAPPING[:2] + OVERLAPPING[3:]], ids=["back-reference", "joined"]
)
@given(st.text("ifnIFN'\" x\n", max_size=30))
def test_the_first_spec_to_match_makes_the_token(specs: list[TokenSpec], text: str) -> None:
    tokens = [(t.type, t.value) for t in make_tokenizer(specs)(text)]
    assert tokens == first_match(specs, text)


def test_specs_with_scoped_flags_join_into_one_pattern() -> None:
    compiled = [(s.type, re.compile(s.pattern, s.flags)) for s in OVERLAPPING]
    assert _combined(compiled[:2] + compiled[3:]) is not None
    assert _combined([("by_name", re.compile(r"(?P<q>')x(?P=q)"))]) is None


# A reference to a group moves with the group; what only looks like one stays as it is.

REFERENCES = [
    TokenSpec("pair", r"(a)(b)\2\1"),
    TokenSpec("quoted", r"(['\"])x*\1"),
    TokenSpec("twice", r"(?i:(y))\1", re.DOTALL),
    TokenSpec("backslash", r"\\1"),
    TokenSpec("octal", r"\101|\060|\01"),
    TokenSpec("class", r"[\1\]\[]"),
    TokenSpec("class_with_bracket", r"[]\1]"),
    TokenSpec("after_a_class", r"(b)[x[]\1"),
    TokenSpec("other", r"."),
]


@given(st.text("abxyYA01'\"\\[]\x01", max_size=30))
def test_specs_with_references_to_groups_tokenize_as_each_spec_does(text: str) -> None:
    tokens = [(t.type, t.value) for t in make_tokenizer(REFERENCES)(text)]
    assert tokens == first_match(REFERENCES, text)


@pytest.mark.parametrize(
    ("text", "token_type"),
    [
        ("abba", "pair"),
        ("'xx'", "quoted"),
        ("yy", "twice"),
        ("b[b", "after_a_class"),
        ("bxb", "after_a_class"),
    ],
)
def test_a_reference_finds_its_own_group(text: str, token_type: str) -> None:
    assert [(t.type, t.value) for t in make_tokenizer(REFERENCES)(text)] == [(token_type, text)]


def test_specs_with_references_to_numbered_groups_join_into_one_pattern() -> None:
    assert _combined([(s.type, re.compile(s.pattern, s.flags)) for s in REFERENCES]) is not None


@pytest.mark.parametrize(
    "spec",
    [
        TokenSpec("named", r"(?P<q>a)(b)\2"),
        TokenSpec("verbose", r"(a)\1  # [", re.VERBOSE),
        TokenSpec("condition", r"(a)?(?(1)b|c)"),
    ],
)
@given(st.text("abc", max_size=10))
def test_a_reference_that_cannot_move_stays_in_its_own_pattern(spec: TokenSpec, text: str) -> None:
    specs = [TokenSpec("first", r"(z)"), spec, TokenSpec("other", r".")]
    assert _combined([(s.type, re.compile(s.pattern, s.flags)) for s in specs]) is None
    assert [(t.type, t.value) for t in make_tokenizer(specs)(text)] == first_match(specs, text)


def test_a_reference_past_what_re_reads_as_one_stays_in_its_own_pattern() -> None:
    groups = "".join(f"({c})" for c in "abcdefghij" * 5)
    specs = [TokenSpec("many", groups), TokenSpec("more", groups), TokenSpec("ref", r"(q)\1")]
    assert _combined([(s.type, re.compile(s.pattern)) for s in specs]) is None
    assert [t.value for t in make_tokenizer(specs)("qq")] == ["qq"]


def test_render_marks_the_character_no_spec_matches() -> None:
    tokenize = make_tokenizer([TokenSpec("word", r"[a-z]+"), TokenSpec("nl", r"\n")])
    with pytest.raises(LexerError) as e:
        list(tokenize("ab\ncd?e\nf"))
    assert e.value.render() == "2,3: cannot tokenize data\n2 | cd?e\n      ^"
    assert e.value.render("ab\ncd?e\nf") == e.value.render()
