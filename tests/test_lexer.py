import pytest
from hypothesis import given
from hypothesis import strategies as st

from funcparserlib.lexer import LexerError, TokenSpec, make_tokenizer


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
