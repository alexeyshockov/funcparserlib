import pytest

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
