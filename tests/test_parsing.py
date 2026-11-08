import unittest

import pytest

from funcparserlib.lexer import LexerError, Token, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    IgnoredValue,
    NoParseError,
    Parser,
    a,
    finished,
    forward_decl,
    many,
    maybe,
    oneplus,
    skip,
    some,
    tok,
)
from funcparserlib.util import expand_tuple_args


class ParsingTest(unittest.TestCase):
    def test_oneplus(self) -> None:
        x = a("x")
        y = a("y")
        expr = oneplus(x + y)
        # noinspection SpellCheckingInspection
        assert expr.parse("xyxyxy") == [("x", "y"), ("x", "y"), ("x", "y")]

    # Issue 31
    def test_many_backtracking(self) -> None:
        x = a("x")
        y = a("y")
        expr = many(x + y) + x + x
        # noinspection SpellCheckingInspection
        assert expr.parse("xyxyxx") == ([("x", "y"), ("x", "y")], "x", "x")

    # Issue 14
    def test_error_info(self) -> None:
        tokenize = make_tokenizer(
            [
                TokenSpec("keyword", r"\b(is|end)\b"),
                TokenSpec("id", r"[a-z_]+"),
                ("space", (r"[ \t]+",)),  # Legacy token spec
                TokenSpec("nl", r"[\n\r]+"),
            ]
        )
        with pytest.raises(LexerError) as ctx:
            list(tokenize("f is ф"))
        assert str(ctx.value) == 'cannot tokenize data: 1,6: "f is ф"'

        @expand_tuple_args
        def make_equality(v1: str, v2: str) -> tuple[str, str]:
            return v1, v2

        tok_id = tok("id")
        equality = tok_id + -tok("keyword", "is") + tok_id >> make_equality
        expr = equality + -tok("nl")
        file = many(expr) + tok("keyword", "end")

        msg = """\
spam is eggs
foo is_not bar
end"""
        tokens = [x for x in tokenize(msg) if x.type != "space"]
        with pytest.raises(NoParseError) as ctx2:
            file.parse(tokens)
        assert ctx2.value.state.pos == 4
        assert ctx2.value.state.max == 5
        # May raise KeyError
        t = tokens[ctx2.value.state.max]
        assert t == Token("id", "is_not")
        assert (t.start, t.end) == ((2, 5), (2, 10))
        assert ctx2.value.msg == "2,5-2,10: got unexpected token: 'is_not', expected: 'is'"

    def test_ok_ignored(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, str] = -x + y
        assert expr.parse("xy") == "y"

    def test_skip_is_ignored(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, str] = skip(x) + y
        assert expr.parse("xy") == "y"

    def test_ignored_ok(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, str] = x + -y
        assert expr.parse("xy") == "x"

    def test_ignored_ok_ok(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, tuple[str, str]] = -x + y + x
        assert expr.parse("xyx") == ("y", "x")

    def test_ok_ignored_ok(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, tuple[str, str]] = x + -y + x
        assert expr.parse("xyx") == ("x", "x")

    def test_ok_ok_ok(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, tuple[str, str, str]] = x + y + x
        assert expr.parse("xyx") == ("x", "y", "x")

    def test_ok_ok_ignored(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, tuple[str, str]] = x + y + -x
        assert expr.parse("xyx") == ("x", "y")

    def test_ignored_ignored_ok(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, str] = -x + -x + y
        assert expr.parse("xxy") == "y"

    def test_ok_ignored_ignored(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, str] = x + -y + -y
        assert expr.parse("xyy") == "x"

    def test_ignored_ignored(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, IgnoredValue] = -x + -y
        assert expr.parse("xy") == IgnoredValue()

    def test_ignored_ignored_ignored(self) -> None:
        x = a("x")
        y = a("y")
        z = a("z")
        expr: Parser[str, IgnoredValue] = -x + -y + -z
        assert expr.parse("xyz") == IgnoredValue()

    def test_ignored_maybe(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, str] = -maybe(x) + y
        assert expr.parse("xy") == "y"
        assert expr.parse("y") == "y"

    def test_maybe_ignored(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, tuple[IgnoredValue | None, str]] = maybe(-x) + y
        assert expr.parse("xy") == "y"
        assert expr.parse("y") == (None, "y")

    def test_ignored_maybe_ignored(self) -> None:
        x = a("x")
        y = a("y")
        expr: Parser[str, str | None] = -x + maybe(y) + -x
        assert expr.parse("xyx") == "y"
        assert expr.parse("xx") is None

    def test_compare_token_with_none(self) -> None:
        # https://github.com/vlasovskikh/funcparserlib/pull/58
        specs = [("id", (r"\w+",))]
        tokenize = make_tokenizer(specs)
        tokens = list(tokenize("foo"))
        expr = maybe(a(None))
        # Tokens fed to a parser of `None` on purpose: the comparison must not raise.
        assert expr.parse(tokens) is None  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_seq_parse_error(self) -> None:
        expr = a("x") + a("y")
        with pytest.raises(NoParseError) as ctx:
            expr.parse("xz")
        assert ctx.value.msg == "got unexpected token: 'z', expected: 'y'"

    def test_alt_2_parse_error(self) -> None:
        expr = a("x") + (a("x") | a("y"))
        with pytest.raises(NoParseError) as ctx:
            expr.parse("xz")
        assert ctx.value.msg == "got unexpected token: 'z', expected: 'x' or 'y'"

    def test_alt_3_parse_error(self) -> None:
        expr = a("x") + (a("x") | a("y") | a("z"))
        with pytest.raises(NoParseError) as ctx:
            expr.parse("xa")
        assert ctx.value.msg == "got unexpected token: 'a', expected: 'x' or 'y' or 'z'"

    def test_alt_3_two_steps_parse_error(self) -> None:
        expr = a("x") + (a("x") | (a("y") + a("a")))
        with pytest.raises(NoParseError) as ctx:
            expr.parse("xyz")
        assert ctx.value.msg == "got unexpected token: 'z', expected: 'a'"

    def test_expected_eof_error(self) -> None:
        expr = a("x") + finished
        with pytest.raises(NoParseError) as ctx:
            expr.parse("xy")
        assert ctx.value.msg == "got unexpected token: 'y', expected: end of input"

    def test_expected_second_in_sequence_error(self) -> None:
        expr = a("x") + a("y")
        with pytest.raises(NoParseError) as ctx:
            expr.parse("xz")
        assert ctx.value.msg == "got unexpected token: 'z', expected: 'y'"

    def test_forward_decl_nested_matching_error(self) -> None:
        expr = forward_decl()
        expr.define(a("x") + maybe(expr) + a("y"))
        with pytest.raises(NoParseError) as ctx:
            expr.parse("xxy")
        assert ctx.value.msg == "got unexpected end of input, expected: 'y'"

    def test_expected_token_type_error(self) -> None:
        expr = tok("number")
        with pytest.raises(NoParseError) as ctx:
            expr.parse([Token("id", "x")])
        assert ctx.value.msg == "got unexpected token: 'x', expected: number"

    def test_expected_exact_token_error(self) -> None:
        expr = tok("operator", "=")
        with pytest.raises(NoParseError) as ctx:
            expr.parse([Token("operator", "+")])
        assert ctx.value.msg == "got unexpected token: '+', expected: '='"

    def test_unexpected_eof(self) -> None:
        expr = (a("x") + a("y")) | a("z")
        with pytest.raises(NoParseError) as ctx:
            expr.parse("x")
        assert ctx.value.msg == "got unexpected end of input, expected: 'y'"

    def test_expected_transform_parsing_results_error(self) -> None:
        expr = (a("1") >> int) | a("2")
        with pytest.raises(NoParseError) as ctx:
            expr.parse("x")
        assert ctx.value.msg == "got unexpected token: 'x', expected: '1' or '2'"

    def test_expected_sequence_with_skipped_parts(self) -> None:
        expr = (-a("x") + a("y")) | a("z")
        with pytest.raises(NoParseError) as ctx:
            expr.parse("b")
        assert ctx.value.msg == "got unexpected token: 'b', expected: 'x' or 'z'"

    def test_expected_some_without_name(self) -> None:
        def lowercase(t: str) -> bool:
            return t.islower()

        expr = some(lowercase)
        with pytest.raises(NoParseError) as ctx:
            expr.parse("A")
        assert ctx.value.msg == "got unexpected token: 'A', expected: some(...)"

    def test_expected_forward_decl_without_name(self) -> None:
        nested = forward_decl()
        nested.define(-a("a") + maybe(nested) + -a("z"))
        expr = nested | a("x")
        with pytest.raises(NoParseError) as ctx:
            expr.parse("y")
        assert ctx.value.msg == "got unexpected token: 'y', expected: 'a' or 'x'"

    def test_expected_forward_decl_with_name(self) -> None:
        nested = forward_decl().named("nested")
        nested.define(-a("a") + maybe(nested) + -a("z"))
        expr = nested | a("x")
        with pytest.raises(NoParseError) as ctx:
            expr.parse("y")
        assert ctx.value.msg == "got unexpected token: 'y', expected: nested or 'x'"

    def test_end_of_input_after_many_alternatives(self) -> None:
        brackets = a("[") + a("]")
        expr = many(a("x") | brackets) + finished
        with pytest.raises(NoParseError) as ctx:
            expr.parse("[")
        assert ctx.value.msg == "got unexpected end of input, expected: ']'"

    def test_parse_one_more_then_rollback_to_single(self) -> None:
        mul = a("x") + many(a("*") + a("y"))
        add = mul + many(a("+") + mul)
        expr = add + finished
        with pytest.raises(NoParseError) as ctx:
            expr.parse("x*")
        assert ctx.value.msg == "got unexpected end of input, expected: 'y'"

    def test_parse_one_more_then_rollback_to_alternative(self) -> None:
        mul = a("x") + many(a("*") + a("y"))
        addsub = mul + many((a("+") | a("-")) + mul)
        expr = addsub + finished
        with pytest.raises(NoParseError) as ctx:
            expr.parse("x*")
        assert ctx.value.msg == "got unexpected end of input, expected: 'y'"
