"""`Parser.compile()`: a compiled parser parses as the parser it was compiled from."""

import ast
import importlib
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

import pytest
from hypothesis import assume, find, given, settings
from hypothesis import strategies as st

from funcparserlib import _compile as compile_module
from funcparserlib import parser as parser_module
from funcparserlib.lexer import LexerError, Token, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    ForwardDeclParser,
    NoParseError,
    ParseErrors,
    Parser,
    ParsingFailure,
    ParsingSuccess,
    State,
    a,
    commit,
    finished,
    forward_decl,
    lit,
    many,
    maybe,
    oneplus,
    pure,
    separated,
    skipping,
    some,
    tok,
    tokenized,
)
from tests.rules import breakages, conversions, kept
from tests.test_binary import grammars as byte_grammars
from tests.test_text import grammars as text_grammars

from . import dot, json
from .conftest import GeneratedCodeError

P: TypeAlias = Parser[Any, Any]


def tokens(types: str) -> list[Token]:
    return [Token(t, "x") for t in types]


def outcome(parse: Any, toks: Sequence[Any]) -> tuple[object, ...]:
    """What a parse gave, in a form that tells a `tuple` from what `+` builds."""
    try:
        return ("value", shape(parse(toks)))
    except ParseErrors as group:
        return ("errors", [str(e) for e in group.exceptions])
    except NoParseError as e:
        return ("error", str(e), e.expected, e.context, e.state.max)
    except (ValueError, NotImplementedError, AttributeError) as e:
        return (type(e), str(e))


def shape(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return (type(value), [shape(item) for item in value])
    return (type(value), value)


def wrap(value: object) -> tuple[str, object]:
    """The value in a tuple: a conversion to another shape."""
    return ("wrapped", value)


def same(value: object) -> object:
    """The value as it is: a conversion that changes nothing."""
    return value


def short(value: object) -> bool:
    """A rule that some values of any shape break."""
    return len(repr(value)) % 3 != 0


def recursive(body: P) -> P:
    """A grammar that reads `body` inside itself."""
    declared = forward_decl()
    declared.define(tok("a") + maybe(body + declared))
    return declared


def run_inner(inner: P) -> P:
    return Parser(inner.run)


leaves: st.SearchStrategy[P] = st.sampled_from("abc").flatmap(
    lambda t: st.sampled_from(
        [
            tok(t),
            tok(t, "xx"),
            tok(t, "yy"),
            -tok(t),
            tok(t) >> str.upper,
            -(tok(t) >> str.upper),
            some(lambda token: token.type == t),
            a(Token(t, "xx")),
            pure(None),
            pure("p"),
        ]
    )
)


def plain_combined(kids: st.SearchStrategy[P]) -> st.SearchStrategy[P]:
    """What generated code decides on its own."""
    pairs = st.tuples(kids, kids)
    return st.one_of(
        pairs.map(lambda k: k[0] + k[1]),
        pairs.map(lambda k: k[0] | k[1]),
        kids.map(lambda k: k + k),
        pairs.map(lambda k: k[0].but_not(k[1])),
        kids.map(lambda k: k >> same),
        kids.map(lambda k: k >> wrap),
        kids.map(lambda k: -k),
        kids.map(many),
        kids.map(maybe),
        kids.map(oneplus),
        st.tuples(kids, kids, st.booleans()).map(lambda k: separated(k[0], k[1], trailing=k[2])),
        kids.map(lambda k: k.when(short)),
        kids.map(lambda k: k.context("label")),
        kids.map(lambda k: k.located()),
        kids.map(lambda k: k.named("name")),
        kids.map(recursive),
        kids.map(lambda k: k + -finished),
        *conversions(kids, wrap),
    )


def combined(kids: st.SearchStrategy[P]) -> st.SearchStrategy[P]:
    """What the grammars are made of, generated code deciding it or not."""
    return st.one_of(
        plain_combined(kids), kids.map(commit), *breakages(kids, short), kids.map(run_inner)
    )


grammars = st.recursive(leaves, combined, max_leaves=12)
plain_grammars = st.recursive(leaves, plain_combined, max_leaves=12)
inputs = st.lists(
    st.builds(Token, st.sampled_from("abc"), st.sampled_from(["xx", "yy"])), max_size=7
)


@settings(max_examples=300, deadline=None)
@given(grammars, st.lists(inputs, min_size=4, max_size=4))
def test_a_compiled_parser_parses_as_its_parser(grammar: P, documents: list[list[Token]]) -> None:
    compiled = grammar.compile()
    for document in documents:
        assert outcome(compiled.parse, document) == outcome(grammar.parse, document)


@settings(max_examples=300, deadline=None)
@given(plain_grammars, st.lists(inputs, min_size=4, max_size=4))
def test_generated_code_decides_a_grammar_without_validated_or_commit(
    grammar: P, documents: list[list[Token]]
) -> None:
    fast = grammar.compile()._fast
    assume(fast is not None)
    assert fast is not None
    for document in documents:
        expected = outcome(grammar.parse, document)
        res = fast(document)
        if expected[0] == "value":
            assert res is not None
            assert ("value", shape(res[0])) == expected
        else:
            assert res is None


def test_a_name_generated_code_lacks_fails_a_test(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(compiler: Any, p: P, pos: str, want: bool, then: Any) -> None:
        compiler.add(ast.Expr(ast.Name("missing", ast.Load())))

    monkeypatch.setattr(compile_module._Compiler, "emit_finished", broken)
    with pytest.raises(GeneratedCodeError, match="missing"):
        finished.compile().parse([])


@pytest.mark.parametrize(
    ("grammar", "types", "value"),
    [
        (tok("a") + tok("b"), "ab", ("x", "x")),
        (tok("a") + tok("b") + tok("c"), "abc", ("x", "x", "x")),
        (tok("a") + (tok("b") + tok("c")), "abc", ("x", ("x", "x"))),
        (-tok("a") + tok("b"), "ab", "x"),
        (tok("a") + -tok("b"), "ab", "x"),
        (-tok("a") + tok("b") + -tok("c") + tok("a"), "abca", ("x", "x")),
        ((tok("a") + tok("b") | tok("b") + tok("a")) + tok("c"), "bac", ("x", "x", "x")),
        ((-tok("a") | tok("b")) + tok("c"), "ac", "x"),
        ((-tok("a") | tok("b")) + tok("c"), "bc", ("x", "x")),
        (tok("c") + (-tok("a") | tok("b")), "ca", "x"),
        (((tok("a") + tok("b")) >> same) + tok("c"), "abc", ("x", "x", "x")),
        (((tok("a") + tok("b")) >> list) + tok("c"), "abc", (["x", "x"], "x")),
        (maybe(tok("a") + tok("b")) + tok("c"), "c", (None, "x")),
        (maybe(tok("a") + tok("b")) + tok("c"), "abc", ("x", "x", "x")),
        ((tok("a") + tok("b")) + (many(tok("c")) >> same), "abc", ("x", "x", ["x"])),
        ((tok("a") + tok("b")) + (-tok("c") | tok("a")), "abc", ("x", "x")),
        ((tok("a") + tok("b")) + (-tok("c") | tok("a")), "aba", ("x", "x", "x")),
        ((tok("a") >> same) + (tok("b") + tok("c")), "abc", ("x", ("x", "x"))),
        (many(-tok("a")), "aa", [parser_module._IGNORED] * 2),
        (oneplus(tok("a") + tok("b")), "abab", [("x", "x"), ("x", "x")]),
        ((tok("a") + tok("b")).starmap(lambda x, y: x + y), "ab", "xx"),
    ],
)
def test_values_keep_their_shape(grammar: P, types: str, value: object) -> None:
    compiled = grammar.compile()
    assert compiled._fast is not None
    assert compiled._fast(tokens(types)) is not None
    assert compiled.parse(tokens(types)) == value


@pytest.fixture
def kdl() -> Any:
    """`examples.kdl`, a grammar of a real format, imported by the tests that run on its Python."""
    return importlib.import_module("examples.kdl")


@pytest.mark.parametrize(
    "text", ["a", 'a 1 "b" k=#true {\n  (t)c\n}', "/- a\nb /-1", "a k = (t)1; b\n"]
)
@pytest.mark.examples
def test_the_kdl_grammar_compiles(kdl: Any, text: str) -> None:
    compiled = kdl.document.compile()
    assert compiled._fast is not None
    toks = list(kdl.tokenize(text))
    fast = compiled._fast(toks)
    assert fast is not None
    assert fast[0] == kdl.document.parse(toks)


def test_the_test_grammars_compile() -> None:
    text = '{"a": [1, 2.5, "x", true, null], "b": {"c": []}}'
    toks = json.tokenize(text)
    assert json._grammar().compile().parse(toks) == json.parse(toks)
    source = 'digraph g { a -> b [label="e"]; subgraph s { c; } }'
    toks = dot.tokenize(source)
    assert dot._grammar().compile().parse(toks) == dot.parse(toks)


def test_a_compiled_parser_reads_nesting_too_deep_for_its_parser() -> None:
    nested = forward_decl()
    nested.define(-a("(") + maybe(nested) + -a(")"))
    text = "(" * 400 + ")" * 400
    with pytest.raises(RecursionError):
        nested.parse(text)
    assert nested.compile().parse(text) is None


def test_too_deep_for_both_raises_recursion_error() -> None:
    nested = forward_decl()
    nested.define(-a("(") + maybe(nested) + -a(")"))
    with pytest.raises(RecursionError):
        nested.compile().parse("(" * 5000 + ")" * 5000)


def test_an_error_is_the_error_of_the_parser() -> None:
    grammar = (tok("a") + tok("b")).context("pair") | tok("b").named("the b")
    toks = [Token("a", "x", (1, 1), (1, 1)), Token("c", "y", (1, 3), (1, 3))]
    with pytest.raises(NoParseError) as caught:
        grammar.compile().parse(toks)
    assert str(caught.value) == "1,3-1,3: got unexpected token: 'y', expected: b in pair"
    assert caught.value.render("a y") == "\n".join(
        ["1,3-1,3: got unexpected token: 'y', expected: b in pair", "1 | a y", "      ^"]
    )


def test_describe_reaches_the_error() -> None:
    def describe(token: str) -> parser_module.TokenDescription:
        return parser_module.TokenDescription(f"<{token}>")

    with pytest.raises(NoParseError, match="<y>"):
        a("x").compile().parse("y", describe=describe)


def test_a_broken_rule_is_reported_as_the_parser_reports_it() -> None:
    grammar = many(tok("a").validated(kept(lambda _: False, "bad")) + tok("b")) + -finished
    with pytest.raises(ParseErrors) as caught:
        grammar.compile().parse(tokens("abab"))
    assert [str(e) for e in caught.value.exceptions] == ["bad", "bad"]


@pytest.mark.interpreted_only
def test_a_rule_on_a_branch_backtracked_from_is_forgotten() -> None:
    seen: list[str] = []
    bad = (tok("a") >> seen.append).validated(kept(lambda _: False, "bad")) + tok("b")
    grammar = bad | tok("a") + tok("c")
    assert grammar.compile().parse(tokens("ac")) == ("x", "x")
    assert seen == ["x", "x"]


def test_a_commit_is_final() -> None:
    grammar = tok("a") + commit(tok("b") + tok("c")) | tok("a") + tok("b") + tok("a")
    with pytest.raises(NoParseError, match="expected: c"):
        grammar.compile().parse(tokens("aba"))


def test_many_of_a_parser_that_reads_nothing_raises_where_a_token_test_would_skip_it() -> None:
    grammar = pure(1) + many(pure(1)) + tok("c") | tok("b")
    with pytest.raises(ValueError, match="matched no tokens"):
        grammar.compile().parse(tokens("b"))


def test_many_of_a_custom_parser_that_reads_nothing_raises() -> None:
    nothing: P = Parser(lambda _toks, s: ParsingSuccess(1, s))
    with pytest.raises(ValueError, match="matched no tokens"):
        many(nothing).compile().parse(tokens("a"))


def test_an_alternative_that_can_read_nothing_is_tried_at_any_token() -> None:
    assert (many(tok("a")) | tok("b")).compile().parse(tokens("b")) == []
    assert (maybe(tok("a")) + tok("c") | tok("b")).compile().parse(tokens("b")) == "x"


def test_a_commit_in_a_parsing_function_is_final() -> None:
    committed = Parser(commit(tok("a") + tok("b")).run)
    grammar = committed | tok("a") + tok("c")
    with pytest.raises(NoParseError, match="expected: b"):
        grammar.compile().parse(tokens("ac"))


@pytest.mark.parametrize("define", [lambda e: e + tok("a") | tok("b"), lambda e: e + tok("a")])
def test_left_recursion_raises_recursion_error(define: Callable[[P], P]) -> None:
    expr = forward_decl()
    expr.define(define(expr))
    with pytest.raises(RecursionError):
        expr.compile().parse(tokens("b"))


def test_forward_decls_defined_as_each_other_raise_recursion_error() -> None:
    first = forward_decl()
    second = forward_decl()
    first.define(second)
    second.define(first)
    with pytest.raises(RecursionError):
        first.compile().parse(tokens("a"))


def sequence(items: Sequence[P]) -> P:
    grammar = items[0]
    for item in items[1:]:
        grammar += item
    return grammar


@pytest.mark.parametrize("size", [33, 100, 150, 400])
def test_a_long_sequence_compiles(size: int) -> None:
    grammar = sequence([tok("a")] * size)
    compiled = grammar.compile()
    assert compiled._fast is not None
    assert compiled._fast(tokens("a" * size)) is not None
    assert compiled.parse(tokens("a" * size)) == grammar.parse(tokens("a" * size))
    assert outcome(compiled.parse, tokens("a" * (size - 1))) == outcome(
        grammar.parse, tokens("a" * (size - 1))
    )


@pytest.mark.parametrize(
    "item",
    [-tok("a"), tok("a") >> same, tok("a") >> wrap, -tok("a") | tok("c"), tok("a") + tok("a")],
)
def test_a_long_sequence_joins_its_values_as_the_parser_does(item: P) -> None:
    grammar = sequence([tok("a"), -tok("a"), item] * 20)
    toks = tokens("a" * 60)
    compiled = grammar.compile()
    assert outcome(compiled.parse, toks * 2) == outcome(grammar.parse, toks * 2)


def test_a_long_sequence_of_ignored_values_compiles() -> None:
    grammar = sequence([-tok("a")] * 70)
    compiled = grammar.compile()
    assert compiled._fast is not None
    assert outcome(compiled.parse, tokens("a" * 70)) == outcome(grammar.parse, tokens("a" * 70))


def test_deeply_nested_loops_compile() -> None:
    grammar: P = tok("a")
    for _ in range(25):
        grammar = many(grammar + tok("b")) + tok("a")
    compiled = grammar.compile()
    assert compiled._fast is not None
    assert outcome(compiled.parse, tokens("a")) == outcome(grammar.parse, tokens("a"))


def test_deeply_nested_options_compile() -> None:
    grammar: P = tok("a")
    for _ in range(200):
        grammar = tok("a") + maybe(grammar)
    compiled = grammar.compile()
    assert compiled._fast is not None
    assert outcome(compiled.parse, tokens("a" * 201)) == outcome(grammar.parse, tokens("a" * 201))


def generated_module(grammar: P) -> ast.Module:
    internals = compile_module.Internals(
        parser_module._Tuple,
        parser_module._IGNORED,
        parser_module._ignore,
        parser_module._token_value,
        parser_module._Undecided,
        parser_module._located_at,
    )
    module, _ = compile_module._Compiler(grammar, internals).module()
    return module


def generated_statements(grammar: P) -> int:
    return sum(isinstance(node, ast.stmt) for node in ast.walk(generated_module(grammar)))


def nested_oneplus(levels: int) -> P:
    grammar: P = tok("a")
    for _ in range(levels):
        grammar = oneplus(grammar + tok("b"))
    return grammar


def test_generated_code_grows_with_the_nesting_of_oneplus() -> None:
    assert generated_statements(nested_oneplus(12)) < 3 * generated_statements(nested_oneplus(6))


def statement_lines(module: ast.Module) -> list[tuple[str, int]]:
    statements = (node for node in ast.walk(module) if isinstance(node, ast.stmt))
    return [(type(statement).__name__, statement.lineno) for statement in statements]


@settings(max_examples=200, deadline=None)
@given(grammars)
def test_generated_statements_have_the_lines_of_the_source(grammar: P) -> None:
    try:
        module = compile_module._with_lines(generated_module(grammar))
    except compile_module._Declined:
        assume(False)
        raise
    assert statement_lines(module) == statement_lines(ast.parse(ast.unparse(module)))


@pytest.mark.examples
def test_the_lines_of_a_whole_grammar_are_those_of_its_source(kdl: Any) -> None:
    module = compile_module._with_lines(generated_module(kdl.document))
    assert statement_lines(module) == statement_lines(ast.parse(ast.unparse(module)))


def frame_lines(path: Path, lines: list[tuple[str, int]]) -> None:
    """Assert each frame of `lines` reports `path`, on a line of it that calls a function."""
    text = path.read_text(encoding="utf-8").splitlines()
    assert lines
    for filename, line in lines:
        assert filename == str(path)
        assert re.fullmatch(r"\s*v\d+ = F\d+\(.*\)", text[line - 1])


def recording(lines: list[tuple[str, int]]) -> Callable[[object], object]:
    def seen(value: object) -> object:
        frame = sys._getframe(1)
        lines.append((frame.f_code.co_filename, frame.f_lineno))
        return value

    return seen


def source_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_a_source_file_holds_the_source_and_frames_report_it(tmp_path: Path) -> None:
    path = tmp_path / "grammar.py"
    lines: list[tuple[str, int]] = []
    declared = forward_decl()
    declared.define((tok("a") >> recording(lines)) + maybe(declared))
    compiled = (Parser(tok("b").run) + declared).compile(source_file=path)
    assert compiled._fast is not None
    assert compiled._fast(tokens("baa")) is not None
    assert source_of(path).startswith("def f1(T, Y, p0):\n")
    frame_lines(path, lines)


def test_a_source_file_holds_both_programs_of_a_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "grammar.py"
    lines: list[tuple[str, int]] = []
    compiled = tokenized(many(tok("a") >> recording(lines)), tokenizer).compile(source_file=path)
    assert compiled._fast_text is not None
    assert compiled._bound is not None
    fast = compiled._bound[2]
    assert fast is not None
    assert compiled._fast_text("aa") is not None
    from_text = list(lines)
    lines.clear()
    assert fast(list(tokenizer("aa"))) is not None
    frame_lines(path, from_text)
    frame_lines(path, lines)
    text = source_of(path).splitlines()
    assert ".value" not in text[from_text[0][1] - 1]
    assert ".value" in text[lines[0][1] - 1]


def test_a_source_file_holds_what_parse_runs_without_functions_for_text(tmp_path: Path) -> None:
    path = tmp_path / "grammar.py"
    compiled = tokenized(some(lambda token: token.type == "a"), tokenizer).compile(source_file=path)
    assert compiled._fast_text is None
    assert source_of(path).count("def f") == 1


def test_a_source_file_holds_a_grammar_of_text(tmp_path: Path) -> None:
    path = tmp_path / "grammar.py"
    many(lit("a")).compile(source_file=path)
    assert "T.startswith('a'" in source_of(path)


def test_a_source_file_is_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "grammar.py"
    path.write_text("old", encoding="utf-8")
    tok("a").compile(source_file=path)
    assert source_of(path).startswith("def f1(")


def test_a_grammar_that_stays_interpreted_writes_no_source_file(tmp_path: Path) -> None:
    path = tmp_path / "grammar.py"
    many(maybe(tok("a"))).compile(source_file=path)
    assert not path.exists()


def test_a_relative_source_file_is_absolute_in_a_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lines: list[tuple[str, int]] = []
    compiled = (tok("a") >> recording(lines)).compile(source_file="grammar.py")
    assert compiled._fast is not None
    assert compiled._fast(tokens("a")) is not None
    frame_lines(tmp_path.resolve() / "grammar.py", lines)


def test_a_module_too_deep_to_unparse_stays_interpreted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_deep(module: ast.AST) -> str:
        raise RecursionError

    monkeypatch.setattr(ast, "unparse", too_deep)
    path = tmp_path / "grammar.py"
    assert tok("a").compile(source_file=path)._fast is None
    assert not path.exists()


def test_a_source_file_that_cannot_be_written_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tok("a").compile(source_file=tmp_path / "missing" / "grammar.py")


def test_a_grammar_too_deep_to_compile_stays_interpreted() -> None:
    grammar = sequence([a("x")] * 2000)
    compiled = grammar.compile()
    assert compiled._fast is None
    with pytest.raises(RecursionError):
        compiled.parse("x" * 2000)


class Kind(StrEnum):
    A = "a"
    B = "b"


@pytest.mark.parametrize(
    "grammar",
    [
        tok(Kind.A),
        tok(Kind.A, "x"),
        tok("a", Kind.A),
        many(tok(Kind.A) + tok(Kind.B)),
        maybe(tok(Kind.A) + tok(Kind.B)) | tok(Kind.B) + tok(Kind.A),
    ],
)
def test_a_str_subclass_compiles_as_a_token_type_or_value(grammar: P) -> None:
    compiled = grammar.compile()
    assert compiled._fast is not None
    for text in ("ab", "a"):
        toks = [Token(t, "a" if t == "x" else "x") for t in text]
        assert outcome(compiled.parse, toks) == outcome(grammar.parse, toks)


@pytest.mark.interpreted_only
@pytest.mark.parametrize(
    ("build", "text"),
    [
        (lambda f: many(commit(a("x") + a("y")) >> f) + -finished, "xyxy"),
        (lambda f: many(commit(some(str.isalpha) + a("1")) >> f) + -finished, "x1y1"),
        (lambda f: many(commit(maybe(a("x")) + a("y")) >> f) + -finished, "xyy"),
        (lambda f: many(commit(a("x") | a("y")) >> f) + -finished, "xy"),
        (lambda f: (commit(a("x") + a("y")) >> f) | (a("z") >> f), "z"),
        (lambda f: (commit(many(a("x")) + a("y")) >> f) + -finished, "xxy"),
    ],
)
def test_a_commit_that_read_no_token_ends_a_loop_or_a_choice_in_the_compiled_pass(
    build: Callable[[Callable[[object], object]], P], text: str
) -> None:
    seen: list[object] = []

    def record(value: object) -> object:
        seen.append(value)
        return value

    grammar = build(record)
    expected = grammar.parse(text)
    calls = len(seen)
    seen.clear()
    assert grammar.compile().parse(text) == expected
    assert len(seen) == calls


def test_a_commit_of_a_parsing_function_that_read_a_token_is_final() -> None:
    inner = Parser((a("x") + a("y")).run)
    grammar = commit(inner) | a("x") + a("z")
    with pytest.raises(NoParseError, match="expected: 'y'"):
        grammar.compile().parse("xz")


def test_a_commit_is_final_after_a_token_read_past_an_optional_start() -> None:
    grammar = commit(maybe(a("x")) + a("y") + a("w")) | a("y") + a("v")
    with pytest.raises(NoParseError, match="expected: 'w'"):
        grammar.compile().parse("yv")


def test_a_commit_of_a_parsing_function_leaves_every_failure_to_the_parser() -> None:
    inner = Parser(a("x").run)
    assert (many(commit(inner)) + a("y")).compile().parse("xy") == (["x"], "y")


def test_a_parsing_function_runs_inside_a_compiled_grammar() -> None:
    def two(toks: Sequence[str], s: State) -> ParsingSuccess[str] | ParsingFailure:
        if toks[s.pos : s.pos + 2] == "ab":
            return ParsingSuccess("two", s.reset_pos(s.pos + 2))
        return ParsingFailure(s)

    grammar = Parser(two) + a("c")
    compiled = grammar.compile()
    assert compiled.parse("abc") == ("two", "c")
    with pytest.raises(NoParseError):
        compiled.parse("acc")


def test_a_function_written_for_1_0_runs_inside_a_compiled_grammar() -> None:
    def legacy(toks: Sequence[str], s: State) -> tuple[str, State]:
        return toks[s.pos], s.reset_pos(s.pos + 1)

    grammar = Parser(cast("Any", legacy)) + a("c")
    with pytest.warns(DeprecationWarning, match="1.0 protocol"):
        assert grammar.compile().parse("bc") == ("b", "c")


def test_bind_runs_inside_a_compiled_grammar() -> None:
    twice = a("x").bind(a) + a("y")
    assert twice.compile().parse("xxy") == ("x", "y")


def test_a_grammar_of_any_tokens_compiles() -> None:
    compiled = (many(a("x")) + some(str.isdigit)).compile()
    assert compiled._fast is not None
    assert compiled.parse("xx1") == (["x", "x"], "1")


def test_tokens_of_another_type_are_left_to_the_parser() -> None:
    assert outcome(tok("a").compile().parse, "a") == outcome(tok("a").parse, "a")


def test_an_object_like_a_token_is_no_token() -> None:
    @dataclass
    class Lookalike:
        type: str
        value: str

    grammar = tok("a", "x")
    toks: list[Any] = [Lookalike("a", "x")]
    with pytest.raises(NoParseError):
        grammar.parse(toks)
    with pytest.raises(NoParseError):
        grammar.compile().parse(toks)


def test_an_undefined_forward_decl_raises_as_in_the_parser() -> None:
    with pytest.raises(NotImplementedError, match="define"):
        forward_decl().compile().parse("x")


def test_a_compiled_grammar_parses_more_than_once() -> None:
    compiled = many(tok("a")).compile()
    assert compiled.parse(tokens("aa")) == ["x", "x"]
    assert compiled.parse(tokens("a")) == ["x"]


def test_parsing_stops_where_the_parser_does() -> None:
    assert tok("a").compile().parse(tokens("ab")) == "x"


def test_a_successful_parse_calls_a_conversion_once() -> None:
    seen: list[str] = []
    (tok("a") >> seen.append).compile().parse(tokens("a"))
    assert seen == ["x"]


@pytest.mark.interpreted_only
def test_a_failed_parse_calls_a_conversion_in_both_passes() -> None:
    seen: list[str] = []
    grammar = (tok("a") >> seen.append) + tok("b")
    with pytest.raises(NoParseError):
        grammar.compile().parse(tokens("ac"))
    assert seen == ["x", "x"]


def test_a_forward_decl_defined_again_keeps_its_meaning_in_the_compiled_pass() -> None:
    declared = forward_decl()
    declared.define(tok("a"))
    compiled = (declared + -finished).compile()
    declared.define(tok("b"))
    assert compiled.parse(tokens("a")) == "x"


def test_debug_logs_as_the_parser_does(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(parser_module, "debug", True)
    compiled = (tok("a") + tok("b")).compile()
    with caplog.at_level(logging.DEBUG, logger="funcparserlib"):
        assert compiled.parse(tokens("ab")) == ("x", "x")
    assert "trying" in caplog.text


# Text parsed without `Token` objects: `parse_text()` gives what the tokenizer and the parser give.

TEXT_SPECS = [
    TokenSpec("space", r"[ \n]+"),
    TokenSpec("a", r"a+"),
    TokenSpec("b", r"b"),
    TokenSpec("c", r"(c)x\1"),
    TokenSpec("empty_in_context", r"(?=z)"),
]
tokenizer = make_tokenizer(TEXT_SPECS, skip={"space"})


def parse_tokens(grammar: P, text: str) -> object:
    return grammar.parse(list(tokenizer(text)))


def text_outcome(parse: Any, text: str) -> tuple[object, ...]:
    """The outcome, every failure at text the tokenizer stops at being the same one."""
    try:
        return outcome(parse, text)
    except LexerError:
        return ("no such tokens",)


def token_leaves() -> st.SearchStrategy[P]:
    return st.sampled_from("abc").flatmap(
        lambda t: st.sampled_from(
            [tok(t), tok(t, "aa"), tok(t, "b"), -tok(t), tok(t) >> str.upper, pure(None), pure("p")]
        )
    )


def plain_combined_without_tokens(kids: st.SearchStrategy[P]) -> st.SearchStrategy[P]:
    """What generated code decides on its own."""
    pairs = st.tuples(kids, kids)
    return st.one_of(
        pairs.map(lambda k: k[0] + k[1]),
        pairs.map(lambda k: k[0] | k[1]),
        pairs.map(lambda k: k[0].but_not(k[1])),
        kids.map(lambda k: k >> wrap),
        kids.map(lambda k: -k),
        kids.map(many),
        kids.map(maybe),
        kids.map(oneplus),
        kids.map(lambda k: k.when(short)),
        kids.map(recursive),
        kids.map(lambda k: k + -finished),
        *conversions(kids, wrap),
    )


def combined_without_tokens(kids: st.SearchStrategy[P]) -> st.SearchStrategy[P]:
    return st.one_of(plain_combined_without_tokens(kids), kids.map(commit), *breakages(kids, short))


token_grammars = st.recursive(token_leaves(), combined_without_tokens, max_leaves=12)
plain_token_grammars = st.recursive(token_leaves(), plain_combined_without_tokens, max_leaves=12)
texts = st.text("ab cx\nz?", max_size=12)


@settings(max_examples=300, deadline=None)
@given(token_grammars, st.lists(texts, min_size=4, max_size=4))
def test_parse_text_parses_as_the_tokenizer_and_the_parser(
    grammar: P, documents: list[str]
) -> None:
    compiled = tokenized(grammar, tokenizer).compile()
    for text in documents:
        expected = text_outcome(lambda t: parse_tokens(grammar, t), text)
        got = text_outcome(compiled.parse_text, text)
        if expected == ("no such tokens",):
            assert got[0] in ("error", "errors", ValueError) or got == expected
        else:
            assert got == expected


@settings(max_examples=300, deadline=None)
@given(plain_token_grammars, st.lists(texts, min_size=4, max_size=4))
def test_generated_code_decides_the_tokens_of_a_text(grammar: P, documents: list[str]) -> None:
    fast = tokenized(grammar, tokenizer).compile()._fast_text
    assume(fast is not None)
    assert fast is not None
    for text in documents:
        try:
            toks = list(tokenizer(text))
        except (LexerError, ValueError):
            continue
        expected = outcome(grammar.parse, toks)
        if not isinstance(expected[0], str):
            with pytest.raises(parser_module._Undecided):
                fast(text)
            continue
        res = fast(text)
        if expected[0] == "value":
            assert res is not None
            assert ("value", shape(res[0])) == expected
        else:
            assert res is None


def test_parse_text_makes_no_tokens() -> None:
    compiled = tokenized((many(tok("a")) + tok("b")), tokenizer).compile()
    assert compiled._fast_text is not None
    assert compiled._fast_text("aa b") == ((["aa"], "b"), 2)


@pytest.mark.parametrize("text", ["a b", "aa\n b  a"])
def test_parse_text_leaves_out_the_skipped_types(text: str) -> None:
    grammar = many(tok("a") | tok("b"))
    expected = grammar.parse(list(tokenizer(text)))
    assert tokenized(grammar, tokenizer).compile().parse_text(text) == expected


def test_parse_text_says_what_was_expected_at_text_no_spec_matches() -> None:
    compiled = tokenized((many(tok("a")) + tok("b")), tokenizer).compile()
    with pytest.raises(NoParseError) as e:
        compiled.parse_text("aa\na ?x")
    assert str(e.value) == "2,3-2,3: got unexpected token: '?x', expected: a or b"
    assert e.value.expected == ("a", "b")
    assert isinstance(e.value.__cause__, LexerError)
    assert e.value.render("aa\na ?x").endswith("2 | a ?x\n      ^")


def test_parse_text_reports_a_mistake_before_text_no_spec_matches() -> None:
    compiled = tokenized((tok("a") + tok("b")), tokenizer).compile()
    with pytest.raises(NoParseError, match="1,3-1,3: got unexpected token: 'a', expected: b"):
        compiled.parse_text("a a ?")


def test_parse_text_says_what_was_expected_after_a_complete_document() -> None:
    compiled = tokenized((tok("a") + -finished), tokenizer).compile()
    with pytest.raises(NoParseError, match=r"1,3-1,3: .*'\?', expected: end of input"):
        compiled.parse_text("a ?")


def test_parse_text_keeps_a_rule_error_with_the_error_at_text_no_spec_matches() -> None:
    compiled = tokenized(
        (tok("a").validated(kept(lambda _: False, "bad")) + tok("b")), tokenizer
    ).compile()
    with pytest.raises(ParseErrors) as e:
        compiled.parse_text("a ?")
    assert [str(x) for x in e.value.exceptions] == [
        "1,1-1,1: bad",
        "1,3-1,3: got unexpected token: '?', expected: b",
    ]


def test_parse_text_raises_the_lexer_error_when_the_parser_stops_before_it() -> None:
    with pytest.raises(LexerError):
        tokenized(tok("a"), tokenizer).compile().parse_text("a ?")


def test_parse_text_raises_the_error_of_the_parser_with_positions() -> None:
    compiled = tokenized((tok("a") + tok("b")), tokenizer).compile()
    with pytest.raises(NoParseError, match="1,3-1,3: got unexpected token: 'a', expected: b"):
        compiled.parse_text("a a")


def test_parse_text_of_a_pattern_matching_nothing_raises_as_the_tokenizer_does() -> None:
    with pytest.raises(ValueError, match="matched the empty string"):
        tokenized(many(tok("a")), tokenizer).compile().parse_text("az")


def test_describe_reaches_an_error_of_parse_text() -> None:
    compiled = tokenized(tok("a"), tokenizer).compile()
    with pytest.raises(NoParseError, match="<b>"):
        compiled.parse_text("b", describe=lambda t: parser_module.TokenDescription(f"<{t.value}>"))


@pytest.mark.interpreted_only
def test_a_failed_parse_of_text_calls_a_conversion_in_two_passes() -> None:
    seen: list[str] = []
    with pytest.raises(NoParseError):
        tokenized(((tok("a") >> seen.append) + tok("b")), tokenizer).compile().parse_text("a a")
    assert seen == ["a", "a"]


a_token: Parser[Token, Token] = some(lambda t: t.type == "a")


@pytest.mark.parametrize(
    "grammar", [a_token + tok("b"), a(Token("a", "a")) + tok("b"), Parser(tok("a").run) + tok("b")]
)
def test_a_grammar_that_reads_tokens_parses_text_through_tokens(grammar: P) -> None:
    compiled = tokenized(grammar, tokenizer).compile()
    assert compiled._fast_text is None
    assert compiled.parse_text("a b") == grammar.parse(list(tokenizer("a b")))


def test_specs_that_do_not_join_parse_text_through_tokens() -> None:
    tokenize = make_tokenizer([TokenSpec("q", r"(?P<q>a)(?P=q)"), TokenSpec("b", r"b")])
    compiled = tokenized((tok("q") + tok("b")), tokenize).compile()
    assert compiled._fast_text is None
    assert compiled.parse_text("aab") == ("aa", "b")


def test_parse_text_of_tokens_without_a_tokenizer_raises() -> None:
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse text"):
        tok("a").compile().parse_text("a")


def test_parse_text_with_debug_runs_no_generated_code(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = tokenized((tok("a") + tok("b")), tokenizer).compile()
    assert compiled._fast_text is not None

    def fail(_text: str) -> None:
        pytest.fail("generated code ran under debug")

    monkeypatch.setattr(compiled, "_fast_text", fail)
    monkeypatch.setattr(parser_module, "debug", True)
    assert compiled.parse_text("a b") == ("a", "b")


def test_a_parser_reading_past_the_last_token_stays_in_the_compiled_pass() -> None:
    compiled = tokenized(many(tok("a")), tokenizer).compile()
    assert compiled._fast_text is not None
    assert compiled._fast_text("a a") == (["a", "a"], 2)


def test_a_pattern_matching_nothing_before_a_token_raises_as_the_tokenizer_does() -> None:
    tokenize = make_tokenizer(
        [TokenSpec("a", r"a"), TokenSpec("gap", r"(?=z)"), TokenSpec("z", r"z")]
    )
    compiled = tokenized(many(tok("a") | tok("z")), tokenize).compile()
    with pytest.raises(ValueError, match="matched the empty string"):
        compiled.parse_text("az")


@pytest.mark.examples
def test_the_kdl_grammar_parses_text_without_tokens(kdl: Any) -> None:
    compiled = tokenized(kdl.document, kdl.tokenize).compile()
    assert compiled._fast_text is not None
    text = 'server "web" port=8080 {\n  (t)route "/api" #true\n}\n'
    assert compiled.parse_text(text) == kdl.parse(text)


# A rule used twice gets a function, and what `+` builds from its value depends on what is known of
# it. A value of several characters tells a tuple from text that was unpacked as one.


def word_tokens(types: str) -> list[Token]:
    return [Token(t, "word") for t in types]


def shared_rule(last: P) -> P:
    rule = tok("a")
    for _ in range(11):
        rule += -tok("a")
    return rule + last


@pytest.mark.parametrize(
    ("last", "tail"),
    [
        (tok("b"), "b"),
        (-tok("b"), "b"),
        (tok("b") >> same, "b"),
        (tok("b") >> wrap, "b"),
        (-tok("b") | tok("c"), "b"),
        (-tok("b") | tok("c"), "c"),
        (many(tok("b")) >> same, "bb"),
    ],
)
def test_a_value_crosses_the_function_of_a_rule_used_twice(last: P, tail: str) -> None:
    rule = shared_rule(last)
    once = word_tokens("a" * 12 + tail)
    end = word_tokens("d")
    for grammar, document in [
        (rule + rule, once + once),
        (rule + rule + tok("d"), once + once + end),
        ((rule | rule) + tok("d"), once + end),
        (tok("d") + rule + rule, end + once + once),
    ]:
        assert outcome(grammar.compile().parse, document) == outcome(grammar.parse, document)


# What a regex can start with


def chars(text: str, *, negated: bool = False) -> compile_module._Chars:
    return compile_module._Chars(frozenset(text), negated=negated)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("abc", chars("a")),
        ("a|b", chars("ab")),
        ("a*b", chars("ab")),
        ("a*?b", chars("ab")),
        ("a++b", chars("a")),
        ("a*+b", chars("ab")),
        ("(?>a)b", chars("a")),
        ("a?", chars("a")),
        ("[a-c]x", chars("abc")),
        ("[^ab]", chars("ab", negated=True)),
        ("(?:a|[^b])", chars("b", negated=True)),
        ("(?:[^ab]|[^bc])", chars("b", negated=True)),
        ("(?=b)c", chars("c")),
        ("(?<=x)c", chars("c")),
        (r"\bc", chars("c")),
        (r"(a)\1", chars("a")),
        ("a(?i:b)", chars("a")),
        (r"\d", None),
        (r"[\d]", None),
        (".", None),
        ("(?i)a", None),
        ("(?i:a)b", None),
        (r"(a)?(?(1)b|c)", None),
        (r"[\x00-\U0010ffff]", None),
    ],
)
def test_the_first_characters_of_a_regex(
    pattern: str, expected: compile_module._Chars | None
) -> None:
    assert compile_module._regex_chars(re.compile(pattern)) == expected


def test_a_regex_with_a_flag_has_no_first_characters() -> None:
    assert compile_module._regex_chars(re.compile("a", re.IGNORECASE)) is None


def test_a_private_parser_that_changed_leaves_a_regex_without_first_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_: object) -> None:
        raise AttributeError

    monkeypatch.setattr("re._parser.parse", broken)
    assert compile_module._regex_chars(re.compile("a")) is None


regex_atoms = st.sampled_from(
    ["a", "b", "A", "[ab]", "[^a]", "[a-c]", ".", r"\d", r"\w", "(?=b)", "(?!a)", "(?<=a)", r"\b"]
    + ["^", "$", r"\Z", "(?i:a)", "(?s:.)"]
)


def regex_forms(kids: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    pairs = st.tuples(kids, kids)
    return st.one_of(
        pairs.map("".join),
        pairs.map("|".join),
        kids.map("({})".format),
        kids.map("(?:{})".format),
        kids.map("(?>{})".format),
        kids.map("(?i:{})".format),
        kids.map("(?:{})*".format),
        kids.map("(?:{})+".format),
        kids.map("(?:{})?".format),
        kids.map("(?:{})*?".format),
        kids.map("(?:{})*+".format),
        kids.map("(?:{}){{0}}".format),
        kids.map("(?:{}){{2}}".format),
    )


@settings(max_examples=500, deadline=None)
@given(st.recursive(regex_atoms, regex_forms, max_leaves=6), st.text("abcAB1 \n", max_size=6))
def test_a_match_of_a_regex_starts_with_one_of_its_first_characters(
    pattern: str, text: str
) -> None:
    regex = re.compile(pattern)
    first = compile_module._regex_chars(regex)
    assume(first is not None)
    assert first is not None
    for start in range(len(text)):
        m = regex.match(text, start)
        if m is not None and m.end() > start:
            assert (text[start] in first.chars) != first.negated


def test_a_repeated_parser_that_matches_nothing_raises_as_the_interpreted_parser_does() -> None:
    spaced = skipping(r"\s+")
    grammar: ForwardDeclParser[str, Any] = forward_decl()
    grammar.define(spaced.regex("a") + maybe(many(many(spaced.regex("a"))) + grammar))
    with pytest.raises(ValueError, match="matched no tokens inside many"):
        grammar.parse_text("a")
    with pytest.raises(ValueError, match="matched no tokens inside many"):
        grammar.compile().parse_text("a")


def forms_of(p: P) -> set[str]:
    """The forms in the grammar `p`, as the compiler reads them."""
    found: set[str] = set()
    seen: set[int] = set()
    todo = [p]
    while todo:
        q = compile_module._target(todo.pop())
        if id(q) not in seen:
            seen.add(id(q))
            found.add(compile_module._form(q)[0])
            todo += compile_module._parts(q)
    return found


# What the compiler emits: a helper of `emit` is no form of a grammar.
HELPERS = {"call", "conversion", "function_call", "groups", "loop", "opaque", "returning"}
FORMS = {
    name.removeprefix("emit_")
    for name in vars(compile_module._Compiler)
    if name.startswith("emit_")
} - HELPERS


def test_the_random_grammars_reach_every_form_of_generated_code() -> None:
    every = st.one_of(grammars, token_grammars, text_grammars, byte_grammars)
    reached: set[str] = set()

    def has_them_all(grammar: P) -> bool:
        """Whether the grammars so far, `grammar` included, reached every form."""
        reached.update(forms_of(grammar))
        return reached >= FORMS

    find(
        every,
        has_them_all,
        settings=settings(max_examples=5000, database=None, derandomize=True, deadline=None),
    )


def test_a_grammar_of_some_and_finished_reads_text_and_bytes_in_generated_code() -> None:
    digits = (many(some(str.isdigit)) + -finished).compile()
    assert digits._fast_text is not None
    assert digits.parse_text("12") == ["1", "2"]
    assert digits.try_parse("1x") is None
    with pytest.raises(NoParseError, match="got unexpected token: 'x'"):
        digits.parse_text("1x")
    ones = (many(a(1)) + -finished).compile()
    assert ones._fast_text is not None
    assert ones.parse_bytes(b"\x01\x01") == [1, 1]
    with pytest.raises(NoParseError, match="got unexpected token: 0x02 at offset 0x1"):
        ones.parse_bytes(b"\x01\x02")


def test_such_a_grammar_still_reads_a_list_of_tokens() -> None:
    assert (many(a("x")) + -finished).compile().parse(["x", "x"]) == ["x", "x"]


def test_a_grammar_of_tok_without_a_tokenizer_generates_nothing_for_text() -> None:
    code = (tok("word") + -finished).compile()
    assert code._fast_text is None
    with pytest.raises(ValueError, match="a grammar of tokens cannot parse text"):
        code.parse_text("a")


def scanned_tokens(scan: Callable[[list[Token]], Any], tokens: list[Token]) -> object:
    """The matches a scan of tokens gives, or what it raises."""
    try:
        return list(scan(tokens))
    except Exception as e:  # noqa: BLE001 — what the scan raises is compared
        return ("raised", type(e), str(e))


@settings(max_examples=300, deadline=None)
@given(grammars, inputs)
def test_a_compiled_scan_of_tokens_finds_what_the_parser_finds(
    grammar: P, tokens: list[Token]
) -> None:
    compiled = grammar.compile()
    assert scanned_tokens(compiled.scan, tokens) == scanned_tokens(grammar.scan, tokens)
