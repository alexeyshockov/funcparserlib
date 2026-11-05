"""Byte grammars, `prefixed()`, `counted()` and `switch()`: compiled code parses as the parser."""

import struct
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import pytest
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st

from funcparserlib.binary import Binary
from funcparserlib.lexer import TokenSpec, make_tokenizer
from funcparserlib.parser import (
    CompiledParser,
    IgnoredValue,
    NoParseError,
    ParseErrors,
    Parser,
    _Undecided,
    commit,
    counted,
    finished,
    forward_decl,
    many,
    maybe,
    oneplus,
    prefixed,
    pure,
    some,
    switch,
    tok,
)
from funcparserlib.parser import longest as typed_longest
from funcparserlib.text import Text

type P = Parser[Any, Any]
type Runner = Callable[[P, bytes], Any]


def longest(*parsers: P) -> P:
    """`longest()` of parsers with values of different types, which its signature does not take."""
    return typed_longest(*parsers)


b = Binary()
plain = Text()
skipping = Text(skip=r"\s+")


def loose(value: Any) -> Any:
    """`value` as a checker sees it where a test passes what the signatures do not take."""
    return value


def interpreted(grammar: P, data: bytes) -> Any:
    return grammar.parse_bytes(data)


def compiled(grammar: P, data: bytes) -> Any:
    return grammar.compile().parse_bytes(data)


@pytest.fixture(params=[interpreted, compiled], ids=["interpreted", "compiled"])
def run(request: pytest.FixtureRequest) -> Runner:
    """Both ways of parsing bytes: the grammar itself and its generated code."""
    return request.param  # type: ignore[no-any-return]


def message(run: Runner, grammar: P, data: bytes) -> str:
    """The text of the error a parse raises, with its `ParseErrors` joined by newlines."""
    with pytest.raises((NoParseError, ParseErrors)) as caught:
        run(grammar, data)
    if isinstance(caught.value, ParseErrors):
        return "\n".join(str(e) for e in caught.value.exceptions)
    return str(caught.value)


def exact(value: object) -> object:
    """A value with the types of its parts: a `tuple` is not a `list`, what `+` builds is a
    `_Tuple`, `bytes` are not `str`.
    """
    if isinstance(value, (tuple, list)):
        return (type(value), [exact(item) for item in value])
    return (type(value), value)


def shape(value: object) -> object:
    """A value with the types of its parts, where every tuple is a `tuple`."""
    if isinstance(value, (tuple, list)):
        return (tuple if isinstance(value, tuple) else list, [shape(item) for item in value])
    return (type(value), value)


def located(token: object) -> object:
    """What an error says of its token: its value as an integer, and its offset."""
    if token is None:
        return None
    if isinstance(token, int):
        return (type(token).__name__, int(token), getattr(token, "offset", None))
    return (type(token).__name__, str(token))


def details(e: NoParseError) -> tuple[object, ...]:
    return (str(e), e.expected, e.context, located(e.token), e.state.pos, e.state.max)


def outcome(parse: Callable[[Any], Any], data: Any) -> tuple[object, ...]:
    """What a parse gave, in a form that compares values, errors and what they carry."""
    try:
        return ("value", exact(parse(data)))
    except ParseErrors as group:
        return ("errors", [details(e) for e in group.exceptions])
    except NoParseError as e:
        return ("error", *details(e))
    except (ValueError, NotImplementedError, AttributeError, TypeError) as e:
        return (type(e), str(e))


def raised(parse: Callable[[Any], Any], data: Any) -> type[BaseException] | None:
    try:
        parse(data)
    except Exception as e:  # noqa: BLE001 — which exception is the thing compared
        return type(e)
    return None


# The grammars of the property: parsers of values of several bytes, and every combinator.


def same(value: object) -> object:
    return value


def wrap(value: object) -> tuple[str, object]:
    return ("wrapped", value)


def short(value: object) -> bool:
    return len(repr(value)) % 3 != 0


def small(value: int) -> int:
    return value % 4


def recursive(body: P) -> P:
    declared: P = forward_decl()
    declared.define(b.byte + maybe(body + declared))
    return declared


leaves: st.SearchStrategy[P] = st.sampled_from(
    [
        b.byte,
        b.fixed(">H"),
        b.fixed("<H"),
        b.fixed("b"),
        b.fixed(">I"),
        b.token(rb"[\x01-\x7f]+"),
        b.token(rb"\x00+"),
        b.token(rb"[\x01\x02]{2,3}"),
        b.literal(b"\x01\x02"),
        b.literal(b"ab"),
        b.literal(b"\x00"),
        b.end,
        b.previous(),
        b.previous(0x0F),
        -b.byte,
        b.byte >> str,
        pure("p"),
    ]
)

lengths: st.SearchStrategy[P] = st.sampled_from(
    [b.byte, b.fixed("b"), b.byte >> small, b.fixed(">H") >> small, b.previous(0x03), pure(1)]
)


def combined(kids: st.SearchStrategy[P], *, risky: bool) -> st.SearchStrategy[P]:
    pairs = st.tuples(kids, kids)
    forms = [
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
        kids.map(lambda k: k + -b.end),
        lengths.map(prefixed),
        st.tuples(lengths, kids).map(lambda lk: prefixed(*lk)),
        st.tuples(lengths, kids).map(lambda lk: prefixed(*lk) + prefixed(lk[0])),
        st.tuples(lengths, kids).map(lambda lk: counted(*lk)),
        st.tuples(lengths, kids).map(lambda lk: many(prefixed(*lk))),
        st.tuples(lengths, kids, kids, kids).map(
            lambda t: switch(t[0], {0: t[1], 1: t[2], range(2, 4): t[3]})
        ),
        st.tuples(lengths, kids, kids).map(
            lambda t: switch(t[0], {range(4): t[1], 2: t[2], 1: pure("one")})
        ),
    ]
    if risky:
        forms += [
            kids.map(commit),
            kids.map(lambda k: k.check(short, "too long")),
            st.tuples(lengths, kids).map(lambda lk: commit(prefixed(*lk))),
            st.tuples(lengths, kids).map(lambda lk: commit(counted(*lk)) | pure("none")),
            st.tuples(lengths, kids, kids).map(
                lambda t: commit(switch(t[0], {0: t[1], range(1, 3): t[2]})) | pure("none")
            ),
        ]
    return st.one_of(forms)


grammars = st.recursive(leaves, lambda kids: combined(kids, risky=True), max_leaves=8)
plain_grammars = st.recursive(leaves, lambda kids: combined(kids, risky=False), max_leaves=8)
documents = st.lists(
    st.lists(st.sampled_from([0, 1, 2, 3, 4, 0x61, 0x62, 0x80, 0xFF]), max_size=9).map(bytes),
    min_size=6,
    max_size=6,
)


@settings(max_examples=150, deadline=None)
@given(grammars, documents)
def test_a_compiled_binary_grammar_parses_as_its_parser(grammar: P, samples: list[bytes]) -> None:
    code = grammar.compile()
    for data in samples:
        expected = outcome(grammar.parse_bytes, data)
        event(str(expected[0]))
        assert outcome(code.parse_bytes, data) == expected
        if expected[0] == "value":
            assert exact(code.try_parse(data)) == expected[1]
        elif expected[0] in ("error", "errors"):
            assert code.try_parse(data) is None
        else:
            assert raised(code.try_parse, data) is expected[0]
        if code._fast_text is None:
            continue
        # What the generated code decides is what the parser decides: a rejection of what the
        # parser parses costs the speed, and a value that differs is wrong.
        try:
            res = loose(code._fast_text)(data)
        except _Undecided:
            event("undecided")
            continue
        if res is None:
            assert expected[0] != "value"
        else:
            assert ("value", exact(res[0])) == expected


@settings(max_examples=100, deadline=None)
@given(plain_grammars, documents)
def test_generated_code_decides_a_grammar_without_check_or_commit(
    grammar: P, samples: list[bytes]
) -> None:
    fast = loose(grammar.compile()._fast_text)
    assume(fast is not None)
    assert fast is not None
    for data in samples:
        expected = outcome(grammar.parse_bytes, data)
        if not isinstance(expected[0], str):
            # What the parser raises is raised by the parser again.
            with pytest.raises(_Undecided):
                fast(data)
            continue
        res = fast(data)
        if expected[0] == "value":
            assert res is not None
            assert ("value", exact(res[0])) == expected
        else:
            assert res is None


# The generated code itself: a function of it that fails falls back to the same result.


def decided(grammar: P, data: bytes) -> tuple[Any, int]:
    """The value and the end `_fast_text` gives for `data`, without the interpreted parser."""
    code = grammar.compile()
    assert code._fast_text is not None, "the grammar stays interpreted"
    try:
        res = loose(code._fast_text)(data)
    except _Undecided as e:
        pytest.fail(f"generated code leaves the decision to the parser: {e.__cause__!r}")
    assert res is not None, "generated code rejects what the parser parses"
    return res


def rejected(grammar: P, data: bytes) -> None:
    code = grammar.compile()
    assert code._fast_text is not None, "the grammar stays interpreted"
    try:
        res = loose(code._fast_text)(data)
    except _Undecided as e:
        pytest.fail(f"generated code leaves the decision to the parser: {e.__cause__!r}")
    assert res is None
    with pytest.raises((NoParseError, ParseErrors)):
        grammar.parse_bytes(data)


class Tripwire:
    """Stands for the interpreted parser of a `CompiledParser`, where it must not run."""

    def parse_bytes(self, data: bytes) -> Any:
        raise AssertionError("the interpreted parser ran")

    parse_text = parse_bytes


def without_parser(code: CompiledParser[Any, Any]) -> CompiledParser[Any, Any]:
    code._parser = loose(Tripwire())
    return code


tree: P = forward_decl()
tree.define(counted(b.byte, tree))

sized: P = forward_decl()
sized.define(prefixed(b.byte, many(sized)))

expression: P = forward_decl()
expression.define(
    switch(
        b.byte,
        {0: b.byte >> str, 1: expression + expression, range(2, 4): prefixed(b.byte, expression)},
    )
)

bound = b.byte.bind(lambda n: counted(pure(n), b.byte))

FORMS: list[tuple[str, P, bytes, object, int]] = [
    ("byte", b.byte, b"\x07\x09", 7, 1),
    ("fixed", b.fixed(">H"), b"\x01\x02\x03", 0x0102, 2),
    ("fixed little", b.fixed("<H"), b"\x01\x02\x03", 0x0201, 2),
    ("fixed signed", b.fixed("b"), b"\xff", -1, 1),
    ("fixed float", b.fixed(">d"), struct.pack(">d", 1.5), 1.5, 8),
    ("token", b.token(rb"[a-z]+"), b"abc1", b"abc", 3),
    ("literal", b.literal(b"\x01\x02"), b"\x01\x02\x03", b"\x01\x02", 2),
    ("end", b.byte + b.end, b"\x01", (1, None), 1),
    ("previous", b.byte + b.previous(0x0F), b"\x7f", (0x7F, 0x0F), 1),
    (
        "seq of values",
        b.byte + b.fixed(">H") + b.token(rb"x+"),
        b"\x01\x00\x02xx",
        (1, 2, b"xx"),
        5,
    ),
    ("alt", b.literal(b"ab") | b.literal(b"cd"), b"cd", b"cd", 2),
    ("many", many(b.fixed(">H")), b"\x00\x01\x00\x02\x09", [1, 2], 4),
    ("oneplus", oneplus(b.token(rb"[xy]{2}")), b"xyyxx", [b"xy", b"yx"], 4),
    ("maybe", maybe(b.literal(b"ab")), b"cd", None, 0),
    ("map", b.fixed(">H") >> (lambda v: v + 1), b"\x00\x01", 2, 2),
    ("longest", longest(b.literal(b"ab"), b.literal(b"abc")), b"abcd", b"abc", 3),
    ("but_not", b.token(rb"[a-z]+").but_not(b.literal(b"abc")), b"abx", b"abx", 3),
    ("when", b.fixed(">H").when(lambda v: v > 5), b"\x00\x09", 9, 2),
    ("named", b.fixed(">H").named("size"), b"\x00\x09", 9, 2),
    ("context", b.fixed(">H").context("size"), b"\x00\x09", 9, 2),
    ("prefixed raw", prefixed(b.byte), b"\x03abcd", b"abc", 4),
    ("prefixed zero", prefixed(b.byte), b"\x00abcd", b"", 1),
    ("prefixed fixed length", prefixed(b.fixed(">H")), b"\x00\x02abcd", b"ab", 4),
    ("prefixed body", prefixed(b.byte, many(b.fixed(">H"))), b"\x04\x00\x01\x00\x02z", [1, 2], 5),
    ("prefixed body with end", prefixed(b.byte, b.byte + b.end), b"\x01\x05z", (5, None), 2),
    ("prefixed body zero", prefixed(b.byte, many(b.byte)), b"\x00z", [], 1),
    ("prefixed nested", prefixed(b.byte, prefixed(b.byte, b.byte)), b"\x02\x01\x09z", 9, 3),
    ("prefixed previous", prefixed(b.byte, b.byte + b.previous()), b"\x01\x05", (5, 5), 2),
    ("prefixed then more", prefixed(b.byte) + b.byte, b"\x01a\x07", (b"a", 7), 3),
    ("counted", counted(b.byte, b.fixed(">H")), b"\x02\x00\x01\x00\x02\x00\x03", [1, 2], 5),
    ("counted zero", counted(b.byte, b.fixed(">H")), b"\x00\x00\x01", [], 1),
    ("counted fixed length", counted(b.fixed(">H"), b.byte), b"\x00\x02\x05\x06\x07", [5, 6], 4),
    (
        "counted of prefixed",
        counted(b.byte, prefixed(b.byte)),
        b"\x02\x01a\x02bc",
        [b"a", b"bc"],
        6,
    ),
    ("switch value", switch(b.byte, {1: b.fixed(">H"), 2: b.token(rb"x+")}), b"\x01\x00\x07", 7, 3),
    ("switch range", switch(b.byte, {range(1, 4): b.byte}), b"\x03\x09", 9, 2),
    (
        "switch previous",
        switch(b.byte, {5: counted(b.previous(0x0F), b.byte)}),
        b"\x05\x01\x02\x03\x04\x05\x06",
        [1, 2, 3, 4, 5],
        6,
    ),
    ("tree", tree, b"\x02\x01\x00\x00", [[[]], []], 4),
    ("recursive prefixed", sized, b"\x04\x01\x00\x01\x00", [[[]], [[]]], 5),
    ("recursive switch", expression, b"\x01\x00\x07\x00\x08", ("7", "8"), 5),
    ("recursive switch region", expression, b"\x02\x05\x01\x00\x01\x00\x02", ("1", "2"), 7),
    ("bind in a sequence", b.byte + bound + b.byte, b"\x01\x01\x07\x09", (1, [7], 9), 4),
    ("bind in a region", prefixed(b.byte, bound + b.byte), b"\x03\x01\x07\x09", ([7], 9), 4),
]


@pytest.mark.parametrize(
    ("grammar", "data", "value", "end"),
    [pytest.param(g, d, v, e, id=name) for name, g, d, v, e in FORMS],
)
def test_generated_code_parses_every_form(grammar: P, data: bytes, value: object, end: int) -> None:
    parsed, stop = decided(grammar, data)
    assert shape(parsed) == shape(value)
    assert stop == end


@pytest.mark.parametrize(
    ("grammar", "data", "value", "end"),
    [pytest.param(g, d, v, e, id=name) for name, g, d, v, e in FORMS if e],
)
def test_the_value_of_generated_code_is_the_value_of_the_parser(
    grammar: P, data: bytes, value: object, end: int
) -> None:
    parsed, _ = decided(grammar, data)
    assert shape(parsed) == shape(grammar.parse_bytes(data))
    assert shape(without_parser(grammar.compile()).parse_bytes(data)) == shape(parsed)


@pytest.mark.parametrize(
    ("grammar", "data"),
    [
        pytest.param(b.byte, b"", id="byte"),
        pytest.param(b.fixed(">H"), b"\x01", id="fixed"),
        pytest.param(b.token(rb"a+"), b"b", id="token"),
        pytest.param(b.literal(b"ab"), b"a", id="literal"),
        pytest.param(b.end, b"a", id="end"),
        pytest.param(b.previous(), b"a", id="previous"),
        pytest.param(prefixed(b.byte), b"\x05ab", id="prefixed past the end"),
        pytest.param(prefixed(b.fixed("b")), b"\xffab", id="prefixed negative"),
        pytest.param(prefixed(b.byte), b"", id="prefixed without length"),
        pytest.param(prefixed(b.byte, b.byte), b"\x02ab", id="prefixed body short"),
        pytest.param(prefixed(b.byte, b.byte + b.byte), b"\x01ab", id="prefixed body long"),
        pytest.param(prefixed(b.byte, b.previous()), b"\x00", id="prefixed previous at start"),
        pytest.param(prefixed(b.byte, many(b.byte) + b.literal(b"a")), b"\x01ab", id="in a region"),
        pytest.param(counted(b.byte, b.byte), b"\x03ab", id="counted short"),
        pytest.param(counted(b.byte, b.byte), b"", id="counted without length"),
        pytest.param(switch(b.byte, {1: b.byte}), b"\x02a", id="switch unknown"),
        pytest.param(switch(b.byte, {1: b.byte}), b"\x01", id="switch case fails"),
        pytest.param(switch(b.byte, {1: b.byte}), b"", id="switch without key"),
        pytest.param(tree, b"\x02\x01", id="tree"),
        pytest.param(sized, b"\x04\x01\x00", id="recursive prefixed"),
        pytest.param(expression, b"\x01\x00\x07", id="recursive switch"),
    ],
)
def test_generated_code_rejects_without_the_parser(grammar: P, data: bytes) -> None:
    rejected(grammar, data)


def test_bind_in_a_grammar_runs_in_generated_code() -> None:
    grammar = b.byte + bound + b.byte
    data = b"\x01\x02\x05\x06\x09"
    code = grammar.compile()
    assert code._fast_text is not None
    assert loose(code._fast_text)(data) == ((1, [5, 6], 9), 5)
    assert without_parser(grammar.compile()).parse_bytes(data) == (1, [5, 6], 9)
    for short in (b"", b"\x01", b"\x01\x03\x05\x06", b"\x01\x02\x05\x06"):
        assert outcome(grammar.compile().parse_bytes, short) == outcome(grammar.parse_bytes, short)


@pytest.mark.parametrize("data", [b"\x02\x05\x06", b"\x00", b"\x03\x05", b""])
def test_a_grammar_of_only_bind_parses_as_the_parser(run: Runner, data: bytes) -> None:
    assert outcome(lambda d: run(bound, d), data) == outcome(bound.parse_bytes, data)


def test_try_parse_of_bytes() -> None:
    grammar = prefixed(b.byte, many(b.fixed(">H"))) + b.end
    code = grammar.compile()
    assert code.try_parse(b"\x04\x00\x01\x00\x02") == ([1, 2], None)
    assert code.try_parse(b"\x03\x00\x01\x00\x02") is None
    assert code.try_parse(b"") is None
    assert code.try_parse(loose(bytearray(b"\x00"))) == ([], None)


def test_try_parse_of_a_check_that_fails_gives_none_and_of_a_commit_gives_none() -> None:
    checked = prefixed(b.byte, b.byte.check(lambda v: v > 5, "too small"))
    assert checked.compile().try_parse(b"\x01\x01") is None
    assert checked.compile().try_parse(b"\x01\x09") == 9
    committed = (b.literal(b"a") + commit(counted(b.byte, b.literal(b"b")))) | pure("other")
    assert committed.compile().try_parse(b"a\x01c") is None
    assert committed.compile().try_parse(b"c") == "other"


def test_generated_code_runs_the_function_after_an_arrow_once_for_a_parse_that_succeeds() -> None:
    seen: list[int] = []

    def record(value: int) -> int:
        seen.append(value)
        return value

    grammar = prefixed(b.byte, many(b.byte >> record))
    assert grammar.compile().parse_bytes(b"\x02\x05\x06") == [5, 6]
    assert seen == [5, 6]


def test_an_exception_of_a_function_in_a_region_is_the_exception_of_the_parser() -> None:
    def explode(value: int) -> int:
        raise KeyError(value)

    grammar = prefixed(b.byte, many(b.byte >> explode))
    for parse in (grammar.parse_bytes, grammar.compile().parse_bytes):
        with pytest.raises(KeyError):
            parse(b"\x01\x05")


# prefixed()


def test_prefixed_reads_as_many_bytes_as_the_length_says(run: Runner) -> None:
    assert run(prefixed(b.byte), b"\x03abcd") == b"abc"
    assert run(prefixed(b.byte) + b.byte, b"\x03abcd") == (b"abc", ord("d"))
    assert run(prefixed(b.fixed(">H")), b"\x00\x02abc") == b"ab"
    assert run(prefixed(b.fixed("<H")), b"\x02\x00abc") == b"ab"


def test_prefixed_of_a_length_of_zero_reads_nothing(run: Runner) -> None:
    assert run(prefixed(b.byte), b"\x00") == b""
    assert run(prefixed(b.byte, many(b.byte)), b"\x00") == []
    assert run(prefixed(b.byte, b.end), b"\x00") is None
    assert run(prefixed(b.byte) + b.byte, b"\x00\x07") == (b"", 7)


def test_prefixed_gives_the_bytes_it_read_as_bytes(run: Runner) -> None:
    value = run(prefixed(b.byte), b"\x02ab")
    assert type(value) is bytes
    assert value == b"ab"


def test_prefixed_with_a_body_gives_what_the_body_parses(run: Runner) -> None:
    grammar = prefixed(b.byte, b.fixed(">H") + b.fixed(">H"))
    assert shape(run(grammar, b"\x04\x00\x01\x00\x02z")) == shape(
        run(b.fixed(">H") + b.fixed(">H"), b"\x00\x01\x00\x02")
    )
    assert run(prefixed(b.byte, many(b.byte)), b"\x03abc") == [97, 98, 99]
    assert run(prefixed(b.byte, b.token(rb"[a-z]+")), b"\x03abc") == b"abc"


def test_a_body_that_does_not_read_to_the_end_fails(run: Runner) -> None:
    assert message(run, prefixed(b.byte, b.byte), b"\x02\x00\x00") == (
        "bytes left in the region: 0x00 at offset 0x2"
    )
    assert message(run, prefixed(b.byte, b.token(rb"a")), b"\x02aa") == (
        "bytes left in the region: 0x61 at offset 0x2"
    )


def test_a_body_that_reads_nothing_of_a_region_fails(run: Runner) -> None:
    assert message(run, prefixed(b.byte, pure(1)), b"\x01a") == (
        "bytes left in the region: 0x61 at offset 0x1"
    )
    assert run(prefixed(b.byte, pure(1)), b"\x00") == 1


def test_a_body_with_end_sees_the_end_of_the_region(run: Runner) -> None:
    grammar = prefixed(b.byte, many(b.byte) + b.end) + b.byte
    assert shape(run(grammar, b"\x02\x00\x00\x07")) == shape(([0, 0], None, 7))
    assert message(run, prefixed(b.byte, b.byte + b.end), b"\x02\x00\x00\x07") == (
        "got unexpected token: 0x00 at offset 0x2, expected: end of input"
    )
    assert message(run, prefixed(b.byte, b.end), b"\x01\x00") == (
        "got unexpected token: 0x00 at offset 0x1, expected: end of input"
    )


def test_end_after_a_region_is_the_end_of_the_buffer(run: Runner) -> None:
    assert shape(run(prefixed(b.byte) + b.end, b"\x01a")) == shape((b"a", None))
    assert message(run, prefixed(b.byte) + b.end, b"\x01ab") == (
        "got unexpected token: 0x62 at offset 0x2, expected: end of input"
    )


def test_a_body_that_asks_for_more_than_the_region_has_fails_at_its_end(run: Runner) -> None:
    grammar = prefixed(b.byte, b.byte + b.byte)
    assert message(run, grammar, b"\x01\x00\x00\x00") == (
        "not enough bytes: 0x00 at offset 0x2, expected: a byte"
    )
    assert message(run, prefixed(b.byte, b.fixed(">H")), b"\x01\x00\x00\x00") == (
        "not enough bytes: 0x00 at offset 0x1, expected: 2 bytes"
    )
    assert message(run, prefixed(b.byte, b.literal(b"ab")), b"\x01aabb") == (
        "got unexpected token: 0x61 at offset 0x1, expected: b'ab'"
    )


def test_a_token_in_a_region_cannot_read_past_it(run: Runner) -> None:
    assert run(prefixed(b.byte, b.token(rb"a+")), b"\x02aaaa") == b"aa"
    assert run(prefixed(b.byte, b.token(rb"a+")) + b.token(rb"a+"), b"\x02aaaa") == (b"aa", b"aa")


def test_regions_nest(run: Runner) -> None:
    grammar = prefixed(b.byte, b.byte + prefixed(b.byte, many(b.byte)))
    assert shape(run(grammar, b"\x04\x09\x02\x01\x02z")) == shape((9, [1, 2]))
    deeper = prefixed(b.byte, prefixed(b.byte, prefixed(b.byte, many(b.byte))))
    assert run(deeper, b"\x04\x03\x02\x01\x07\x08") == [1, 7]
    assert message(run, deeper, b"\x04\x03\x05\x01\x07") == (
        "length 5 runs past the end: 0x01 at offset 0x3"
    )


def test_a_region_of_a_region_that_asks_for_more_than_its_outer_one_has_fails(run: Runner) -> None:
    grammar = prefixed(b.byte, prefixed(b.byte, many(b.byte)))
    assert message(run, grammar, b"\x02\x05ab") == (
        "length 5 runs past the end: 0x61 at offset 0x2"
    )


def test_a_negative_length_from_a_signed_fixed_fails(run: Runner) -> None:
    assert message(run, prefixed(b.fixed("b")), b"\xffab") == (
        "length -1 runs past the end: 0x61 at offset 0x1"
    )
    assert message(run, prefixed(b.fixed(">h")), b"\xff\xfeab") == (
        "length -2 runs past the end: 0x61 at offset 0x2"
    )
    assert message(run, prefixed(b.fixed("b"), many(b.byte)), b"\x80ab") == (
        "length -128 runs past the end: 0x61 at offset 0x1"
    )


def test_a_length_past_the_end_fails_where_its_bytes_start(run: Runner) -> None:
    assert message(run, prefixed(b.byte), b"\x03ab") == (
        "length 3 runs past the end: 0x61 at offset 0x1"
    )
    assert message(run, b.byte + prefixed(b.byte, many(b.byte)), b"\x00\x09ab") == (
        "length 9 runs past the end: 0x61 at offset 0x2"
    )
    assert run(prefixed(b.byte), b"\x02ab") == b"ab"


def test_a_length_of_a_buffer_that_ends_before_it_fails(run: Runner) -> None:
    assert message(run, prefixed(b.byte), b"") == (
        "got unexpected end of input at offset 0x0, expected: a byte"
    )
    assert message(run, prefixed(b.fixed(">H")), b"\x00") == (
        "not enough bytes: 0x00 at offset 0x0, expected: 2 bytes"
    )


def test_a_check_in_a_region_reports_the_offset_in_the_buffer(run: Runner) -> None:
    big = b.byte.check(lambda v: v > 5, "too small")
    grammar = b.byte + b.byte + prefixed(b.byte, big + big)
    assert message(run, grammar, b"\x01\x02\x02\x01\x09") == "offset 0x3: too small"
    assert message(run, grammar, b"\x01\x02\x02\x09\x01") == "offset 0x4: too small"
    assert message(run, grammar, b"\x01\x02\x02\x01\x01") == (
        "offset 0x3: too small\noffset 0x4: too small"
    )


def test_a_check_in_a_nested_region_reports_the_offset_in_the_buffer(run: Runner) -> None:
    big = b.byte.check(lambda v: v > 5, "too small")
    grammar = b.byte + prefixed(b.byte, b.byte + b.byte + prefixed(b.byte, big))
    assert message(run, grammar, b"\x00\x04\x00\x00\x01\x01") == "offset 0x5: too small"
    assert run(grammar, b"\xff\x04\x00\x00\x01\x09") == (255, (0, 0, 9))


def test_a_check_in_a_region_that_fails_after_it_keeps_both_errors(run: Runner) -> None:
    big = b.byte.check(lambda v: v > 5, "too small")
    grammar = prefixed(b.byte, big + b.literal(b"x"))
    assert message(run, grammar, b"\x02\x01y") == (
        "offset 0x1: too small\ngot unexpected token: 0x79 at offset 0x2, expected: b'x'"
    )


def test_a_check_failure_reports_the_offsets_of_the_region_it_ended_in(run: Runner) -> None:
    grammar = many(prefixed(b.byte, b.byte.check(lambda v: v > 5, "too small")))
    with pytest.raises(ParseErrors) as caught:
        run(grammar, b"\x01\x01\x01\x02\x01\x07")
    assert [str(e) for e in caught.value.exceptions] == [
        "offset 0x1: too small",
        "offset 0x3: too small",
    ]
    assert [e.state.pos for e in caught.value.exceptions] == [1, 3]


def test_an_error_in_a_region_reports_the_offset_in_the_buffer(run: Runner) -> None:
    grammar = b.byte + prefixed(b.byte, b.byte + b.literal(b"x"))
    assert message(run, grammar, b"\x00\x02\x00\x00") == (
        "got unexpected token: 0x00 at offset 0x3, expected: b'x'"
    )
    nested = b.byte + prefixed(b.byte, b.byte + prefixed(b.byte, b.byte + b.literal(b"x")))
    assert message(run, nested, b"\x00\x05\x00\x02\x01\x01\x01") == (
        "got unexpected token: 0x01 at offset 0x5, expected: b'x'"
    )


def test_the_error_of_a_region_carries_the_byte_and_the_offset_in_the_buffer(run: Runner) -> None:
    grammar = b.byte + prefixed(b.byte, b.byte + b.literal(b"x"))
    with pytest.raises(NoParseError) as caught:
        run(grammar, b"\x00\x02\x00\x7a")
    assert int(loose(caught.value.token)) == 0x7A
    assert loose(caught.value.token).offset == 3
    assert caught.value.state.pos == 3
    assert caught.value.state.max == 3
    assert caught.value.expected == ("b'x'",)


def test_the_context_of_a_region_reaches_its_errors(run: Runner) -> None:
    grammar = prefixed(b.byte, b.byte + b.literal(b"x")).context("record")
    assert message(run, grammar, b"\x02\x00\x00") == (
        "got unexpected token: 0x00 at offset 0x2, expected: b'x' in record"
    )
    inner = prefixed(b.byte, (b.byte + b.literal(b"x")).context("entry")).context("record")
    assert message(run, inner, b"\x02\x00\x00") == (
        "got unexpected token: 0x00 at offset 0x2, expected: b'x' in entry in record"
    )


def test_a_failure_after_a_region_reports_the_furthest_offset(run: Runner) -> None:
    grammar = prefixed(b.byte, b.byte) + b.literal(b"x")
    assert message(run, grammar, b"\x01a\x7a") == (
        "got unexpected token: 0x7a at offset 0x2, expected: b'x'"
    )
    alt = (prefixed(b.byte, b.byte + b.byte) | prefixed(b.byte, b.byte)) + b.literal(b"x")
    assert (
        message(run, alt, b"\x01a\x7a")
        == "got unexpected token: 0x7a at offset 0x2, expected: a byte or b'x'"
    )


def test_a_failure_in_a_region_does_not_hide_a_failure_further_on(run: Runner) -> None:
    far = b.byte + b.byte + b.literal(b"z")
    grammar = far | prefixed(b.byte, b.literal(b"q"))
    assert message(run, grammar, b"\x01ab") == (
        "got unexpected token: 0x62 at offset 0x2, expected: b'z'"
    )


def test_previous_at_the_start_of_a_region_fails(run: Runner) -> None:
    grammar = prefixed(b.byte, b.previous() + b.byte)
    assert message(run, grammar, b"\x01\x05") == (
        "got unexpected token: 0x05 at offset 0x1, expected: the byte before"
    )
    assert message(run, prefixed(b.byte, b.previous()), b"\x00") == (
        "got unexpected end of input at offset 0x1, expected: the byte before"
    )


def test_previous_inside_a_region_sees_the_byte_before_in_the_region(run: Runner) -> None:
    assert shape(run(prefixed(b.byte, b.byte + b.previous()), b"\x01\x05")) == shape((5, 5))
    assert shape(
        run(prefixed(b.byte, b.byte + b.byte + b.previous(0x0F)), b"\x02\x01\x7f")
    ) == shape((1, 0x7F, 0x0F))


def test_previous_after_a_region_is_the_last_byte_of_it(run: Runner) -> None:
    assert shape(run(prefixed(b.byte) + b.previous(), b"\x02ab")) == shape((b"ab", ord("b")))
    assert shape(run(prefixed(b.byte, many(b.byte)) + b.previous(), b"\x02ab")) == shape(
        ([97, 98], 98)
    )


def test_previous_after_an_empty_region_is_the_byte_of_the_length(run: Runner) -> None:
    assert shape(run(prefixed(b.byte) + b.previous(), b"\x00")) == shape((b"", 0))
    assert shape(run(prefixed(b.byte, many(b.byte)) + b.previous(), b"\x00")) == shape(([], 0))


def test_many_regions_of_a_length_of_zero(run: Runner) -> None:
    assert shape(run(many(prefixed(b.byte)), b"\x00\x00\x00")) == shape([b"", b"", b""])
    assert shape(run(many(prefixed(b.byte, many(b.byte))), b"\x00\x01a\x00")) == shape(
        [[], [97], []]
    )
    assert shape(run(many(prefixed(b.byte, b.end)) + b.end, b"\x00\x00")) == shape(
        ([None, None], None)
    )


def test_many_regions_of_a_length_that_reads_nothing(run: Runner) -> None:
    with pytest.raises(ValueError, match="matched no tokens inside many"):
        run(many(prefixed(pure(0))), b"")


def test_a_region_sequence_stops_at_the_first_that_does_not_fit(run: Runner) -> None:
    grammar = many(prefixed(b.byte)) + b.byte
    assert shape(run(grammar, b"\x01a\x01b\x09c")) == shape(([b"a", b"b"], 9))


def test_an_alternative_after_a_failed_region_starts_where_it_began(run: Runner) -> None:
    grammar = prefixed(b.byte, b.literal(b"xx")) | b.token(rb"\x02a+")
    assert run(grammar, b"\x02aa") == b"\x02aa"
    assert run(grammar, b"\x02xx") == b"xx"


def test_a_committed_region_that_fails_is_final(run: Runner) -> None:
    grammar = commit(prefixed(b.byte, b.byte + b.literal(b"x"))) | pure("other")
    assert (
        message(run, grammar, b"\x01a")
        == "got unexpected end of input at offset 0x2, expected: b'x'"
    )
    assert message(run, grammar, b"\x02\x00y") == (
        "got unexpected token: 0x79 at offset 0x2, expected: b'x'"
    )
    assert run(grammar, b"\x02ax") == (97, b"x")
    # The length was read, so a length past the end is final too.
    assert message(run, grammar, b"\x05") == ("got unexpected end of input at offset 0x1")


def test_a_region_recovers_with_the_rest_of_the_input(run: Runner) -> None:
    record = prefixed(b.byte, b.byte + b.literal(b"x")).recover(b.literal(b"\n"))
    grammar = many(record + -b.literal(b"\n")) + -b.end
    with pytest.raises(ParseErrors) as caught:
        run(grammar, b"\x02ax\n\x02ay\n\x02bx\n\x02cz\n")
    errors = caught.value.exceptions
    assert [e.state.pos for e in errors] == [6, 14]
    assert [str(e).split(":")[0] for e in errors] == ["offset 0x6", "offset 0xe"]
    assert all(str(e).endswith("expected: b'x'") for e in errors)


# prefixed(), counted() and switch() over Text and tokens


number = plain.token(r"\d+") >> int


def test_prefixed_over_text_reads_as_many_characters() -> None:
    assert plain_run(prefixed(number), "3abcd") == "abc"
    assert plain_run(prefixed(number, plain.token(r"\w+")), "3abcd") == "abc"
    assert plain_run(prefixed(number), "0abcd") == ""
    assert plain_run(prefixed(number, many(plain.token("[a-z]"))), "3abcd") == ["a", "b", "c"]


def plain_run(grammar: P, text: str) -> Any:
    """What both ways of parsing text give: they must agree."""
    first = grammar.parse_text(text)
    second = grammar.compile().parse_text(text)
    assert shape(first) == shape(second)
    return first


def text_message(grammar: P, text: str) -> str:
    messages = []
    for parse in (grammar.parse_text, grammar.compile().parse_text):
        with pytest.raises((NoParseError, ParseErrors)) as caught:
            parse(text)
        if isinstance(caught.value, ParseErrors):
            messages.append("\n".join(str(e) for e in caught.value.exceptions))
        else:
            messages.append(str(caught.value))
    assert messages[0] == messages[1]
    return messages[0]


def test_prefixed_over_text_with_a_length_past_the_end_fails() -> None:
    assert text_message(prefixed(number), "9abcd") == (
        "1,2-1,2: length 9 runs past the end: 'abcd'"
    )


def fast_text(grammar: P, text: str, tokenizer: Any = None) -> tuple[Any, int]:
    """The value and the end `_fast_text` gives for `text`, which the interpreted parser has no part
    in.
    """
    code = grammar.compile(tokenizer)
    assert code._fast_text is not None, "the grammar stays interpreted"
    try:
        res = code._fast_text(text)
    except _Undecided as e:
        pytest.fail(f"generated code leaves the decision to the parser: {e.__cause__!r}")
    assert res is not None, "generated code rejects what the parser parses"
    return res


def test_generated_code_parses_prefixed_counted_and_switch_over_text() -> None:
    letters = plain.token("[a-z]")
    assert fast_text(prefixed(number), "3abcd") == ("abc", 4)
    assert fast_text(prefixed(number, plain.token(r"\w+")), "3abcd") == ("abc", 4)
    assert fast_text(prefixed(number, many(letters)), "3abcd") == (["a", "b", "c"], 4)
    assert fast_text(counted(number, letters), "3abcd") == (["a", "b", "c"], 4)
    assert fast_text(switch(letters, {"a": plain.token("1"), "b": plain.token("2")}), "b2") == (
        "2",
        2,
    )
    assert fast_text(switch(number, {range(1, 3): letters}), "2a") == ("a", 2)


def test_generated_code_rejects_prefixed_counted_and_switch_over_text() -> None:
    letters = plain.token("[a-z]")
    for grammar, text in (
        (prefixed(number), "9abc"),
        (prefixed(number, letters), "3abcd"),
        (counted(number, letters), "3ab1"),
        (switch(letters, {"a": letters}), "ca"),
    ):
        code = grammar.compile()
        assert code._fast_text is not None
        assert code._fast_text(text) is None


def test_generated_code_parses_counted_and_switch_over_tokens_given_a_tokenizer() -> None:
    assert fast_text(counted(count, tok("w")), "2 a b c", tokenize) == (["a", "b"], 3)
    assert fast_text(switch(count, {1: tok("w"), 2: tok("w") + tok("w")}), "2 a b", tokenize) == (
        ("a", "b"),
        3,
    )


def test_a_prefixed_over_tokens_stays_with_the_tokens_of_a_tokenizer() -> None:
    code = prefixed(count, many(tok("w"))).compile(tokenize)
    assert code.parse_text("2 a b c") == ["a", "b"]
    assert code.parse(list(tokenize("2 a b c"))) == ["a", "b"]


def test_a_body_over_text_sees_the_region_as_its_whole_input() -> None:
    message = text_message(prefixed(number, plain.token("[a-z]")), "3abcd")
    assert message.startswith("1,3-1,3: bytes left in the region: ")
    assert plain_run(prefixed(number, plain.token("[a-z]+") + plain.end), "3abcd") == ("abc", None)


def test_text_in_a_region_does_not_match_past_it() -> None:
    assert plain_run(prefixed(number, plain.token(r"a+")), "2aaaa") == "aa"
    assert plain_run(prefixed(number, plain.token(r"a+")) + plain.token("a+"), "2aaaa") == (
        "aa",
        "aa",
    )


def test_a_text_with_skip_in_a_region() -> None:
    grammar = prefixed(skipping.token(r"\d+") >> int, many(skipping.token("[a-z]")))
    assert plain_run(grammar, "5 a b c d") == ["a", "b", "c"]


def test_counted_over_text() -> None:
    assert plain_run(counted(number, plain.token("[a-z]")), "3abcd") == ["a", "b", "c"]
    assert plain_run(counted(number, plain.token("[a-z]")), "0abcd") == []
    assert text_message(counted(number, plain.token("[a-z]")), "3ab1") == (
        "1,4-1,4: got unexpected token: '1', expected: /[a-z]/"
    )


def test_switch_over_text_with_keys_of_strings() -> None:
    grammar = switch(plain.token("[a-z]"), {"a": plain.token("1"), "b": plain.token("2")})
    assert plain_run(grammar, "a1") == "1"
    assert plain_run(grammar, "b2") == "2"
    assert text_message(grammar, "c2") == (
        "1,1-1,1: got unexpected token: 'c2', expected: one of the cases of /[a-z]/"
    )
    assert text_message(grammar, "a2") == "1,2-1,2: got unexpected token: '2', expected: /1/"


def test_switch_over_text_with_a_range_of_values_read_as_numbers() -> None:
    grammar = switch(number, {range(1, 3): plain.token("a"), 5: plain.token("b")})
    assert plain_run(grammar, "2a") == "a"
    assert plain_run(grammar, "5b") == "b"
    assert text_message(grammar, "3a").startswith("1,1-1,1: got unexpected token")


class Letter(StrEnum):
    A = "a"
    B = "b"


def test_switch_with_a_str_enum_key_is_found_by_its_value() -> None:
    grammar = switch(plain.token("[a-z]"), {Letter.A: plain.token("1"), Letter.B: plain.token("2")})
    assert plain_run(grammar, "a1") == "1"
    assert plain_run(grammar, "b2") == "2"


def test_switch_with_a_str_enum_value_gives_the_enum_member_a_str_key_matches() -> None:
    members = switch(
        plain.token("[a-z]") >> Letter, loose({"a": plain.token("1"), Letter.B: plain.token("2")})
    )
    assert plain_run(members, "a1") == "1"
    assert plain_run(members, "b2") == "2"


def test_switch_of_bytes_with_str_enum_keys(run: Runner) -> None:
    grammar = switch(b.byte >> (lambda v: Letter(chr(v))), loose({Letter.A: pure(1), "b": pure(2)}))
    assert run(grammar, b"a") == 1
    assert run(grammar, b"b") == 2


tokenize = make_tokenizer(
    [TokenSpec("n", r"\d+"), TokenSpec("w", r"[a-z]+"), TokenSpec("sp", r"\s+")], skip={"sp"}
)
count = tok("n") >> int


def token_outcomes(grammar: P, text: str) -> list[tuple[object, ...]]:
    """What parsing the tokens of `text` gives: the parser, the compiled parser, and the compiled
    parser given a tokenizer, as text and as tokens.
    """
    tokens = list(tokenize(text))
    with_tokenizer = grammar.compile(tokenize)
    without = grammar.compile()

    def parse(parser: Any, how: str) -> Callable[[str], Any]:
        if how == "tokens":
            return lambda _: parser.parse(tokens)
        return parser.parse_text

    return [
        outcome(parse(grammar, "tokens"), text),
        outcome(parse(without, "tokens"), text),
        outcome(parse(with_tokenizer, "tokens"), text),
        outcome(parse(with_tokenizer, "text"), text),
    ]


def agree(grammar: P, text: str) -> tuple[object, ...]:
    """The outcome of `text`, which every way of parsing it gives."""
    outcomes = token_outcomes(grammar, text)
    for other in outcomes[1:]:
        assert other == outcomes[0]
    return outcomes[0]


def test_prefixed_over_tokens_reads_as_many_tokens() -> None:
    grammar = prefixed(loose(count))
    tokens = list(tokenize("2 a b c"))
    assert agree(grammar, "2 a b c")[0] == "value"
    assert grammar.parse(tokens) == tokens[1:3]
    assert grammar.compile().parse(tokens) == tokens[1:3]
    assert grammar.compile(tokenize).parse_text("2 a b c") == tokens[1:3]
    assert type(grammar.parse(tokens)) is list


def test_prefixed_with_a_body_over_tokens() -> None:
    grammar = prefixed(count, many(tok("w")))
    assert grammar.parse(list(tokenize("2 a b c"))) == ["a", "b"]
    assert grammar.compile().parse(list(tokenize("2 a b c"))) == ["a", "b"]
    assert grammar.compile(tokenize).parse_text("2 a b c") == ["a", "b"]
    assert grammar.compile(tokenize).parse(list(tokenize("2 a b c"))) == ["a", "b"]
    for text in ("2 a", "3 a b", "2 a 1", "0", "1 1", "2 a b"):
        agree(grammar, text)


def test_a_body_over_tokens_that_does_not_read_to_the_end_fails() -> None:
    grammar = prefixed(count, tok("w"))
    first = agree(grammar, "2 a b c")
    assert first[:2] == ("error", "1,5-1,5: bytes left in the region: 'b'")


def test_a_body_over_tokens_with_finished_sees_the_end_of_the_region() -> None:
    grammar = prefixed(count, many(tok("w")) + finished) + tok("w")
    assert grammar.parse(list(tokenize("2 a b c"))) == (["a", "b"], None, "c")
    assert grammar.compile().parse(list(tokenize("2 a b c"))) == (["a", "b"], None, "c")
    assert grammar.compile(tokenize).parse_text("2 a b c") == (["a", "b"], None, "c")


def test_prefixed_over_tokens_with_a_length_past_the_end_fails() -> None:
    grammar = prefixed(loose(count))
    for text in ("5 a b", "5", "3 a"):
        first = agree(grammar, text)
        assert first[0] == "error"
    assert agree(grammar, "5 a b")[1] == "1,3-1,3: length 5 runs past the end: 'a'"


def test_counted_over_tokens() -> None:
    grammar = counted(count, tok("w"))
    assert grammar.parse(list(tokenize("2 a b c"))) == ["a", "b"]
    assert grammar.compile().parse(list(tokenize("2 a b c"))) == ["a", "b"]
    assert grammar.compile(tokenize).parse_text("2 a b c") == ["a", "b"]
    assert grammar.compile(tokenize).parse_text("0 a") == []
    for text in ("2 a", "3 a b", "2 a 1", "0", "2 a b"):
        agree(grammar, text)
    assert agree(grammar, "2 a 1")[1] == "1,5-1,5: got unexpected token: '1', expected: w"


def test_switch_over_tokens() -> None:
    grammar = switch(count, {1: tok("w"), range(2, 4): tok("w") + tok("w"), 9: tok("n")})
    assert grammar.parse(list(tokenize("1 a"))) == "a"
    for text in ("1 a", "2 a b", "3 a b", "9 5", "4 a", "1 1", "2 a", "9"):
        agree(grammar, text)
    assert (
        agree(grammar, "4 a")[1]
        == "1,1-1,1: got unexpected token: '4', expected: one of the cases of n"
    )
    assert grammar.compile(tokenize).parse_text("3 a b") == ("a", "b")


def test_switch_over_tokens_with_a_key_of_values() -> None:
    grammar = switch(tok("w"), {"a": tok("n"), "b": tok("w")}) + -finished
    assert grammar.parse(list(tokenize("a 1"))) == "1"
    assert grammar.compile(tokenize).parse_text("b c") == "c"
    for text in ("a 1", "b c", "c c", "a c"):
        agree(grammar, text)


def test_the_tokens_of_a_region_are_not_tokens_of_a_tokenizer_that_does_not_join() -> None:
    separate = make_tokenizer([TokenSpec("q", r"(?P<q>a)(?P=q)"), TokenSpec("n", r"\d")])
    grammar = prefixed(tok("n") >> int, many(tok("q"))) + -finished
    assert grammar.parse(list(separate("1aa"))) == ["aa"]
    assert grammar.compile(separate).parse_text("1aa") == ["aa"]


def test_a_compiled_grammar_of_tokens_without_a_tokenizer_cannot_parse_bytes() -> None:
    grammar = prefixed(count, many(tok("w")))
    with pytest.raises(ValueError, match="without a tokenizer"):
        grammar.compile().parse_bytes(b"\x01a")
    with pytest.raises(ValueError, match="without a tokenizer"):
        grammar.compile().parse_text("1 a")


def test_prefixed_over_any_tokens_that_are_not_text_or_bytes() -> None:
    letters = some(str.isalpha)
    digit = some(str.isdigit) >> int
    grammar = prefixed(digit, many(letters))
    assert grammar.parse("2ab") == ["a", "b"]
    assert grammar.compile().parse("2abc") == ["a", "b"]
    assert counted(digit, letters).parse("2abc") == ["a", "b"]
    assert counted(digit, letters).compile().parse("2abc") == ["a", "b"]
    assert switch(digit, {1: letters, 2: letters + letters}).compile().parse("2ab") == ("a", "b")
    assert prefixed(digit).parse("2abc") == "ab"
    assert prefixed(digit).compile().parse("2abc") == "ab"


def test_prefixed_over_a_list_of_integers_is_a_list() -> None:
    byte = some(lambda v: isinstance(v, int))
    grammar = prefixed(byte)
    assert grammar.parse([2, 7, 8, 9]) == [7, 8]
    assert grammar.compile().parse([2, 7, 8, 9]) == [7, 8]
    assert grammar.parse(b"\x02\x07\x08\x09") == b"\x07\x08"


# counted()


def test_counted_reads_as_many_items_as_the_length_says(run: Runner) -> None:
    assert run(counted(b.byte, b.fixed(">H")), b"\x02\x00\x01\x00\x02\x00\x03") == [1, 2]
    assert run(counted(b.fixed(">H"), b.byte), b"\x00\x03abcd") == [97, 98, 99]
    assert run(counted(b.byte, b.token(rb"[a-z]{2}")), b"\x02abcd") == [b"ab", b"cd"]


def test_counted_zero_reads_no_item(run: Runner) -> None:
    value = run(counted(b.byte, b.byte), b"\x00z")
    assert value == []
    assert type(value) is list
    assert run(counted(b.byte, b.byte) + b.byte, b"\x00z") == ([], ord("z"))
    assert run(counted(b.byte, b.byte) + b.end, b"\x00") == ([], None)


def test_counted_fails_where_an_item_fails_in_the_middle(run: Runner) -> None:
    assert message(run, counted(b.byte, b.fixed(">H")), b"\x03\x00\x01\x00\x02\x00") == (
        "not enough bytes: 0x00 at offset 0x5, expected: 2 bytes"
    )
    assert message(run, counted(b.byte, b.literal(b"ab")), b"\x03ababcb") == (
        "got unexpected token: 0x63 at offset 0x5, expected: b'ab'"
    )
    assert message(run, counted(b.byte, b.byte), b"\x03ab") == (
        "got unexpected end of input at offset 0x3, expected: a byte"
    )


def test_counted_fails_where_the_length_fails(run: Runner) -> None:
    assert message(run, counted(b.byte, b.byte), b"") == (
        "got unexpected end of input at offset 0x0, expected: a byte"
    )
    assert message(run, counted(b.fixed(">H"), b.byte), b"\x01") == (
        "not enough bytes: 0x01 at offset 0x0, expected: 2 bytes"
    )


def test_counted_of_a_huge_count_fails_where_the_input_ends(run: Runner) -> None:
    assert message(run, counted(b.fixed(">I"), b.byte), b"\xff\xff\xff\xffab") == (
        "got unexpected end of input at offset 0x6, expected: a byte"
    )


@pytest.mark.parametrize("item", [b.end, pure("x"), b.previous()])
def test_counted_of_an_item_that_reads_nothing_raises(run: Runner, item: P) -> None:
    with pytest.raises(ValueError, match="inside counted"):
        run(counted(b.byte, item), b"\x03")


def test_counted_recurses_through_forward_decl(run: Runner) -> None:
    assert run(tree, b"\x02\x01\x00\x00") == [[[]], []]
    assert run(tree, b"\x00") == []
    assert message(run, tree, b"\x02\x01\x00") == (
        "got unexpected end of input at offset 0x3, expected: a byte"
    )


def test_counted_of_regions(run: Runner) -> None:
    grammar = counted(b.byte, prefixed(b.byte, many(b.byte)))
    assert run(grammar, b"\x02\x01a\x02bc") == [[97], [98, 99]]
    assert message(run, grammar, b"\x02\x01a\x05bc").startswith(
        "length 5 runs past the end: 0x62 at offset 0x4"
    )


def test_counted_keeps_the_values_in_a_list_of_the_item_type(run: Runner) -> None:
    value = run(counted(b.byte, b.byte >> str), b"\x02\x01\x02")
    assert shape(value) == shape(["1", "2"])


# switch()


def test_switch_chooses_by_a_value(run: Runner) -> None:
    grammar = switch(b.byte, {1: b.fixed(">H"), 2: b.token(rb"x+"), 3: pure("three")})
    assert run(grammar, b"\x01\x00\x07") == 7
    assert run(grammar, b"\x02xxy") == b"xx"
    assert run(grammar, b"\x03") == "three"


def test_switch_chooses_by_a_range(run: Runner) -> None:
    grammar = switch(b.byte, {range(1, 4): b.byte, range(10, 20, 5): pure("ten")})
    for key in (1, 2, 3):
        assert run(grammar, bytes([key, 9])) == 9
    assert run(grammar, b"\x0a") == "ten"
    assert run(grammar, b"\x0f") == "ten"
    for key in (0, 4, 9, 11, 12, 20):
        assert "one of the cases" in message(run, grammar, bytes([key, 9]))


def test_switch_of_a_range_with_a_negative_start(run: Runner) -> None:
    grammar = switch(b.fixed("b"), {range(-3, 0): pure("negative"), 0: pure("zero")})
    assert run(grammar, b"\xfe") == "negative"
    assert run(grammar, b"\x00") == "zero"
    assert "one of the cases" in message(run, grammar, b"\x01")


def test_switch_of_an_empty_range_has_no_case(run: Runner) -> None:
    grammar = switch(b.byte, {range(3, 3): pure("never"), 1: pure("one")})
    assert "one of the cases" in message(run, grammar, b"\x03")
    assert run(grammar, b"\x01") == "one"


def test_switch_without_a_case_fails_for_any_key(run: Runner) -> None:
    assert "one of the cases" in message(run, switch(b.byte, {}), b"\x05")


def test_overlapping_ranges_resolve_to_the_last_case_that_has_the_key(run: Runner) -> None:
    grammar = switch(b.byte, {range(5): pure("a"), 2: pure("b"), range(1, 3): pure("c")})
    assert run(grammar, b"\x00") == "a"
    assert run(grammar, b"\x01") == "c"
    assert run(grammar, b"\x02") == "c"
    assert run(grammar, b"\x03") == "a"
    flipped = switch(b.byte, {2: pure("b"), range(5): pure("a")})
    assert run(flipped, b"\x02") == "a"


def test_switch_with_an_unknown_value_fails_at_the_start_of_the_key(run: Runner) -> None:
    grammar = b.byte + switch(b.byte, {1: b.byte})
    assert message(run, grammar, b"\x00\x07a") == (
        "got unexpected token: 0x07 at offset 0x1, expected: one of the cases of a byte"
    )
    with pytest.raises(NoParseError) as caught:
        run(grammar, b"\x00\x07a")
    assert caught.value.state.pos == 1
    assert int(loose(caught.value.token)) == 7
    assert loose(caught.value.token).offset == 1


def test_switch_does_not_try_another_case_when_one_fails(run: Runner) -> None:
    grammar = switch(b.byte, {1: b.literal(b"a"), 2: b.literal(b"b")}) | b.literal(b"\x01b")
    assert run(grammar, b"\x01b") == b"\x01b"
    assert message(run, switch(b.byte, {1: b.literal(b"a")}), b"\x01b") == (
        "got unexpected token: 0x62 at offset 0x1, expected: b'a'"
    )


def test_switch_with_a_name_reports_it(run: Runner) -> None:
    grammar = switch(b.byte, {1: b.byte}).named("item")
    assert (
        message(run, grammar, b"\x07") == "got unexpected token: 0x07 at offset 0x0, expected: item"
    )
    assert message(run, grammar, b"") == "got unexpected end of input at offset 0x0, expected: item"


def test_switch_without_a_name_says_what_it_chooses_by(run: Runner) -> None:
    grammar = switch(b.fixed(">H"), {1: b.byte})
    assert message(run, grammar, b"\x00\x02") == (
        "got unexpected token: 0x00 at offset 0x0, expected: one of the cases of 2 bytes"
    )


def test_switch_labels_its_cases(run: Runner) -> None:
    grammar = switch(b.byte, {1: b.byte.named("small"), 2: b.fixed(">H").named("wide")})
    assert message(run, grammar, b"\x01") == (
        "got unexpected end of input at offset 0x1, expected: small"
    )
    assert message(run, grammar, b"\x02\x00") == (
        "not enough bytes: 0x00 at offset 0x1, expected: wide"
    )


def test_a_switch_with_a_context_puts_it_on_the_errors_of_its_cases(run: Runner) -> None:
    grammar = switch(b.byte, {1: b.byte + b.literal(b"x")}).context("record")
    assert message(run, grammar, b"\x01\x00\x00") == (
        "got unexpected token: 0x00 at offset 0x2, expected: b'x' in record"
    )
    assert message(run, grammar, b"\x09") == (
        "got unexpected token: 0x09 at offset 0x0, expected: one of the cases of a byte in record"
    )


def test_switch_with_an_unhashable_key_value_raises_what_a_lookup_raises() -> None:
    grammar = switch(loose(b.byte >> (lambda v: [v])), {1: b.byte})
    for parse in (grammar.parse_bytes, grammar.compile().parse_bytes):
        with pytest.raises(TypeError, match="unhashable"):
            parse(b"\x01\x02")


def test_switch_with_an_unhashable_case_key_raises_when_it_is_built() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        switch(b.byte, loose({(1, [2]): b.byte}))


def test_switch_matches_keys_that_are_equal(run: Runner) -> None:
    assert run(switch(b.byte, {True: pure("yes")}), b"\x01") == "yes"
    assert run(switch(b.fixed(">f"), {1.5: pure("float")}), b"?\xc0\x00\x00") == "float"
    assert run(switch(b.token(rb"[ab]"), {b"a": pure("A")}), b"a") == "A"


def test_switch_does_not_change_with_the_mapping_after_it_is_built(run: Runner) -> None:
    cases: dict[Any, P] = {1: pure("one")}
    grammar = switch(b.byte, cases)
    cases[2] = pure("two")
    assert run(grammar, b"\x01") == "one"
    assert "one of the cases" in message(run, grammar, b"\x02")


def test_switch_of_a_case_that_is_a_forward_declaration(run: Runner) -> None:
    case: P = forward_decl()
    grammar = switch(b.byte, {1: case})
    case.define(b.fixed(">H"))
    assert run(grammar, b"\x01\x00\x05") == 5


def test_switch_recurses_through_forward_decl(run: Runner) -> None:
    assert run(expression, b"\x00\x07") == "7"
    assert shape(run(expression, b"\x01\x00\x07\x00\x08")) == shape(("7", "8"))
    assert run(expression, b"\x02\x02\x00\x07") == "7"


def test_a_tag_that_holds_a_value_is_read_with_previous(run: Runner) -> None:
    grammar = switch(
        b.byte,
        {range(0x90, 0xA0): counted(b.previous(0x0F), b.byte), range(0xA0, 0xB0): pure("other")},
    )
    assert run(grammar, b"\x92\x01\x02\x03") == [1, 2]
    assert run(grammar, b"\x90\x01") == []
    assert run(grammar, b"\xa5") == "other"


# Binary


@pytest.mark.parametrize(
    ("fmt", "data", "value"),
    [
        ("B", b"\xff", 255),
        ("b", b"\xff", -1),
        ("?", b"\x01", True),
        ("c", b"a", b"a"),
        ("h", b"\xff\xfe", struct.unpack("=h", b"\xff\xfe")[0]),
        ("<h", b"\xff\xfe", -257),
        (">h", b"\xff\xfe", -2),
        ("<H", b"\x01\x02", 0x0201),
        (">H", b"\x01\x02", 0x0102),
        ("<i", b"\x01\x02\x03\x04", 0x04030201),
        (">i", b"\x01\x02\x03\x04", 0x01020304),
        ("<I", b"\xff\xff\xff\xff", 0xFFFFFFFF),
        (">I", b"\xff\xff\xff\xff", 0xFFFFFFFF),
        ("<q", b"\x01\x02\x03\x04\x05\x06\x07\x08", 0x0807060504030201),
        (">q", b"\x01\x02\x03\x04\x05\x06\x07\x08", 0x0102030405060708),
        ("<Q", b"\xff" * 8, 2**64 - 1),
        (">Q", b"\xff" * 8, 2**64 - 1),
        ("<e", struct.pack("<e", 1.5), 1.5),
        (">e", struct.pack(">e", 1.5), 1.5),
        ("<f", struct.pack("<f", 1.5), 1.5),
        (">f", struct.pack(">f", 1.5), 1.5),
        ("<d", struct.pack("<d", 1.5), 1.5),
        (">d", struct.pack(">d", 1.5), 1.5),
        ("3s", b"abc", b"abc"),
        ("s", b"a", b"a"),
        ("!H", b"\x01\x02", 0x0102),
        ("=H", struct.pack("=H", 258), 258),
        ("@H", struct.pack("@H", 258), 258),
    ],
)
def test_fixed_reads_every_struct_format_of_one_value(
    run: Runner, fmt: str, data: bytes, value: object
) -> None:
    parsed = run(b.fixed(fmt), data + b"\x00")
    assert shape(parsed) == shape(value)
    assert shape(run(b.fixed(fmt) + b.byte, data + b"\x07")) == shape((value, 7))


def test_fixed_names_the_bytes_it_reads(run: Runner) -> None:
    assert message(run, b.fixed(">H"), b"\x01") == (
        "not enough bytes: 0x01 at offset 0x0, expected: 2 bytes"
    )
    assert message(run, b.fixed("<q"), b"\x01\x02\x03\x04\x05\x06\x07") == (
        "not enough bytes: 0x01 at offset 0x0, expected: 8 bytes"
    )
    assert message(run, b.fixed("b"), b"") == (
        "got unexpected end of input at offset 0x0, expected: 1 byte"
    )
    assert message(run, b.byte + b.fixed(">I"), b"\x00\x01\x02\x03") == (
        "not enough bytes: 0x01 at offset 0x1, expected: 4 bytes"
    )


def test_fixed_of_a_format_with_the_size_of_its_alignment() -> None:
    # `@` aligns to the native size, `=`, `<`, `>` and `!` do not.
    assert b.fixed("@i").parse_bytes(struct.pack("@i", 5)) == 5
    assert b.fixed("=i").parse_bytes(struct.pack("=i", 5)) == 5


@pytest.mark.parametrize("fmt", ["Hh", ">HH", "2H", "bB", "3B", "<IH", "x", "3x", "", ">", "0s"])
def test_fixed_of_a_format_that_does_not_read_one_value_raises(fmt: str) -> None:
    with pytest.raises(ValueError, match="reads no single value") as caught:
        b.fixed(fmt)
    assert repr(fmt) in str(caught.value)
    assert "Binary.fixed()" in "\n".join(getattr(caught.value, "__notes__", []))


@pytest.mark.parametrize("fmt", ["z", "<z", ">Hz", "2", "5"])
def test_fixed_of_an_invalid_format_raises(fmt: str) -> None:
    with pytest.raises(ValueError, match="is invalid"):
        b.fixed(fmt)


def test_fixed_of_a_repeated_string_is_one_value(run: Runner) -> None:
    assert run(b.fixed("4s"), b"abcdef") == b"abcd"


def test_byte_gives_an_int(run: Runner) -> None:
    value = run(b.byte, b"\x07")
    assert value == 7
    assert type(value) is int


def test_byte_fails_at_the_end_of_the_input(run: Runner) -> None:
    assert (
        message(run, b.byte, b"") == "got unexpected end of input at offset 0x0, expected: a byte"
    )
    assert message(run, b.byte + b.byte, b"a") == (
        "got unexpected end of input at offset 0x1, expected: a byte"
    )


def test_token_gives_the_bytes_the_pattern_matches(run: Runner) -> None:
    assert run(b.token(rb"[a-z]+"), b"abc1") == b"abc"
    assert run(b.token(rb"\x00\x01"), b"\x00\x01\x02") == b"\x00\x01"
    assert run(b.token(rb"[\x80-\xff]+"), b"\x80\xff\x01") == b"\x80\xff"
    assert type(run(b.token(rb"a+"), b"aa")) is bytes


def test_token_matches_where_the_parsing_is_and_not_at_the_start(run: Runner) -> None:
    assert run(b.byte + b.token(rb"a+"), b"\x00aa") == (0, b"aa")
    assert run(b.token(rb"^a"), b"a") == b"a"
    assert message(run, b.byte + b.token(rb"^a"), b"\x00a") == (
        "got unexpected token: 0x61 at offset 0x1, expected: b'^a'"
    )


def test_token_does_not_search_ahead(run: Runner) -> None:
    assert message(run, b.token(rb"b+"), b"ab") == (
        "got unexpected token: 0x61 at offset 0x0, expected: b'b+'"
    )


def test_token_with_flags(run: Runner) -> None:
    import re

    grammar = b.token(rb"[a-z]+", re.IGNORECASE)
    assert run(grammar, b"aBc1") == b"aBc"
    assert message(run, b.token(rb"[a-z]+"), b"ABC") == (
        "got unexpected token: 0x41 at offset 0x0, expected: b'[a-z]+'"
    )
    dotall = b.token(rb"a.b", re.DOTALL)
    assert run(dotall, b"a\nb") == b"a\nb"
    assert message(run, b.token(rb"a.b"), b"a\nb") == (
        "got unexpected token: 0x61 at offset 0x0, expected: b'a.b'"
    )
    assert run(b.token(rb"^b$", re.MULTILINE), b"b\n") == b"b"


@pytest.mark.parametrize("pattern", [rb"a*", rb"", rb"(?:)", rb"a?", rb"$", rb"^"])
def test_token_that_matches_no_bytes_raises(pattern: bytes) -> None:
    with pytest.raises(ValueError, match="matches no bytes") as caught:
        b.token(pattern)
    assert repr(pattern) in str(caught.value)
    assert "`+` rather than `*`" in "\n".join(getattr(caught.value, "__notes__", []))


def test_a_token_of_a_zero_width_pattern_fails_where_it_matches(run: Runner) -> None:
    for pattern in (rb"\b", rb"(?=a)", rb"(?<=a)"):
        grammar = b.token(pattern)
        assert "expected: b" in message(run, b.byte + grammar, b"aa")
        assert run(many(grammar) + b.byte, b"a") == ([], 97)


def test_a_token_whose_pattern_may_match_nothing_fails_where_it_does(run: Runner) -> None:
    grammar = b.token(rb"a+|(?=b)")
    assert run(grammar, b"aa") == b"aa"
    assert message(run, grammar, b"b") == (
        "got unexpected token: 0x62 at offset 0x0, expected: b'a+|(?=b)'"
    )
    assert run(many(grammar) + b.byte, b"aab") == ([b"aa"], 98)


def test_literal_gives_its_value(run: Runner) -> None:
    assert run(b.literal(b"\x01\x02"), b"\x01\x02\x03") == b"\x01\x02"
    assert run(b.literal(b"\xff"), b"\xff") == b"\xff"
    assert run(b.byte + b.literal(b"ab"), b"\x00ab") == (0, b"ab")
    assert type(run(b.literal(b"a"), b"a")) is bytes


def test_literal_fails_where_the_bytes_differ_or_end(run: Runner) -> None:
    assert message(run, b.literal(b"\x01\x02"), b"\x01\x03") == (
        "got unexpected token: 0x01 at offset 0x0, expected: b'\\x01\\x02'"
    )
    assert message(run, b.literal(b"\x01\x02"), b"\x01") == (
        "got unexpected token: 0x01 at offset 0x0, expected: b'\\x01\\x02'"
    )
    assert message(run, b.literal(b"\x01\x02"), b"") == (
        "got unexpected end of input at offset 0x0, expected: b'\\x01\\x02'"
    )


def test_literal_of_nothing_raises() -> None:
    with pytest.raises(ValueError, match="literal value is empty") as caught:
        b.literal(b"")
    assert "A token consumes at least one byte." in "\n".join(
        getattr(caught.value, "__notes__", [])
    )


def test_previous_gives_the_byte_just_read(run: Runner) -> None:
    assert shape(run(b.byte + b.previous(), b"\x7f")) == shape((0x7F, 0x7F))
    assert shape(run(b.byte + b.previous(0x0F), b"\x7f")) == shape((0x7F, 0x0F))
    assert shape(run(b.byte + b.previous(0xF0), b"\x7f")) == shape((0x7F, 0x70))
    assert shape(run(b.byte + b.previous(0x00), b"\x7f")) == shape((0x7F, 0))


def test_previous_reads_nothing(run: Runner) -> None:
    assert shape(run(b.byte + b.previous() + b.byte, b"\x01\x02")) == shape((1, 1, 2))
    assert shape(run(b.byte + b.previous() + b.previous(), b"\x05")) == shape((5, 5, 5))


def test_previous_sees_the_last_byte_of_what_ran_before_it(run: Runner) -> None:
    assert shape(run(b.fixed(">H") + b.previous(), b"\x01\x02")) == shape((0x0102, 2))
    assert shape(run(b.token(rb"a+") + b.previous(), b"aab")) == shape((b"aa", 97))
    assert shape(run(b.literal(b"xy") + b.previous(), b"xy")) == shape((b"xy", ord("y")))
    assert shape(run(maybe(b.literal(b"z")) + b.byte + b.previous(), b"y")) == shape(
        (None, ord("y"), ord("y"))
    )


def test_previous_at_offset_zero_fails(run: Runner) -> None:
    assert message(run, b.previous(), b"\x07") == (
        "got unexpected token: 0x07 at offset 0x0, expected: the byte before"
    )
    assert message(run, b.previous(0x0F), b"") == (
        "got unexpected end of input at offset 0x0, expected: the byte before"
    )
    assert run(b.previous() | b.byte, b"\x07") == 7


def test_previous_after_what_read_nothing_at_offset_zero_fails(run: Runner) -> None:
    assert "the byte before" in message(
        run, maybe(b.byte.when(lambda v: False)) + b.previous(), b"\x07"
    )
    assert "the byte before" in message(run, pure(1) + b.previous(), b"\x07")


def test_previous_reads_the_byte_before_whatever_ran_before_it(run: Runner) -> None:
    grammar = b.byte + b.byte + (b.previous() >> same)
    assert shape(run(grammar, b"\x01\x02")) == shape((1, 2, 2))
    assert shape(run(b.byte + -b.byte + b.previous(), b"\x01\x02")) == shape((1, 2))


def test_end_succeeds_at_the_end(run: Runner) -> None:
    assert run(b.end, b"") is None
    assert run(b.byte + b.end, b"a") == (97, None)
    assert run(b.end + b.end, b"") == (None, None)
    assert (
        message(run, b.end, b"a")
        == "got unexpected token: 0x61 at offset 0x0, expected: end of input"
    )
    assert message(run, b.byte + b.end, b"ab") == (
        "got unexpected token: 0x62 at offset 0x1, expected: end of input"
    )


def test_end_gives_none_and_reads_nothing(run: Runner) -> None:
    assert run(many(b.byte) + b.end + b.end, b"ab") == ([97, 98], None, None)
    assert run(b.byte + -b.end, b"a") == 97


def test_end_of_input_named(run: Runner) -> None:
    assert (
        message(run, b.end.named("eof"), b"a")
        == "got unexpected token: 0x61 at offset 0x0, expected: eof"
    )


def test_a_binary_parser_without_a_buffer_at_all(run: Runner) -> None:
    assert run(many(b.byte), b"") == []
    assert run(b.end, b"") is None
    assert run(maybe(b.byte), b"") is None
    assert run(counted(b.byte, b.byte) | pure("none"), b"") == "none"


# Errors


def test_an_error_names_the_byte_in_hexadecimal_and_its_offset(run: Runner) -> None:
    assert message(run, b.byte + b.literal(b"a"), b"\x00\xab") == (
        "got unexpected token: 0xab at offset 0x1, expected: b'a'"
    )
    assert message(run, b.token(rb"a"), b"\x0a") == (
        "got unexpected token: 0x0a at offset 0x0, expected: b'a'"
    )


def test_an_offset_past_255_is_in_hexadecimal(run: Runner) -> None:
    grammar = counted(b.fixed(">H"), b.byte) + b.literal(b"z")
    data = b"\x01\x2c" + b"\x00" * 300 + b"y"
    assert (
        message(run, grammar, data) == "got unexpected token: 0x79 at offset 0x12e, expected: b'z'"
    )
    assert message(run, grammar, b"\x01\x2c" + b"\x00" * 300) == (
        "got unexpected end of input at offset 0x12e, expected: b'z'"
    )


def test_the_token_of_an_error_is_usable_as_an_int(run: Runner) -> None:
    with pytest.raises(NoParseError) as caught:
        run(b.byte + b.literal(b"a"), b"\x00\xab")
    token = caught.value.token
    assert isinstance(token, int)
    assert token == 0xAB
    assert token + 1 == 0xAC
    assert bytes([token]) == b"\xab"
    assert hex(token) == "0xab"
    assert loose(token).offset == 1
    assert str(token) == "0xab at offset 0x1"


def test_the_token_of_an_error_at_the_end_of_the_input_is_none(run: Runner) -> None:
    with pytest.raises(NoParseError) as caught:
        run(b.byte + b.byte, b"\x00")
    assert caught.value.token is None
    assert caught.value.state.pos == 1
    assert caught.value.description is None or caught.value.description.start is None


def test_the_description_of_an_error_has_no_place(run: Runner) -> None:
    with pytest.raises(NoParseError) as caught:
        run(b.byte + b.literal(b"a"), b"\x00\xab")
    description = caught.value.description
    assert description is None or (description.start is None and description.end is None)
    assert caught.value.render() == str(caught.value)


def test_the_error_at_the_end_of_the_input_has_the_offset_of_the_end(run: Runner) -> None:
    assert message(run, counted(b.byte, b.byte), b"\x02\x01") == (
        "got unexpected end of input at offset 0x2, expected: a byte"
    )
    assert message(run, b.literal(b"a") | b.literal(b"b"), b"") == (
        "got unexpected end of input at offset 0x0, expected: b'a' or b'b'"
    )


def test_an_error_says_what_every_alternative_expected(run: Runner) -> None:
    assert message(run, b.byte + (b.literal(b"a") | b.literal(b"b")), b"\x00c") == (
        "got unexpected token: 0x63 at offset 0x1, expected: b'a' or b'b'"
    )
    assert "expected: b'a' or b'b+'" in message(run, b.literal(b"a") | b.token(rb"b+"), b"c")


def test_a_check_failure_in_a_byte_grammar_has_the_offset_in_hexadecimal(run: Runner) -> None:
    grammar = b.byte + b.fixed(">H").check(lambda v: v < 10, "too big") + b.byte
    assert message(run, grammar, b"\x00\x01\x00\x07") == "offset 0x1: too big"
    assert run(grammar, b"\x00\x00\x09\x07") == (0, 9, 7)


def test_a_check_failure_that_spans_bytes_reports_the_start(run: Runner) -> None:
    grammar = b.byte + b.fixed(">I").check(lambda v: v < 10, "too big")
    with pytest.raises(ParseErrors) as caught:
        run(grammar, b"\x00\x00\x00\x01\x00")
    (error,) = caught.value.exceptions
    assert str(error) == "offset 0x1: too big"
    assert error.state.pos == 1


def test_several_failed_checks_are_reported_in_order_of_offset(run: Runner) -> None:
    item = b.byte.check(lambda v: v < 3, "too big")
    grammar = many(item)
    with pytest.raises(ParseErrors) as caught:
        run(grammar, b"\x01\x05\x02\x09")
    assert [str(e) for e in caught.value.exceptions] == [
        "offset 0x1: too big",
        "offset 0x3: too big",
    ]


def test_a_check_failure_and_a_parse_failure_are_reported_together(run: Runner) -> None:
    item = b.byte.check(lambda v: v < 3, "too big")
    grammar = item + item + b.literal(b"x")
    with pytest.raises(ParseErrors) as caught:
        run(grammar, b"\x09\x01\x7a")
    assert [str(e) for e in caught.value.exceptions] == [
        "offset 0x0: too big",
        "got unexpected token: 0x7a at offset 0x2, expected: b'x'",
    ]
    assert caught.value.exceptions[1].token == 0x7A


def test_named_replaces_what_a_byte_parser_expected(run: Runner) -> None:
    assert message(run, b.byte.named("tag"), b"") == (
        "got unexpected end of input at offset 0x0, expected: tag"
    )
    assert message(run, b.token(rb"a+").named("As"), b"b") == (
        "got unexpected token: 0x62 at offset 0x0, expected: As"
    )
    assert message(run, b.fixed(">H").named("size"), b"\x01") == (
        "not enough bytes: 0x01 at offset 0x0, expected: size"
    )
    assert message(run, b.previous().named("tag"), b"") == (
        "got unexpected end of input at offset 0x0, expected: tag"
    )
    assert (
        message(run, b.end.named("eof"), b"a")
        == "got unexpected token: 0x61 at offset 0x0, expected: eof"
    )


def test_a_name_is_not_a_name_of_what_follows(run: Runner) -> None:
    grammar = b.literal(b"a").named("first") + b.literal(b"b")
    assert (
        message(run, grammar, b"ac") == "got unexpected token: 0x63 at offset 0x1, expected: b'b'"
    )


def test_context_labels_the_errors_of_a_byte_grammar(run: Runner) -> None:
    grammar = (b.literal(b"a") + b.literal(b"b")).context("pair")
    assert message(run, grammar, b"ac") == (
        "got unexpected token: 0x63 at offset 0x1, expected: b'b' in pair"
    )
    nested = ((b.literal(b"a") + b.literal(b"b")).context("pair") + b.byte).context("record")
    assert message(run, nested, b"ac") == (
        "got unexpected token: 0x63 at offset 0x1, expected: b'b' in pair in record"
    )


def test_the_context_of_an_error_is_its_attribute(run: Runner) -> None:
    with pytest.raises(NoParseError) as caught:
        run((b.literal(b"a") + b.literal(b"b")).context("pair").context("record"), b"ac")
    assert caught.value.context == ("pair", "record")
    assert caught.value.expected == ("b'b'",)


def test_recover_goes_on_in_a_byte_grammar(run: Runner) -> None:
    record = (b.literal(b"a") + b.literal(b"x")).recover(b.literal(b"\n"))
    grammar = many(record + -b.literal(b"\n")) + -b.end
    with pytest.raises(ParseErrors) as caught:
        run(grammar, b"ab\nac\n")
    errors = caught.value.exceptions
    assert [e.state.pos for e in errors] == [1, 4]
    assert [str(e).split(":")[0] for e in errors] == ["offset 0x1", "offset 0x4"]
    assert all(str(e).endswith("expected: b'x'") for e in errors)
    assert run(grammar, b"ax\nax\n") == [(b"a", b"x"), (b"a", b"x")]


def test_a_recovered_error_names_the_byte_in_hexadecimal(run: Runner) -> None:
    record = (b.literal(b"a") + b.literal(b"x")).recover(b.literal(b"\n"))
    with pytest.raises(ParseErrors) as caught:
        run(many(record + -b.literal(b"\n")) + -b.end, b"ab\n")
    (error,) = caught.value.exceptions
    assert str(error) == "offset 0x1: got unexpected token: 0x62, expected: b'x'"


def test_recover_with_a_default_gives_it_where_the_record_failed(run: Runner) -> None:
    record = (b.literal(b"a") + b.literal(b"x")).recover(b.literal(b"\n"), (b"?", b"?"))
    grammar = many(record + -b.literal(b"\n"))
    assert shape(run(grammar, b"ax\n")) == shape([(b"a", b"x")])


def test_recover_gives_up_where_until_is_never_found(run: Runner) -> None:
    record = (b.literal(b"a") + b.literal(b"x")).recover(b.literal(b"\n"))
    with pytest.raises((NoParseError, ParseErrors)):
        run(record, b"ab")


def test_text_parsers_on_bytes_raise_what_re_raises() -> None:
    for grammar in (plain.token("a"), plain.literal("a"), plain.token("a") + plain.end):
        for parse in (grammar.parse_bytes, grammar.compile().parse_bytes):
            with pytest.raises(TypeError):
                parse(b"a")


def test_binary_parsers_on_text_raise_what_the_standard_library_raises() -> None:
    for grammar in (b.token(rb"a"), b.literal(b"a"), b.fixed(">H"), b.byte + b.previous()):
        for parse in (grammar.parse_text, grammar.compile().parse_text):
            with pytest.raises(TypeError):
                parse("ab")


def test_a_byte_and_the_end_work_on_text_and_give_its_characters() -> None:
    assert b.byte.parse_text("a") == "a"
    assert b.end.parse_text("") is None
    assert (b.byte + b.byte + b.end).parse_text("ab") == ("a", "b", None)


def test_tok_on_bytes_is_a_parser_of_tokens_that_bytes_are_not() -> None:
    with pytest.raises(AttributeError, match="type"):
        tok("a").parse_bytes(b"a")
    with pytest.raises(ValueError, match="without a tokenizer"):
        tok("a").compile().parse_bytes(b"a")


def test_some_reads_bytes_as_integers_in_a_byte_grammar(run: Runner) -> None:
    grammar = some(lambda v: v == 97) + b.byte
    assert shape(run(grammar, b"ab")) == shape((97, 98))
    assert shape(run(prefixed(b.byte, many(some(lambda v: v > 96))), b"\x02ab")) == shape([97, 98])


def test_parse_of_bytes_is_a_parse_of_integers() -> None:
    grammar = prefixed(some(lambda v: True), many(some(lambda v: True)))
    assert grammar.parse(b"\x02ab") == [97, 98]
    assert grammar.compile().parse(b"\x02ab") == [97, 98]
    assert prefixed(some(lambda v: True)).parse(b"\x02ab") == b"ab"


def test_parse_bytes_of_a_bytearray_and_a_memoryview() -> None:
    grammar = prefixed(b.byte, many(b.byte)) + b.fixed(">H")
    data = b"\x02ab\x00\x07"
    assert shape(grammar.parse_bytes(data)) == shape(([97, 98], 7))
    assert shape(grammar.parse_bytes(loose(bytearray(data)))) == shape(([97, 98], 7))


def test_the_message_of_an_error_does_not_depend_on_the_data_type() -> None:
    grammar = b.byte + b.literal(b"a")
    for data in (b"\x00b", bytearray(b"\x00b")):
        with pytest.raises(NoParseError, match="0x62 at offset 0x1"):
            grammar.parse_bytes(loose(data))


# commit()


@pytest.mark.parametrize(
    "form",
    [
        pytest.param(lambda: prefixed(b.byte), id="prefixed"),
        pytest.param(lambda: prefixed(b.byte, many(b.literal(b"a"))), id="prefixed body"),
        pytest.param(lambda: prefixed(b.byte, b.byte + b.literal(b"a")), id="prefixed body fails"),
        pytest.param(lambda: counted(b.byte, b.literal(b"a")), id="counted"),
        pytest.param(lambda: counted(b.byte, b.fixed(">H")), id="counted of fixed"),
        pytest.param(lambda: switch(b.byte, {1: b.literal(b"a"), 2: b.byte}), id="switch"),
        pytest.param(lambda: switch(b.byte, {range(3): b.literal(b"a")}), id="switch range"),
        pytest.param(lambda: b.fixed(">H"), id="fixed"),
        pytest.param(lambda: b.byte + b.previous(), id="previous"),
        pytest.param(lambda: b.byte + b.end, id="end"),
        pytest.param(lambda: b.token(rb"a+"), id="token"),
        pytest.param(lambda: tree, id="tree"),
        pytest.param(lambda: sized, id="recursive prefixed"),
        pytest.param(lambda: expression, id="recursive switch"),
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"\x01",
        b"\x01a",
        b"\x01b",
        b"\x02aa",
        b"\x02ab",
        b"\x02\x00\x01",
        b"\x01\x00\x07",
        b"\x01\x01\x00",
        b"\x00\x00",
        b"aa",
        b"\x03\x02a\x01",
    ],
)
def test_commit_around_a_form_parses_as_the_parser(form: Callable[[], P], data: bytes) -> None:
    for grammar in (
        commit(form()),
        commit(form()) | pure("other"),
        commit(form() + b.byte) | pure("other"),
        (b.literal(b"\x01") + commit(form())) | pure("other"),
        many(commit(form())) + b.end,
        maybe(commit(form())),
    ):
        assert outcome(grammar.compile().parse_bytes, data) == outcome(grammar.parse_bytes, data)


def test_a_failure_after_a_commit_of_a_region_is_final(run: Runner) -> None:
    grammar = (b.literal(b"\x01") + commit(prefixed(b.byte, b.literal(b"a")))) | pure("other")
    assert run(grammar, b"\x01\x01a") == (b"\x01", b"a")


def test_a_failure_after_a_commit_of_counted_is_final(run: Runner) -> None:
    grammar = (b.literal(b"\x01") + commit(counted(b.byte, b.literal(b"a")))) | pure("other")
    assert message(run, grammar, b"\x01\x02ab") == (
        "got unexpected token: 0x62 at offset 0x3, expected: b'a'"
    )
    assert run(grammar, b"\x02") == "other"


def test_a_failure_after_a_commit_of_a_switch_is_final(run: Runner) -> None:
    grammar = (b.literal(b"\x01") + commit(switch(b.byte, {1: b.literal(b"a")}))) | pure("other")
    assert message(run, grammar, b"\x01\x01b") == (
        "got unexpected token: 0x62 at offset 0x2, expected: b'a'"
    )
    assert run(grammar, b"\x01\x02") == "other"
    assert run(grammar, b"\x02") == "other"


# A forward declaration of every form: a grammar nested through a name.


def test_a_region_declared_before_it_is_defined(run: Runner) -> None:
    declared: P = forward_decl()
    grammar = prefixed(b.byte, declared)
    declared.define(many(b.byte))
    assert run(grammar, b"\x02ab") == [97, 98]


def test_a_recursive_grammar_of_the_three_forms(run: Runner) -> None:
    value: P = forward_decl()
    value.define(
        switch(
            b.byte,
            {
                0: pure(None),
                1: b.fixed(">H"),
                2: prefixed(b.byte),
                3: counted(b.byte, value),
                4: prefixed(b.byte, many(value)),
            },
        )
    )
    data = b"\x03\x03\x01\x00\x05\x02\x01a\x04\x04\x01\x00\x07\x00" + b"\x00" * 0
    assert run(value, b"\x01\x00\x05") == 5
    assert run(value, b"\x03\x02\x01\x00\x05\x02\x01a") == [5, b"a"]
    assert run(value, b"\x04\x05\x01\x00\x05\x00\x00") == [5, None, None]
    assert run(value, data) == run(value, data)
    assert message(run, value, b"\x03\x01\x07") == (
        "got unexpected token: 0x07 at offset 0x2, expected: one of the cases of a byte"
    )


# Found by review: the furthest position, regions and what an error says around them.


def test_recover_at_its_first_token_fails_though_another_branch_got_further(run: Runner) -> None:
    assert run(longest(b.byte, b.literal(b"q").recover(b.end)), b"ab") == 97
    grammar = (
        (b.byte + b.literal(b"y")) | (b.literal(b"q").recover(b.literal(b";")) + b.byte) | b.byte
    )
    assert run(grammar, b"x;") == 120


def test_oneplus_of_an_ignored_rule_keeps_its_first_item(run: Runner) -> None:
    rule: P = forward_decl()
    rule.define(-(b.literal(b"ab") + maybe(rule)))
    assert len(run(oneplus(rule), b"ab")) == 1


def test_a_rule_of_an_ignored_value_gives_it_to_a_context(run: Runner) -> None:
    rule: P = forward_decl()
    rule.define(-(b.literal(b"a") + maybe(rule)))
    assert isinstance(run(rule.context("c"), b"a"), IgnoredValue)


def test_a_count_that_is_no_integer_raises(run: Runner) -> None:
    with pytest.raises(TypeError):
        run(counted(b.fixed(">f"), b.byte), b"\x40\x00\x00\x00ab")


def test_a_negative_count_fails(run: Runner) -> None:
    with pytest.raises(NoParseError, match="count -1 is negative"):
        run(counted(b.fixed(">b"), b.byte), b"\xffa")


def test_counted_of_what_reads_nothing_raises_at_once(run: Runner) -> None:
    with pytest.raises(ValueError, match="inside counted"):
        run(counted(b.fixed(">H"), maybe(b.literal(b"x"))), b"\xff\xff")


def test_a_commit_sees_the_length_read(run: Runner) -> None:
    with pytest.raises(NoParseError, match="length 9 runs past the end"):
        run(commit(prefixed(b.byte)) | b.byte, b"\x09hi")


def test_recover_sees_the_length_read(run: Runner) -> None:
    grammar = many(prefixed(b.byte).recover(b.literal(b";")) + -b.literal(b";")) + -b.end
    with pytest.raises(ParseErrors, match="1 parsing error"):
        run(grammar, b"\x09hi;")


def test_what_a_closed_region_expected_is_no_part_of_the_next_error(run: Runner) -> None:
    grammar = prefixed(b.byte, many(b.literal(b"a"))) + b.literal(b"z")
    assert message(run, grammar, b"\x01aq") == (
        "got unexpected token: 0x71 at offset 0x2, expected: b'z'"
    )


def test_what_the_length_left_expected_stays_in_the_error(run: Runner) -> None:
    length = b.byte + -many(b.literal(b"\xaa"))
    assert message(run, prefixed(length, b.literal(b"z")), b"\x01q") == (
        "got unexpected token: 0x71 at offset 0x1, expected: b'\\xaa' or b'z'"
    )


def test_a_check_on_a_length_stays_with_the_failure(run: Runner) -> None:
    with pytest.raises(ParseErrors) as caught:
        run(prefixed(b.byte.check(lambda n: n < 5, "too long")), b"\x09hi")
    assert len(caught.value.exceptions) == 2
    with pytest.raises(ParseErrors) as caught:
        run(switch(b.byte.check(lambda n: n < 5, "too big"), {1: b.byte}), b"\x09")
    assert len(caught.value.exceptions) == 2


def test_a_region_left_unread_says_so(run: Runner) -> None:
    assert message(run, prefixed(b.byte, b.literal(b"a")), b"\x02ab") == (
        "bytes left in the region: 0x62 at offset 0x2"
    )
