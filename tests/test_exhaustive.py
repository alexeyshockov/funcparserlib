"""Every small grammar, on every short text, gives the same result compiled and interpreted.

The random grammars of the other tests find what is common. A bug of the compiler can sit in one
shape, such as a loop of a loop inside a recursive grammar, which they may never draw. So this walks
all of them up to a size, and finds it on every run.
"""

from collections.abc import Callable, Iterator, Sequence
from itertools import islice, product
from typing import Any

import pytest

from funcparserlib.parser import (
    Parser,
    ParsingFailure,
    ParsingResult,
    ParsingSuccess,
    State,
    Text,
    a,
    commit,
    finished,
    forward_decl,
    lit,
    many,
    maybe,
    oneplus,
    pure,
    some,
)
from tests.rules import kept, rejected
from tests.test_text import outcome, shape

P = Parser[Any, Any]

skipping = Text(skip=r"\s+")
# The walk in parts, for the workers of a parallel run.
SHARDS = 8
TEXTS = ["".join(chars) for size in range(4) for chars in product("ab", repeat=size)]
BYTES = [bytes(values) for size in range(4) for values in product(b"ab", repeat=size)]


def wrap(value: object) -> tuple[str, object]:
    """The value in a tuple: a conversion to another shape."""
    return ("wrapped", value)


def short(value: object) -> bool:
    """A rule that some values break."""
    return len(repr(value)) % 3 != 0


def recursive(body: P) -> P:
    """A grammar that reads `body` inside itself."""
    declared: P = forward_decl()
    declared.define(skipping.regex("a") + maybe(body + declared))
    return declared


def reading(item: object) -> P:
    """A parsing function that reads `item`."""

    def run(tokens: Any, s: State) -> ParsingResult[Any]:
        if s.pos < len(tokens) and tokens[s.pos] == item:
            return ParsingSuccess(tokens[s.pos], s.reset_pos(s.pos + 1))
        return ParsingFailure(s.expect(p))

    p: P = Parser(run).named("reading")
    return p


def without_tok(first: object, second: object) -> list[Callable[[], P]]:
    """Leaves that say nothing of what they read, so the call of the parse decides it."""
    return [
        lambda: a(first),
        lambda: some(lambda item: item == second),
        lambda: finished,
        lambda: pure("p"),
        lambda: reading(first),
        lambda: a(first).bind(lambda value: a(second) | pure(value)),
    ]


LEAVES: list[Callable[[], P]] = [
    lambda: skipping.regex("a"),
    lambda: lit("b"),
    lambda: -lit("a"),
    *without_tok("a", "b"),
]
BYTE_LEAVES = without_tok(ord("a"), ord("b"))
UNARY: list[Callable[[P], P]] = [
    many,
    maybe,
    oneplus,
    recursive,
    commit,
    lambda p: -p,
    lambda p: p >> wrap,
    lambda p: p.validated(wrap),
    lambda p: p.validated(rejected(short, "rejected")),
    lambda p: p.validated(kept(short, "kept")),
    lambda p: p.recover(default="broken"),
]
BYTE_UNARY = [form for form in UNARY if form is not recursive]
BINARY: list[Callable[[P, P], P]] = [lambda p, q: p + q, lambda p, q: p | q]


def chains(
    depth: int, leaves: list[Callable[[], P]], unary: list[Callable[[P], P]]
) -> Iterator[Callable[[], P]]:
    """Every leaf under up to `depth` of the unary forms."""
    if depth == 0:
        yield from leaves
        return
    yield from chains(depth - 1, leaves, unary)
    for form, make in product(unary, chains(depth - 1, leaves, unary)):
        yield lambda form=form, make=make: form(make())


def grammars(leaves: list[Callable[[], P]], unary: list[Callable[[P], P]]) -> Iterator[P]:
    """The chains of up to 3 forms, and every pair of those of up to 1."""
    for make in chains(3, leaves, unary):
        yield make()
    small = list(chains(1, leaves, unary))
    for form, left, right in product(BINARY, small, small):
        yield form(left(), right())
    for outer, form, left, right in product(unary, BINARY, small, small):
        yield outer(form(left(), right()))


def differences(grammar: P, call: str, sources: Sequence[Any]) -> Iterator[tuple[object, ...]]:
    """Where the compiled grammar gives another result than `grammar`."""
    code = grammar.compile()
    for source in sources:
        expected = outcome(getattr(grammar, call), source)
        found = outcome(getattr(code, call), source)
        if found != expected:
            yield source, expected, found
        if expected[0] in ("value", "error", "errors"):
            value = expected if expected[0] == "value" else ("value", shape(None))
            found = outcome(code.try_parse, source)
            if found != value:
                yield "try_parse", source, value, found


@pytest.mark.exhaustive
@pytest.mark.timeout(120)
@pytest.mark.parametrize("shard", range(SHARDS))
def test_every_small_grammar_parses_the_same_compiled(shard: int) -> None:
    every = islice(grammars(LEAVES, UNARY), shard, None, SHARDS)
    found = [d for grammar in every for d in differences(grammar, "parse_text", TEXTS)]
    assert found == []


@pytest.mark.exhaustive
@pytest.mark.timeout(120)
@pytest.mark.parametrize("shard", range(SHARDS))
def test_every_small_grammar_without_tok_reads_bytes_the_same_compiled(shard: int) -> None:
    every = islice(grammars(BYTE_LEAVES, BYTE_UNARY), shard, None, SHARDS)
    found = [d for grammar in every for d in differences(grammar, "parse_bytes", BYTES)]
    assert found == []
