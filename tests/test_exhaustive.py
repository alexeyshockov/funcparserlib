"""Every small grammar, on every short text, gives the same result compiled and interpreted.

The random grammars of the other tests find what is common. A bug of the compiler can sit in one
shape, such as a loop of a loop inside a recursive grammar, which they may never draw. So this walks
all of them up to a size, and finds it on every run.
"""

from collections.abc import Callable, Iterator
from itertools import product
from typing import Any

import pytest

from funcparserlib.parser import Parser, commit, forward_decl, many, maybe, oneplus, pure
from funcparserlib.text import Text
from tests.rules import kept, rejected
from tests.test_text import outcome

P = Parser[Any, Any]

text = Text()
skipping = Text(skip=r"\s+")
TEXTS = ["".join(chars) for size in range(4) for chars in product("ab", repeat=size)]


def wrap(value: object) -> tuple[str, object]:
    """The value in a tuple: a conversion to another shape."""
    return ("wrapped", value)


def short(value: object) -> bool:
    """A rule that some values break."""
    return len(repr(value)) % 3 != 0


def recursive(body: P) -> P:
    """A grammar that reads `body` inside itself."""
    declared: P = forward_decl()
    declared.define(skipping.token("a") + maybe(body + declared))
    return declared


LEAVES: list[Callable[[], P]] = [
    lambda: skipping.token("a"),
    lambda: text.literal("b"),
    lambda: -text.literal("a"),
    lambda: pure("p"),
]
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
BINARY: list[Callable[[P, P], P]] = [lambda p, q: p + q, lambda p, q: p | q]


def chains(depth: int) -> Iterator[Callable[[], P]]:
    """Every leaf under up to `depth` of the unary forms."""
    if depth == 0:
        yield from LEAVES
        return
    yield from chains(depth - 1)
    for form, make in product(UNARY, chains(depth - 1)):
        yield lambda form=form, make=make: form(make())


def grammars() -> Iterator[P]:
    """The chains of up to 3 forms, and every pair of those of up to 1."""
    for make in chains(3):
        yield make()
    small = list(chains(1))
    for form, left, right in product(BINARY, small, small):
        yield form(left(), right())
    for unary, form, left, right in product(UNARY, BINARY, small, small):
        yield unary(form(left(), right()))


def differences(grammar: P) -> Iterator[tuple[str, object, object]]:
    """Where the compiled grammar gives another result than `grammar`."""
    code = grammar.compile()
    for source in TEXTS:
        expected = outcome(grammar.parse_text, source)
        found = outcome(code.parse_text, source)
        if found != expected:
            yield source, expected, found


@pytest.mark.exhaustive
@pytest.mark.timeout(300)
def test_every_small_grammar_parses_the_same_compiled() -> None:
    found = [difference for grammar in grammars() for difference in differences(grammar)]
    assert found == []
