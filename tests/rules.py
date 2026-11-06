"""Rules on values for the tests, and what the property tests add to a grammar with them."""

from collections.abc import Callable
from typing import Any

from hypothesis import strategies as st

from funcparserlib.parser import Invalid, Parser, pure

P = Parser[Any, Any]
Grammars = st.SearchStrategy[P]


def kept[T](pred: Callable[[T], bool], message: str) -> Callable[[T], T]:
    """A function for `Parser.validated()` that reports `message` and keeps the value."""

    def rule(value: T) -> T:
        """The value, with an error when `pred` is false for it."""
        if not pred(value):
            raise Invalid(message, keep=True)
        return value

    return rule


def rejected[T](pred: Callable[[T], bool], message: str) -> Callable[[T], T]:
    """A function for `Parser.validated()` that rejects a value `pred` is false for."""

    def rule(value: T) -> T:
        """The value, rejected when `pred` is false for it."""
        if not pred(value):
            raise Invalid(message)
        return value

    return rule


def nothing(_: object) -> None:
    """No value at all, for a conversion that gives `None`."""
    return


def spread(*values: object) -> tuple[object, ...]:
    """The values `starmap()` passes, together."""
    return values


def conversions(kids: Grammars, wrap: Callable[[Any], Any]) -> list[Grammars]:
    """What gives another value and rejects nothing, so generated code decides it.

    One branch of a grammar, so these forms do not crowd out the others.
    """
    forms: list[Callable[[P], P]] = [
        lambda k: (k + pure("p") + pure("q")).starmap(spread),
        lambda k: k.validated(wrap),
        lambda k: k.validated(nothing),
        lambda k: (-k).validated(wrap),
    ]
    return [st.tuples(st.sampled_from(forms), kids).map(lambda t: t[0](t[1]))]


def breakages(kids: Grammars, short: Callable[[Any], bool]) -> list[Grammars]:
    """`validated()` that reports, and the `recover()` that gives a broken part a value.

    One branch of a grammar, so these forms do not crowd out the others.
    """
    forms: list[Callable[[P, P], P]] = [
        lambda k, _: k.validated(kept(short, "too long")),
        lambda k, _: k.validated(rejected(short, "too long")),
        lambda k, _: k.validated(rejected(short, "too long")).recover(default="broken"),
        lambda k, _: k.recover(default="broken"),
        lambda k, until: k.recover(until, default="skipped"),
    ]
    return [st.tuples(st.sampled_from(forms), kids, kids).map(lambda t: t[0](t[1], t[2]))]
