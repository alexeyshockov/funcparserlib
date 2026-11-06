"""A rule on a real value, for the tests that do not care how a rule is written."""

from collections.abc import Callable

from funcparserlib.parser import Invalid


def kept[T](pred: Callable[[T], bool], message: str) -> Callable[[T], T]:
    """A function for `Parser.validated()` that reports `message` and keeps the value."""

    def rule(value: T) -> T:
        if not pred(value):
            raise Invalid(message, keep=True)
        return value

    return rule
