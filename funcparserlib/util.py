import re
from bisect import bisect_right
from collections.abc import Callable, Sequence
from functools import wraps
from inspect import signature
from typing import Any, TypeVar, final, overload

from ._commons import with_hint
from .lexer import Place

A = TypeVar("A")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

__all__ = ["Lines", "pretty_tree", "expand_tuple_args"]


@final
class Lines:
    """The lines of a text, to turn an offset into a `Place` and back.

    It is built once from the text, and each conversion then takes a binary search, so use one for
    as many positions as you have. A `Place` is the (_line_, _column_) of a
    `funcparserlib.lexer.Token`, both counted from 1, and lines are counted by `"\\n"` as there.

    Examples:
        ```pycon
        >>> lines = Lines("ab\\ncd")
        >>> lines.place(4)
        (2, 2)
        >>> lines.offset((2, 2))
        4

        ```
    """

    __slots__ = ("_length", "_starts")

    def __init__(self, text: str) -> None:
        """The lines of `text`."""
        self._length = len(text)
        self._starts = [0, *(m.end() for m in re.finditer("\n", text))]

    def place(self, offset: int) -> Place:
        """The line and the column of `offset`, which may be the end of the text.

        It raises `ValueError` for an offset outside the text.
        """
        if not 0 <= offset <= self._length:
            raise with_hint(
                ValueError(f"offset {offset} is outside the text"),
                f"The text has {self._length} characters.",
            )
        line = bisect_right(self._starts, offset)
        return line, offset - self._starts[line - 1] + 1

    def offset(self, place: Place) -> int:
        """The offset of the line and the column in `place`.

        It raises `ValueError` for a place outside the text.
        """
        line, column = place
        if not 1 <= line <= len(self._starts) or column < 1:
            raise with_hint(
                ValueError(f"place {place} is outside the text"),
                f"The text has {len(self._starts)} lines, and lines and columns start at 1.",
            )
        offset = self._starts[line - 1] + column - 1
        last = self._starts[line] if line < len(self._starts) else self._length + 1
        if offset >= last:
            raise with_hint(
                ValueError(f"place {place} is outside the text"),
                f"Line {line} has {last - self._starts[line - 1] - 1} characters.",
            )
        return offset


def pretty_tree(x: A, kids: Callable[[A], Sequence[A]], show: Callable[[A], str]) -> str:
    """Return a pseudo-graphic tree representation of the object `x` similar to the `tree` command
    in Unix.

    It applies the parameter `show` to get a textual representation of the objects to show.

    It applies the parameter `kids` to list the children of the object to show.

    Examples:
        ```pycon
        >>> print(pretty_tree(
        ...     ["foo", ["bar", "baz"], "quux"],
        ...     lambda obj: obj if isinstance(obj, list) else [],
        ...     lambda obj: "[]" if isinstance(obj, list) else str(obj),
        ... ))
        []
        |-- foo
        |-- []
        |   |-- bar
        |   `-- baz
        `-- quux

        ```
    """
    (MID, END, CONT, LAST, ROOT) = ("|-- ", "`-- ", "|   ", "    ", "")

    def rec(obj: A, indent: str, sym: str) -> str:
        line = indent + sym + show(obj)
        obj_kids = kids(obj)
        if len(obj_kids) == 0:
            return line
        else:
            if sym == MID:
                next_indent = indent + CONT
            elif sym == ROOT:
                next_indent = indent + ROOT
            else:
                next_indent = indent + LAST
            chars = [MID] * (len(obj_kids) - 1) + [END]
            lines = [rec(kid, next_indent, sym) for kid, sym in zip(obj_kids, chars, strict=True)]
            return "\n".join([line] + lines)

    return rec(x, "", ROOT)


@overload
def expand_tuple_args(f: Callable[[T1, T2], A]) -> Callable[[tuple[T1, T2]], A]: ...


@overload
def expand_tuple_args(f: Callable[[T1, T2, T3], A]) -> Callable[[tuple[T1, T2, T3]], A]: ...


@overload
def expand_tuple_args(f: Callable[[T1, T2, T3, T4], A]) -> Callable[[tuple[T1, T2, T3, T4]], A]: ...


@overload
def expand_tuple_args(
    f: Callable[[T1, T2, T3, T4, T5], A],
) -> Callable[[tuple[T1, T2, T3, T4, T5]], A]: ...


def expand_tuple_args(f: Callable[..., A]) -> Callable[..., A]:
    """Adapt `f` to take its positional arguments as one tuple, the value of `p1 + ... + pN`.

    A method, whose first parameter is `self` or `cls`, keeps it and takes the tuple second, so a
    method decorated with it works with `>>` as well. A function defined outside a class is no
    method, whatever its parameters are called.

    Examples:
        ```pycon
        >>> from funcparserlib.parser import a
        >>> expr = (a("1") + a("2")) >> expand_tuple_args(lambda x, y: int(x) + int(y))
        >>> expr.parse("12")
        3

        ```
    """

    @wraps(f)
    def _unwrap_fn(args: tuple[Any, ...]) -> A:
        return f(*args)

    @wraps(f)
    def _unwrap_method(cls_or_self: type | object, args: tuple[Any, ...]) -> A:
        return f(cls_or_self, *args)

    return _unwrap_method if _is_method(f) else _unwrap_fn


def _is_method(f: Callable[..., Any]) -> bool:
    """Whether `f` is defined in a class and takes `self` or `cls`; a built-in has no signature."""
    scope = getattr(f, "__qualname__", "").rpartition(".")[0]
    if not scope or scope.endswith("<locals>"):
        return False
    try:
        return next(iter(signature(f).parameters), None) in {"self", "cls"}
    except (TypeError, ValueError):
        return False
