# Copyright © 2009/2023 Andrey Vlasovskikh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from functools import wraps
from inspect import signature
from typing import TypeVar, Callable, Sequence, overload, Union

_A = TypeVar("_A")


def pretty_tree(
    x: _A,
    kids: Callable[[_A], Sequence[_A]],
    show: Callable[[_A], str],
) -> str:
    """Return a pseudo-graphic tree representation of the object `x` similar to the
    `tree` command in Unix.

    Type: `(T, Callable[[T], List[T]], Callable[[T], str]) -> str`

    It applies the parameter `show` (which is a function of type `(T) -> str`) to get a
    textual representation of the objects to show.

    It applies the parameter `kids` (which is a function of type `(T) -> List[T]`) to
    list the children of the object to show.

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

    def rec(obj: _A, indent: str, sym: str) -> str:
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
            lines = [rec(kid, next_indent, sym) for kid, sym in zip(obj_kids, chars)]
            return "\n".join([line] + lines)

    return rec(x, "", ROOT)


# Additional type variables tuple overloads (up to 5-tuple)
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")


@overload
def expand_tuple_args(f: Callable[[_T1, _T2], _A]) -> Callable[[tuple[_T1, _T2]], _A]:
    ...


@overload
def expand_tuple_args(
    f: Callable[[_T1, _T2, _T3], _A]
) -> Callable[[tuple[_T1, _T2, _T3]], _A]:
    ...


@overload
def expand_tuple_args(
    f: Callable[[_T1, _T2, _T3, _T4], _A]
) -> Callable[[tuple[_T1, _T2, _T3, _T4]], _A]:
    ...


@overload
def expand_tuple_args(
    f: Callable[[_T1, _T2, _T3, _T4, _T5], _A]
) -> Callable[[tuple[_T1, _T2, _T3, _T4, _T5]], _A]:
    ...


def expand_tuple_args(f: Callable[..., _A]) -> Callable[..., _A]:
    @wraps(f)
    def _unwrap_fn(args: tuple) -> _A:
        return f(*args)

    @wraps(f)
    def _unwrap_method(cls_or_self: Union[type, object], args: tuple) -> _A:
        return f(cls_or_self, *args)

    sig = signature(f)
    if (params := tuple(sig.parameters.keys())) and params[0] in {"self", "cls"}:
        return _unwrap_method
    else:
        return _unwrap_fn
