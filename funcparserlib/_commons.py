"""Common helpers.

Generated; propose changes in an issue rather than editing this file. The project's own helpers go
in `_utils.py`.
"""

import dataclasses as dc
import functools
from collections.abc import Callable
from typing import Any, TypeVar, dataclass_transform, overload

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


@overload
def refclass(cls: type[T], /) -> type[T]: ...


@overload
def refclass(
    *, frozen: bool = False, slots: bool = True, kw_only: bool = False, init: bool = True
) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(eq_default=False, field_specifiers=(dc.field,))
def refclass(
    cls: type[T] | None = None,
    /,
    *,
    frozen: bool = False,
    slots: bool = True,
    kw_only: bool = False,
    init: bool = True,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Declare a reference-semantics class with dataclass field syntax.

    The counterpart to `@valueclass`: this one keeps Python's default identity `__eq__` / `__hash__`
    while still generating `__init__` from the annotations. `frozen=True` makes the fields read-only
    to type checkers; at runtime the class is not frozen, and a field set after `__init__` (in
    `__post_init__`) is set with `object.__setattr__`.
    """
    del frozen  # Read by the type checkers alone, through `dataclass_transform`.

    def wrap(c: type[T]) -> type[T]:
        _check_slots(c, slots)
        return dc.dataclass(eq=False, slots=slots, kw_only=kw_only, init=init)(c)

    return wrap if cls is None else wrap(cls)


@overload
def valueclass(cls: type[T], /) -> type[T]: ...


@overload
def valueclass(
    *,
    slots: bool = True,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    kw_only: bool = False,
) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(frozen_default=True, field_specifiers=(dc.field,))
def valueclass(
    cls: type[T] | None = None,
    /,
    *,
    slots: bool = True,
    init: bool = True,
    repr: bool = True,  # noqa: A002 — `dataclass`'s own keyword
    eq: bool = True,
    order: bool = False,
    kw_only: bool = False,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Declare a value-semantics class: dataclass field syntax, compared and hashed by its fields.

    Frozen and slotted by default, as `@dataclass(frozen=True, slots=True)` is. The fields are
    read-only to type checkers; at runtime the class is not frozen, so `__init__` assigns its fields
    directly. A field set after `__init__` (in `__post_init__`) is set with `object.__setattr__`.
    """

    def wrap(c: type[T]) -> type[T]:
        _check_slots(c, slots)
        # A frozen dataclass hashes by its fields; a mutable one does not unless asked. A hash the
        # class body defines is kept, as a frozen dataclass keeps it.
        own_hash = vars(c).get("__hash__", dc.MISSING)
        defines_hash = not (own_hash is dc.MISSING or (own_hash is None and "__eq__" in vars(c)))
        return dc.dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=eq and not defines_hash,
            slots=slots,
            kw_only=kw_only,
        )(c)

    return wrap if cls is None else wrap(cls)


def _check_slots(c: type, slots: bool, /) -> None:
    # `slots=True` rebuilds the class without `__dict__`, which `cached_property` needs to cache
    # into. Left to dataclass that fails at first property access, far from the declaration.
    if slots and any(isinstance(v, functools.cached_property) for v in vars(c).values()):
        raise TypeError(
            f"{c.__name__} uses @cached_property, which needs __dict__ (pass slots=False)"
        )


def with_hint(error: E, note: str) -> E:
    """Attach a hint to `error` as a note and hand the exception back for raising.

    Returns rather than raises so the helper adds no frame between the caller and the `raise`.
    """
    error.add_note(note)
    return error


def def_full_name(func: Any, /) -> str:
    """`module.qualname` of a function or an object's type; bare for `builtins`, `__main__`."""
    try:
        module = func.__module__
        name = func.__qualname__
    except AttributeError:
        object_type = type(func)
        module = object_type.__module__
        name = object_type.__qualname__
    return f"{module}.{name}" if module and module not in ("builtins", "__main__") else name
