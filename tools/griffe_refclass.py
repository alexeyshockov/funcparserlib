"""Teach griffe that `@refclass` and `@valueclass` are `@dataclass`."""

# Generated; propose changes in an issue rather than editing this file.
#
# `<pkg>._commons.refclass` and `<pkg>._commons.valueclass` call `dataclasses.dataclass(...)` at
# runtime, but griffe matches decorators statically by canonical path, so their classes got no
# `__init__` and no constructor signature. This appends a synthetic `dataclass(...)` carrying the
# decorator's own keywords, then hands the class to griffe's `DataclassesExtension`.
#
# The class is passed directly because the built-in runs from `on_package`, which fires before the
# modules are visited; a decorator appended during the visit would arrive after it.
#
# Docs-only: nothing here is imported by the packages it documents or by the tests.

from __future__ import annotations

from typing import Any

from griffe import Class, DataclassesExtension, Decorator, Expr, ExprCall, ExprName, Extension

# Matched by suffix, so any package's `_commons` matches; `_commons` as a directory resolves
# the same as a module.
DECORATOR_SUFFIXES = ("._commons.refclass", "._commons.valueclass")

_DATACLASS = ExprName("dataclass", parent="dataclasses")
"""`parent` as a string is what makes `canonical_path` the literal griffe's built-in matches on."""


class RefclassExtension(Extension):
    """Make griffe rebuild the `__init__` of `@refclass` and `@valueclass` classes."""

    _dataclasses = DataclassesExtension()

    def on_class_members(self, *, cls: Class, **kwargs: Any) -> None:
        for decorator in cls.decorators:
            if not decorator.callable_path.endswith(DECORATOR_SUFFIXES):
                continue
            value: Expr = _DATACLASS
            if isinstance(decorator.value, ExprCall):
                # `@refclass(frozen=True)`, `@valueclass(order=True)`: `dataclass`'s keywords.
                if _keyword(decorator.value, "init") == "False":
                    # `init=False` inherits the parent's constructor. Griffe would still synthesize
                    # an empty one and the reference would claim the class takes no arguments, so
                    # leave it alone.
                    return
                value = ExprCall(function=_DATACLASS, arguments=decorator.value.arguments)
            decorated = Decorator(value, lineno=decorator.lineno, endlineno=decorator.endlineno)
            cls.decorators.append(decorated)
            self._dataclasses.on_package(pkg=cls)  # type: ignore[arg-type]  # a class is a root
            return


def _keyword(call: ExprCall, name: str) -> str | None:
    """The value of one keyword argument, as source text.

    `None` if it was not passed.
    """
    for argument in call.arguments:
        if getattr(argument, "name", None) == name:
            return str(getattr(argument, "value", ""))
    return None
