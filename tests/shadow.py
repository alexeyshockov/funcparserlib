"""A pytest plugin that checks every parse of a run against the compiled parser.

With `-p tests.shadow`, each `Parser.parse()`, `parse_text()` and `parse_bytes()` also runs on
`Parser.compile()`, and a different value or error fails the test that made the call. So every test
and every doc example is a test of generated code, with the inputs someone chose by hand.

A function after `>>` runs once more in such a run, so a test that counts its calls is marked with
`pytest.mark.interpreted_only`, as is one where the two differ on purpose.
"""

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from funcparserlib.parser import NoParseError, ParseErrors, Parser

Outcome = tuple[object, ...]

# A doc page whose example counts what a parse calls.
COUNTING_PAGES = {"performance.md"}

_compiled: dict[int, tuple[Parser[Any, Any], Any]] = {}
_inside = False


def _error(e: NoParseError) -> Outcome:
    return (str(e), e.expected, e.context, e.description)


def _outcome(error: Exception) -> Outcome:
    """What an exception of a parse says."""
    if isinstance(error, ParseErrors):
        return ("errors", [_error(e) for e in error.exceptions])
    if isinstance(error, NoParseError):
        return ("error", _error(error))
    return ("raised", type(error), str(error))


def _same(found: object, expected: object) -> bool:
    """Whether two values are equal, a `NaN` in them being equal to itself."""
    try:
        return found == expected or repr(found) == repr(expected)
    except ValueError:  # the `repr()` of an `int` past the limit of its digits
        return False


def _code(p: Parser[Any, Any]) -> Any:
    """The compiled parser of `p`, or `None` where it does not compile."""
    if id(p) not in _compiled:
        try:
            code = p.compile()
        except Exception:  # noqa: BLE001 — what compile() raises has tests of its own
            code = None
        _compiled[id(p)] = (p, code)
    return _compiled[id(p)][1]


def _compare(code: Any, name: str, source: Any, options: Any, value: object, error: Any) -> None:
    """Fail unless the compiled parser gives `value` or `error` for `source`."""
    found, failure = None, None
    try:
        found = getattr(code, name)(source, **options)
    except Exception as e:  # noqa: BLE001 — what a function after `>>` raises is compared too
        failure = e
    if isinstance(failure, RecursionError) or isinstance(error, RecursionError):
        return  # generated code uses less of the stack, by design
    if failure is None and error is None:
        assert _same(found, value), f"on {source!r}"
    else:
        assert failure is not None, f"compiled gave {found!r} on {source!r}, not {error!r}"
        assert error is not None, f"compiled raised {failure!r} on {source!r}, not {value!r}"
        assert _outcome(failure) == _outcome(error), f"on {source!r}"


def _shadowed(method: Callable[..., Any], name: str) -> Callable[..., Any]:
    def shadowed(self: Parser[Any, Any], source: Any, **options: Any) -> Any:
        """What the parser gives, after the compiled parser gave the same."""
        global _inside  # noqa: PLW0603 — the compiled parser calls this method to make its errors
        if _inside:
            return method(self, source, **options)
        _inside = True
        value, error = None, None
        try:
            try:
                value = method(self, source, **options)
            except Exception as e:  # noqa: BLE001 — raised again below
                error = e
            code = _code(self)
            if code is not None and not (name == "parse_text" and code._reads_tokens):
                _compare(code, name, source, options, value, error)
        finally:
            _inside = False
        if error is not None:
            raise error
        return value

    return shadowed


@pytest.fixture(autouse=True)
def shadow(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Shadow each parse of a test, unless it is marked `interpreted_only`."""
    marked = request.node.get_closest_marker("interpreted_only") is not None
    if not marked and request.node.path.name not in COUNTING_PAGES:
        for name in ("parse", "parse_text", "parse_bytes"):
            monkeypatch.setattr(Parser, name, _shadowed(getattr(Parser, name), name))
    yield
    _compiled.clear()
