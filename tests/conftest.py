"""Generated code that breaks fails the test, where a user gets the interpreted parser instead."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from funcparserlib import _compile


class GeneratedCodeError(BaseException):
    """A name generated code reads has no binding.

    It derives from `BaseException` to pass the `except Exception` around generated code.
    """


def _watched(run: Callable[..., Any]) -> Callable[..., Any]:
    def watched(*args: Any) -> Any:
        try:
            return run(*args)
        except NameError as e:
            tb = e.__traceback__
            while tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
            if tb is not None and tb.tb_frame.f_code.co_filename == _compile._FILENAME:
                raise GeneratedCodeError(str(e)) from e
            raise

    return watched


@pytest.fixture(scope="session", autouse=True)
def generated_code_is_watched() -> Iterator[None]:
    functions = _compile._functions

    def watched(*args: Any, **kwargs: Any) -> Any:
        found = functions(*args, **kwargs)
        return None if found is None else (_watched(found[0]), *found[1:])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_compile, "_functions", watched)
        yield
