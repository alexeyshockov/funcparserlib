"""What the tests run: the examples and the pages that teach from them need their own Python.

The examples are written in the newest syntax the docs teach, at the version ruff's
`per-file-target-version` gives `examples/**` in `pyproject.toml`. On an older interpreter the
examples' tests are not collected, and the docs, the README and the tests marked `examples` are
skipped: their sessions import the examples or use that syntax.
"""

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent

with (ROOT / "pyproject.toml").open("rb") as f:
    _TARGET: str = tomllib.load(f)["tool"]["ruff"]["per-file-target-version"]["examples/**"]

EXAMPLES_PYTHON = (3, int(_TARGET.removeprefix("py3")))
"""The Python the examples, the docs and the README are written for."""

_OLDER = sys.version_info < EXAMPLES_PYTHON

collect_ignore = ["tests/examples"] if _OLDER else []


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip, on an older interpreter, what is written for the examples' Python.

    The docs and the README are arguments of the run (`testpaths`), which `collect_ignore` does not
    reach, and reading their sessions imports nothing, so they are collected and skipped.
    """
    if not _OLDER:
        return
    skip = pytest.mark.skip(reason=f"written for Python {'.'.join(map(str, EXAMPLES_PYTHON))}")
    for item in items:
        path = item.path.relative_to(ROOT)
        if (
            path.parts[:1] == ("docs",)
            or path == Path("README.md")
            or item.get_closest_marker("examples")
        ):
            item.add_marker(skip)
