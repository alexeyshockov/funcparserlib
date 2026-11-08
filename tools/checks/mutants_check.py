"""The names `just mutants` runs by default exist, so that a rename cannot shrink the run unseen.

The justfile's `mutants_default` names parts of `funcparserlib/parser.py` the way mutmut mangles
them: `x_<function>` for a module-level function and `xǁ<Class>ǁ` for a class's methods.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
JUSTFILE = ROOT / "justfile"
PARSER = ROOT / "funcparserlib" / "parser.py"

NAME = re.compile(r"funcparserlib\.parser\.x(?:ǁ(\w+)ǁ|_(\w+))\*")


def default_names(justfile: str) -> list[str]:
    """The functions and classes of `parser.py` that `mutants_default` names."""
    line = next(s for s in justfile.splitlines() if s.startswith("mutants_default"))
    return [cls or function for cls, function in NAME.findall(line)]


def defined(source: str) -> set[str]:
    """The module-level function and class names of a module."""
    return {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def test_the_default_mutants_name_what_parser_has() -> None:
    names = default_names(JUSTFILE.read_text(encoding="utf-8"))
    assert sorted(names) == ["Skipping", "lit", "regex", "skipping"]
    assert [n for n in names if n not in defined(PARSER.read_text(encoding="utf-8"))] == []


def test_a_renamed_function_is_found_missing() -> None:
    line = "mutants_default := \"'funcparserlib.parser.x_gone*' 'funcparserlib.parser.xǁKeptǁ*'\""
    names = default_names(line)
    assert names == ["gone", "Kept"]
    assert [n for n in names if n not in defined("class Kept: ...\n")] == ["gone"]
