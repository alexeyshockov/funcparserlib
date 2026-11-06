"""The names `just mutants` runs by default exist, so that a rename cannot shrink the run unseen."""

import re
from pathlib import Path

from funcparserlib import parser

JUSTFILE = Path(__file__).parent.parent / "justfile"


def test_the_default_mutants_name_what_parser_has() -> None:
    line = next(s for s in JUSTFILE.read_text().splitlines() if s.startswith("mutants_default"))
    names = [
        cls or function
        for cls, function in re.findall(r"funcparserlib\.parser\.x(?:ǁ(\w+)ǁ|_(\w+))\*", line)
    ]
    assert sorted(names) == ["Skipping", "lit", "regex", "skipping"]
    assert [name for name in names if not hasattr(parser, name)] == []
