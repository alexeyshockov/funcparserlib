"""`examples/structured_fields.py` over the hand-written cases of the HTTP working group's suite.

`structured_fields/` beside this module is a copy of them from
github.com/httpwg/structured-field-tests, under the licence there.
"""

import base64
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from examples import structured_fields as sf
from funcparserlib.parser import NoParseError, ParseErrors

CASES = [
    pytest.param(case, id=f"{path.stem}: {case['name']}")
    for path in sorted((Path(__file__).parent / "structured_fields").glob("*.json"))
    for case in json.loads(path.read_text())
]


def bare(value: sf.Bare) -> object:
    """A value in the JSON of the suite."""
    match value:
        case sf.Token():
            return {"__type": "token", "value": value.value}
        case sf.Date():
            return {"__type": "date", "value": value.seconds}
        case sf.DisplayString():
            return {"__type": "displaystring", "value": value.value}
        case bytes():
            return {"__type": "binary", "value": base64.b32encode(value).decode()}
        case Decimal():
            return float(value)
        case _:
            return value


def parameters(params: sf.Parameters) -> list[list[object]]:
    return [[k, bare(v)] for k, v in params.items()]


def member(m: sf.Member) -> list[object]:
    if isinstance(m, sf.InnerList):
        return [[member(i) for i in m.items], parameters(m.parameters)]
    return [bare(m.value), parameters(m.parameters)]


def parsed(field_type: str, field: str) -> object:
    match field_type:
        case "item":
            return member(sf.parse_item(field))
        case "list":
            return [member(m) for m in sf.parse_list(field)]
        case _:
            return [[k, member(m)] for k, m in sf.parse_dictionary(field).items()]


def without_padding(value: Any) -> Any:
    """`expected` with its base32 unpadded, as some cases write it that way."""
    if isinstance(value, list):
        return [without_padding(v) for v in value]
    if isinstance(value, dict) and value.get("__type") == "binary":
        return {**value, "value": value["value"].rstrip("=")}
    return value


@pytest.mark.parametrize("case", CASES)
def test_a_field_reads_as_the_suite_says(case: dict[str, Any]) -> None:
    field = ", ".join(case["raw"])
    if case.get("must_fail"):
        with pytest.raises((NoParseError, ParseErrors)):
            parsed(case["header_type"], field)
    elif not case.get("can_fail"):
        value = parsed(case["header_type"], field)
        assert without_padding(value) == without_padding(case["expected"])


def test_the_suite_is_there() -> None:
    assert len(CASES) > 200
