"""The formats a run is printed in."""

import json
from collections.abc import Sequence

from ._engine import Finding

FORMATS = ("text", "json", "github")


def render(findings: Sequence[Finding], fmt: str) -> str:
    match fmt:
        case "text":
            return "".join(f"{_location(f)}: {f.code} {f.message}\n" for f in findings)
        case "json":
            rows = [
                {"path": f.path, "line": f.line, "code": f.code, "message": f.message}
                for f in findings
            ]
            return json.dumps(rows, indent=2) + "\n"
        case "github":
            return "".join(_annotation(f) for f in findings)
        case _:
            raise ValueError(f"unknown format {fmt!r}")


def _location(f: Finding) -> str:
    return f.path if f.line is None else f"{f.path}:{f.line}"


def _annotation(f: Finding) -> str:
    """A GitHub Actions `error` workflow command, escaped as the runner parses it."""
    props = f"file={_property(f.path)}"
    if f.line is not None:
        props += f",line={f.line}"
    props += f",title={_property(f.code)}"
    return f"::error {props}::{_data(f.message)}\n"


def _data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _property(value: str) -> str:
    return _data(value).replace(":", "%3A").replace(",", "%2C")
