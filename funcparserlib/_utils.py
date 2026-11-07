"""The helpers of funcparserlib's own modules."""


def place_of(text: str, offset: int) -> tuple[int, int]:
    """The line and the column of `offset` in `text`, from 1, lines counted by `"\\n"` alone."""
    return text.count("\n", 0, offset) + 1, offset - text.rfind("\n", 0, offset)


def line_of(text: str, line: int) -> str:
    """Line `line` of `text`, counted from 1 by `"\\n"` alone, without a `"\\r"` at its end."""
    return text.split("\n")[line - 1].rstrip("\r")
