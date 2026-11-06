"""An arithmetic calculator: `+ - * /` over numbers, with parentheses.

Goal: the smallest whole program, from text to a value, with precedence in the grammar.
"""

from collections.abc import Callable
from functools import reduce

from funcparserlib.lexer import Token, TokenSpec, make_tokenizer
from funcparserlib.parser import Parser, finished, forward_decl, many, tok

# --8<-- [start:tokenizer]
tokenize = make_tokenizer(
    [TokenSpec("space", r"\s+"), TokenSpec("number", r"\d+(\.\d*)?"), TokenSpec("op", r"[-+*/()]")],
    skip={"space"},
)


def tokens(text: str) -> list[Token]:
    """The tokens of `text`, without whitespace."""
    return list(tokenize(text))


# --8<-- [end:tokenizer]

# --8<-- [start:grammar]
type Op = Callable[[float, float], float]

OPS: dict[str, Op] = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
}


def fold(first_and_rest: tuple[float, list[tuple[str, float]]]) -> float:
    """Apply `x op y` left to right over the first operand and the (operator, operand) pairs."""
    first, rest = first_and_rest
    return reduce(lambda acc, op_y: OPS[op_y[0]](acc, op_y[1]), rest, first)


def op(chars: str) -> Parser[Token, str]:
    """Any one of the operators in `chars`."""
    parsers = [tok("op", c) for c in chars]
    return reduce(lambda p, q: p | q, parsers)


expr = forward_decl()
number = tok("number") >> float
factor = number | (-tok("op", "(") + expr + -tok("op", ")"))
term = factor + many(op("*/") + factor) >> fold
expr.define(term + many(op("+-") + term) >> fold)
document = expr + -finished
# --8<-- [end:grammar]


def calculate(text: str) -> float:
    """The value of the arithmetic expression `text`."""
    result: float = document.parse(tokens(text))
    return result


if __name__ == "__main__":
    print(calculate("2 * (3 + 4) - 10 / 4"))
