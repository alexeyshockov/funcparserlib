from examples.calculator import calculate


def test_calculator_respects_precedence_and_parentheses() -> None:
    assert calculate("2 * (3 + 4) - 10 / 4") == 11.5


def test_calculator_folds_left_to_right() -> None:
    assert calculate("8 - 3 - 2") == 3.0
