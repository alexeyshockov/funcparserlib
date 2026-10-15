import json as stdlib_json
import unittest

from hypothesis import given
from hypothesis import strategies as st

from funcparserlib.lexer import LexerError
from funcparserlib.parser import NoParseError

from . import json


class JsonTest(unittest.TestCase):
    def t(self, data: str, expected: object | None = None) -> None:
        assert json.loads(data) == expected

    def test_1_array(self) -> None:
        self.t("[1]", [1])

    def test_1_object(self) -> None:
        self.t('{"foo": "bar"}', {"foo": "bar"})

    def test_bool_and_null(self) -> None:
        self.t("[null, true, false]", [None, True, False])

    def test_empty_array(self) -> None:
        self.t("[]", [])

    def test_empty_object(self) -> None:
        self.t("{}", {})

    def test_many_array(self) -> None:
        self.t("[1, 2, [3, 4, 5], 6]", [1, 2, [3, 4, 5], 6])

    def test_many_object(self) -> None:
        # noinspection SpellCheckingInspection
        self.t(
            """
            {
                "foo": 1,
                "bar":
                {
                    "baz": 2,
                    "quux": [true, false],
                    "{}": {}
                },
                "spam": "eggs"
            }
        """,
            {"foo": 1, "bar": {"baz": 2, "quux": [True, False], "{}": {}}, "spam": "eggs"},
        )

    def test_null(self) -> None:
        try:
            self.t("")
        except NoParseError:
            pass
        else:
            self.fail("must raise NoParseError")

    def test_numbers(self) -> None:
        self.t(
            """\
            [
                0, 1, -1, 14, -14, 65536,
                0.0, 3.14, -3.14, -123.456,
                6.67428e-11, -1.602176e-19, 6.67428E-11
            ]
        """,
            [
                0,
                1,
                -1,
                14,
                -14,
                65536,
                0.0,
                3.14,
                -3.14,
                -123.456,
                6.67428e-11,
                -1.602176e-19,
                6.67428e-11,
            ],
        )

    def test_strings(self) -> None:
        # noinspection SpellCheckingInspection
        self.t(
            r"""
            [
                ["", "hello", "hello world!"],
                ["привет, мир!", "λx.x"],
                ["\"", "\\", "\/", "\b", "\f", "\n", "\r", "\t"],
                ["\u0000", "\u03bb", "\uffff", "\uFFFF"],
                ["вот функция идентичности:\nλx.x\nили так:\n\u03bbx.x"]
            ]
        """,
            [
                ["", "hello", "hello world!"],
                ["привет, мир!", "λx.x"],
                ['"', "\\", "/", "\x08", "\x0c", "\n", "\r", "\t"],
                ["\u0000", "\u03bb", "\uffff", "\uffff"],
                ["вот функция идентичности:\nλx.x\nили так:\n\u03bbx.x"],
            ],
        )

    def test_toplevel_string(self) -> None:
        try:
            self.t("неправильно")
        except LexerError:
            pass
        else:
            self.fail("must raise LexerError")


# Property: the example grammar reads what the stdlib's json module writes, as json.loads does.

JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda kids: st.lists(kids, max_size=4) | st.dictionaries(st.text(), kids, max_size=4),
    max_leaves=20,
)


def same(x: object, y: object) -> bool:
    """`x == y`, with the types compared too (`True == 1` and `1 == 1.0` in Python)."""
    if type(x) is not type(y):
        return False
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(map(same, x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    return x == y


@given(JSON, st.booleans())
def test_the_grammar_reads_json_as_the_stdlib_does(value: object, ensure_ascii: bool) -> None:
    text = stdlib_json.dumps(value, ensure_ascii=ensure_ascii)
    assert same(json.loads(text), stdlib_json.loads(text))
