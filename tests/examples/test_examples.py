import json
import math
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import msgpack
import pytest
from hypothesis import given
from hypothesis import strategies as st
from markdown_it.token import Token as MarkdownToken
from ruamel.yaml.error import YAMLError

from examples import (
    adr,
    changelog,
    http_headers,
    kdl,
    kdl_text,
    markdown,
    migration,
    msgpack_format,
    notes,
    partial_json,
    uri,
    yaml,
)
from examples import markdown as md
from examples.calculator import calculate
from examples.kdl import Node, Value
from examples.partial_json import MISSING
from funcparserlib.lexer import LexerError
from funcparserlib.parser import NoParseError, ParseErrors


def test_calculator_respects_precedence_and_parentheses() -> None:
    assert calculate("2 * (3 + 4) - 10 / 4") == 11.5


def test_calculator_folds_left_to_right() -> None:
    assert calculate("8 - 3 - 2") == 3.0


def test_markdown_builds_the_tree_of_every_supported_construct() -> None:
    text = (
        "# Title *here*\n\n"
        "Some **bold** and `code`, a [link *x*](http://x.y).\nnext\\\nhard\n\n"
        "- one\n- two\n  - nested\n\n"
        "3. three\n\n"
        "> quoted\n\n"
        "```py\nprint(1)\n```\n\n"
        "---\n\n"
        "    indented\n"
    )
    assert markdown.parse(text) == [
        md.Heading(1, [md.Text("Title "), md.Emphasis([md.Text("here")])]),
        md.Paragraph(
            [
                md.Text("Some "),
                md.Strong([md.Text("bold")]),
                md.Text(" and "),
                md.Code("code"),
                md.Text(", a "),
                md.Link("http://x.y", [md.Text("link "), md.Emphasis([md.Text("x")])]),
                md.Text("."),
                md.Break(hard=False),
                md.Text("next"),
                md.Break(hard=True),
                md.Text("hard"),
            ]
        ),
        md.BulletList(
            [
                [md.Paragraph([md.Text("one")])],
                [
                    md.Paragraph([md.Text("two")]),
                    md.BulletList([[md.Paragraph([md.Text("nested")])]]),
                ],
            ]
        ),
        md.OrderedList(3, [[md.Paragraph([md.Text("three")])]]),
        md.Blockquote([md.Paragraph([md.Text("quoted")])]),
        md.CodeBlock("py", "print(1)\n"),
        md.ThematicBreak(),
        md.CodeBlock("", "indented\n"),
    ]


@pytest.mark.parametrize("text", ["2 * 3 * 4", "a ** b", "an_underscored_name", "[not a link]"])
def test_markdown_leaves_text_that_only_looks_like_markup(text: str) -> None:
    assert markdown.parse(text) == [md.Paragraph([md.Text(text)])]


def test_markdown_nests_blocks_inside_quotes_and_items() -> None:
    assert markdown.parse("> - a\n>   > b") == [
        md.Blockquote(
            [
                md.BulletList(
                    [[md.Paragraph([md.Text("a")]), md.Blockquote([md.Paragraph([md.Text("b")])])]]
                )
            ]
        )
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("![alt](i.png)", "image on line 1, expected: inline content or end of input"),
        ("text\n\n<div>x</div>", "html_block on line 3, expected: block or end of input"),
        ("- a\n- b ![c](d)", "image on line 2, expected: inline content or end of input"),
    ],
)
def test_markdown_rejects_what_the_tree_does_not_cover(text: str, message: str) -> None:
    with pytest.raises(NoParseError) as e:
        markdown.parse(text)
    assert str(e.value) == f"got unexpected token: {message}"


SUPPORTED = [
    "# heading",
    "## *heading*",
    "text",
    "*em*",
    "**strong**",
    "`code`",
    "[link](http://x.y)",
    "- ",
    "1. ",
    "> ",
    "  ",
    "```py\nprint(1)\n```",
    "---",
    "    indented",
    "2 * 3",
    "\\*",
    "&amp;",
]
UNSUPPORTED = ["![image](i.png)", "<b>html</b>", "<div>block</div>"]
SEPARATORS = ["", " ", "\n", "\n\n"]


def documents(fragments: list[str]) -> st.SearchStrategy[str]:
    """Documents made of `fragments` joined by spaces and line breaks, so that blocks nest."""
    pieces = st.tuples(st.sampled_from(fragments), st.sampled_from(SEPARATORS))
    return st.lists(pieces, max_size=12).map(lambda ps: "".join(f + sep for f, sep in ps))


@given(documents(SUPPORTED))
def test_markdown_of_supported_constructs_always_builds_a_tree(text: str) -> None:
    assert isinstance(markdown.parse(text), list)


@given(documents(SUPPORTED + UNSUPPORTED))
def test_markdown_either_builds_a_tree_or_raises_no_parse_error(text: str) -> None:
    try:
        markdown.parse(text)
    except NoParseError:
        pass


def test_yaml_builds_every_supported_construct() -> None:
    text = (
        "# a comment\n"
        "block:\n"
        "  - one\n"
        "  - key: value\n"
        "    other: 'it''s'\n"
        "  - - nested\n"
        "indentless:\n"
        "- x\n"
        "flow: {a: [1, 2, {b: c}], d, }\n"
        "pair: [k: v, w]\n"
        "literal: |\n"
        "  one\n"
        "  two\n"
        "folded: >-\n"
        "  one\n"
        "  two\n"
        "multi line: plain\n"
        "  text\n"
        "empty:\n"
    )
    assert yaml.parse(text) == {
        "block": ["one", {"key": "value", "other": "it's"}, ["nested"]],
        "indentless": ["x"],
        "flow": {"a": [1, 2, {"b": "c"}], "d": None},
        "pair": [{"k": "v"}, "w"],
        "literal": "one\ntwo\n",
        "folded": "one two",
        "multi line": "plain text",
        "empty": None,
    }


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("~", None),
        ("", None),
        ("true", True),
        ("no", "no"),
        ("010", 10),
        ("0o17", 15),
        ("0x1F", 31),
        ("1:20", "1:20"),
        ("1.5e3", 1500.0),
        ("-.inf", float("-inf")),
        ("'1'", "1"),
        ('"true"', "true"),
    ],
)
def test_yaml_resolves_only_plain_scalars_by_the_core_schema(text: str, value: object) -> None:
    assert yaml.parse(text) == value


def test_yaml_alias_shares_the_object_of_its_anchor() -> None:
    result = yaml.parse("a: &x {k: v}\nb: *x\n")
    assert isinstance(result, dict)
    assert result["a"] is result["b"]


def test_yaml_alias_refers_to_the_last_anchor_of_its_name() -> None:
    assert yaml.parse("- &a 1\n- *a\n- &a 2\n- *a\n") == [1, 1, 2, 2]


@pytest.mark.parametrize("text", ["a: &x\n  self: *x\n", "a: &x {self: *x}\n"])
def test_yaml_alias_inside_its_anchored_mapping_builds_a_cycle(text: str) -> None:
    result = yaml.parse(text)
    assert isinstance(result, dict)
    assert result["a"]["self"] is result["a"]


def test_yaml_alias_inside_its_anchored_sequence_builds_a_cycle() -> None:
    result = yaml.parse("&x [*x]")
    assert isinstance(result, list)
    assert result[0] is result


def test_yaml_anchored_sequence_at_the_indentation_of_its_mapping() -> None:
    assert yaml.parse("a: &x\n- 1\nb: *x\n") == {"a": [1], "b": [1]}


def test_yaml_merge_key_fills_in_what_the_mapping_lacks() -> None:
    text = "a: &a {x: 1, y: 2}\nb: &b {y: 3, z: 4}\nc:\n  <<: [*a, *b]\n  x: 0\n"
    result = yaml.parse(text)
    assert isinstance(result, dict)
    assert result["c"] == {"x": 0, "y": 2, "z": 4}


def test_yaml_quoted_merge_key_is_an_ordinary_key() -> None:
    assert yaml.parse('"<<": 1\n') == {"<<": 1}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a: !Ref x\n", "1,4-1,4: got unexpected token: <tag>"),
        ("? x\n: 1\n", "1,1-1,1: got unexpected token: ?"),
        ("[a, b]: 1\n", "1,1-1,1: got unexpected token: ["),
        ("%YAML 1.2\n---\na\n", "1,1-1,1: got unexpected token: <directive>"),
        ("---\na\n---\nb\n", "3,1-3,1: got unexpected token: <document start>"),
    ],
)
def test_yaml_rejects_what_the_subset_does_not_cover(text: str, message: str) -> None:
    with pytest.raises(NoParseError) as e:
        yaml.parse(text)
    assert str(e.value).startswith(message)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a: *nope\n", "undefined alias 'nope' at (1, 4)"),
        ("a: 1\na: 2\n", "repeated key 'a' at (2, 1)"),
        ("1: a\ntrue: b\n", "repeated key True at (2, 1)"),
        ("<<: {a: 1}\n<<: {b: 2}\n", "repeated key '<<' at (2, 1)"),
        ("<<: 1\n", "merge value is not a mapping at (1, 1)"),
    ],
)
def test_yaml_rejects_a_document_without_a_meaning(text: str, message: str) -> None:
    with pytest.raises(yaml.ResolveError) as e:
        yaml.parse(text)
    assert str(e.value) == message


def test_yaml_too_deep_a_document_raises_recursion_error() -> None:
    with pytest.raises(RecursionError):
        yaml.parse("[" * 5000 + "]" * 5000)


YAML_FRAGMENTS = ["a", "1", "'q'", "- ", "k: ", "[", "]", "{", "}", ", ", "&x ", "*x", "<<: ", "|"]
YAML_SEPARATORS = ["", " ", "\n", "\n  "]


@given(
    st.lists(
        st.tuples(st.sampled_from(YAML_FRAGMENTS), st.sampled_from(YAML_SEPARATORS)), max_size=12
    ).map(lambda ps: "".join(f + sep for f, sep in ps))
)
def test_yaml_either_gives_a_value_or_raises_a_documented_error(text: str) -> None:
    try:
        yaml.parse(text)
    except (NoParseError, yaml.ResolveError, YAMLError):
        pass


def test_kdl_builds_the_tree_of_every_node_field() -> None:
    text = """
    // a comment
    (app)server "web" 8080 host=localhost {
        route (path)"/api" methods=#null; log
    }
    """
    assert kdl.parse(text) == [
        Node(
            "server",
            [Value("web"), Value(8080)],
            {"host": Value("localhost")},
            [Node("route", [Value("/api", "path")], {"methods": Value(None)}), Node("log")],
            "app",
        )
    ]


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("", []),
        ("a", ["a"]),
        ("a\n\n\n  b\n", ["a", "b"]),
        ("a;b;\nc", ["a", "b", "c"]),
        ("a // comment\nb", ["a", "b"]),
        ("a /* comment */\nb", ["a", "b"]),
        ("a\n\\\n/-\nb\nc", ["a", "c"]),
        ("\ufeffa", ["a"]),
        ("a\r\nb\x0bc\u2028d", ["a", "b", "c", "d"]),
        ("/- kdl-version 2\na", ["a"]),
    ],
)
def test_kdl_finds_the_nodes_between_the_terminators(text: str, names: list[str]) -> None:
    assert [n.name for n in kdl.parse(text)] == names


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/-a 1", []),
        ("/- a {b}\nc", [Node("c")]),
        ("a /-1 2", [Node("a", [Value(2)])]),
        ("a/-1", [Node("a")]),
        ("a /-k=1 l=2", [Node("a", props={"l": Value(2)})]),
        ("a /-{b}", [Node("a")]),
        ("a /-{b} {c} /-{d}", [Node("a", children=[Node("c")])]),
        ("a /-\n  // comment\n  1", [Node("a")]),
    ],
)
def test_kdl_slashdash_drops_what_follows_it(text: str, expected: list[Node]) -> None:
    assert kdl.parse(text) == expected


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("bare-word", "bare-word"),
        ("--flag", "--flag"),
        (".md", ".md"),
        ('"tab\\tu\\u{e9}\\s\\""', 'tab\tu\u00e9 "'),
        ('"joined\\   \n   line"', "joinedline"),
        ('#"C:\\dir\\"#', "C:\\dir\\"),
        ('##"a"#b"##', 'a"#b'),
        ('"""\n    one\n      two\n\n    """', "one\n  two\n"),
        ('"""\n"""', ""),
        ('"""\r\n  a\r\n  b\r\n  """', "a\nb"),
        ('"""\n  a\\\n b\n  """', "ab"),
        ('#"""\n  \\n\n  """#', "\\n"),
        ("0xFF", 255),
        ("-0o17", -15),
        ("0b1_01", 5),
        ("1_000", 1000),
        ("+7", 7),
        ("1.5e2", Decimal("150")),
        ("1.23E+1000", Decimal("1.23E+1000")),
        ("#true", True),
        ("#false", False),
        ("#null", None),
        ("#-inf", Decimal("-Infinity")),
    ],
)
def test_kdl_gives_each_literal_its_value(text: str, value: kdl.Scalar) -> None:
    [node] = kdl.parse(f"a {text}")
    assert node.args == [Value(value)]


def test_kdl_reads_an_integer_past_the_digit_limit_of_int() -> None:
    [node] = kdl.parse("a " + "1" * 5000)
    assert node.args == [Value((10**5000 - 1) // 9)]


@pytest.mark.parametrize("text", ["a k=1", "a k =1", "a k= 1", "a k = 1"])
def test_kdl_allows_whitespace_around_the_equals_sign(text: str) -> None:
    assert kdl.parse(text)[0].props == {"k": Value(1)}


def test_kdl_allows_whitespace_before_an_annotated_property_value() -> None:
    assert kdl.parse("a k= (t)1")[0].props == {"k": Value(1, "t")}


def test_kdl_keeps_the_rightmost_of_a_repeated_property() -> None:
    assert kdl.parse("a k=1 k=2")[0].props == {"k": Value(2)}


@pytest.mark.parametrize(
    "text",
    [
        'a "b"c',
        'a 1"b"',
        "a{}b",
        "a {} 1",
        "a /-{} 1",
        "a{}{}",
        "a;;b",
        "a {",
        "a }",
        "a=1",
        "a 1=2",
        "true",
        "a null",
        "(t)(u)a",
        '"""one line"""',
        "a /- /- 1",
    ],
)
def test_kdl_grammar_rejects_tokens_out_of_place(text: str) -> None:
    with pytest.raises(NoParseError):
        kdl.parse(text)


@pytest.mark.parametrize(
    "text",
    [
        "a 1.0v2",
        "a 0x",
        "a .5",
        "a #truthy",
        'a #"unbalanced""',
        'a #"""one line"""#',
        'a "line\nbreak"',
        "a /* unterminated",
        'a "\x01"',
    ],
)
def test_kdl_tokenizer_rejects_what_is_no_token(text: str) -> None:
    with pytest.raises(LexerError):
        kdl.parse(text)


def test_kdl_comments_nest_up_to_the_level_of_the_regex() -> None:
    def nested(levels: int) -> str:
        return "a " + "/*" * levels + "*/" * levels

    assert kdl.parse(nested(9)) == [Node("a")]
    with pytest.raises(LexerError):
        kdl.parse(nested(10))


def test_kdl_reports_every_bad_string_of_a_document() -> None:
    text = 'a "\\q" "\\u{D800}"\nb """\n    x\n  y\n    """\n'
    with pytest.raises(ParseErrors) as e:
        kdl.parse(text)
    assert [str(error) for error in e.value.exceptions] == [
        f"1,3-1,6: {kdl.BAD_ESCAPE}",
        f"1,8-1,17: {kdl.BAD_ESCAPE}",
        f"2,3-5,7: {kdl.BAD_INDENT}",
    ]


def test_kdl_too_deep_a_document_raises_recursion_error() -> None:
    with pytest.raises(RecursionError):
        kdl.parse("a {" * 5000 + "}" * 5000)


KDL_DOCUMENTS = [
    'a 1 "x" k=#true {\n  b; c (t)0x1F\n}\n',
    "/- kdl-version 2\na; /- b; c\n",
    'a """\n    one\n      two\n    """\nb #"raw"# \\\n  1.5e3 /* c */ k=#null\n',
    'a "\\u{1F600}\\n" ""',
]


@pytest.mark.parametrize("text", KDL_DOCUMENTS)
def test_kdl_text_gives_the_tree_of_the_tokenizer(text: str) -> None:
    assert kdl_text.parse(text) == kdl.parse(text)


def test_kdl_text_comments_nest_to_any_depth() -> None:
    assert kdl_text.parse("a " + "/*" * 40 + "*/" * 40) == [Node("a")]


def test_kdl_text_reads_a_multiline_string_the_quoted_alternative_starts() -> None:
    assert kdl_text.parse('a """\n  x\n  """')[0].args == [Value("x")]


def test_kdl_text_error_of_an_unterminated_string_says_where() -> None:
    text = 'a "abc\nb'
    with pytest.raises(NoParseError) as e:
        kdl_text.parse(text)
    assert str(e.value) == (
        "1,7-1,7: got unexpected token: '\\n', expected: text or escape or '\"' in string"
    )
    assert e.value.render(text) == f'{e.value}\n1 | a "abc\n          ^'


def test_kdl_text_reports_every_bad_string_of_a_document() -> None:
    with pytest.raises(ParseErrors) as e:
        kdl_text.parse('a "\\q" "\\u{D800}"')
    assert [str(error) for error in e.value.exceptions] == [
        f"1,3-1,6: {kdl.BAD_ESCAPE}",
        f"1,8-1,17: {kdl.BAD_ESCAPE}",
    ]


KDL_FRAGMENTS = ["a", "1", '"q"', "/-", "k=", "(t)", "{", "}", ";", "\\", "#true", '#"r"#', '"""']
KDL_SEPARATORS = ["", " ", "\n", "\n  "]


@given(
    st.lists(
        st.tuples(st.sampled_from(KDL_FRAGMENTS), st.sampled_from(KDL_SEPARATORS)), max_size=12
    ).map(lambda ps: "".join(f + sep for f, sep in ps))
)
def test_kdl_either_gives_nodes_or_raises_a_documented_error(text: str) -> None:
    try:
        kdl.parse(text)
    except (LexerError, NoParseError, ParseErrors, InvalidOperation):
        pass


def own(value: Any) -> object:
    """`value` with the timestamps of `msgpack` as the example's."""
    if isinstance(value, msgpack.Timestamp):
        return msgpack_format.Timestamp(value.seconds, value.nanoseconds)
    if isinstance(value, list):
        return [own(item) for item in value]
    if isinstance(value, dict):
        return {key: own(item) for key, item in value.items()}
    return value


def unpack(data: bytes) -> object:
    return own(msgpack.unpackb(data, raw=False, strict_map_key=False, timestamp=0))


def assert_agrees(value: object, **options: object) -> None:
    data = msgpack.packb(value, **options)
    expected = unpack(data)
    assert msgpack_format.parse(data) == expected
    assert msgpack_format.document.parse_bytes(data) == expected


def test_msgpack_reads_what_unpackb_reads_of_every_type() -> None:
    assert_agrees(None)
    assert_agrees(True)
    assert_agrees(False)
    assert_agrees([1, "a", b"b", 2.5, None, {"k": [True]}, msgpack.ExtType(3, b"xyz")])


@pytest.mark.parametrize(
    "n",
    [0, 1, 127, 128, 255, 256, 65535, 65536, 2**32 - 1, 2**32, 2**64 - 1]
    + [-1, -32, -33, -128, -129, -32768, -32769, -(2**31), -(2**31) - 1, -(2**63)],
)
def test_msgpack_reads_an_integer_at_the_edge_of_each_width(n: int) -> None:
    assert_agrees(n)


def test_msgpack_reads_the_marker_of_each_width_of_an_integer() -> None:
    markers = {n: msgpack.packb(n)[0] for n in [0, 128, 256, 65536, 2**32, -1, -33, -129]}
    assert sorted(markers.values()) == sorted([0x00, 0xCC, 0xCD, 0xCE, 0xCF, 0xFF, 0xD0, 0xD1])


@pytest.mark.parametrize("x", [0.0, -0.0, 1.5, 1e300, -1e-300, float("inf"), float("-inf")])
def test_msgpack_reads_a_float_of_either_width(x: float) -> None:
    assert_agrees(x)
    if abs(x) < 1e30:
        assert_agrees(x, use_single_float=True)


def test_msgpack_reads_a_nan() -> None:
    parsed = msgpack_format.parse(msgpack.packb(float("nan")))
    assert isinstance(parsed, float)
    assert math.isnan(parsed)


@pytest.mark.parametrize("size", [0, 1, 31, 32, 255, 256, 65535, 65536])
def test_msgpack_reads_a_string_at_the_edge_of_each_header(size: int) -> None:
    assert_agrees("é" * (size // 2) + "x" * (size % 2))
    assert_agrees("x" * size)


@pytest.mark.parametrize("size", [0, 1, 255, 256, 65535, 65536])
def test_msgpack_reads_binary_at_the_edge_of_each_header(size: int) -> None:
    assert_agrees(bytes(range(256)) * (size // 256) + bytes(range(size % 256)))


@pytest.mark.parametrize("size", [0, 1, 15, 16, 65535, 65536])
def test_msgpack_reads_an_array_at_the_edge_of_each_header(size: int) -> None:
    assert_agrees(list(range(size)))


@pytest.mark.parametrize("size", [0, 1, 15, 16, 65535, 65536])
def test_msgpack_reads_a_map_at_the_edge_of_each_header(size: int) -> None:
    assert_agrees({i: str(i) for i in range(size)})


def ext_bytes(code: int, data: bytes) -> bytes:
    """An extension in the shortest header, written out as `packb()` writes only codes from 0."""
    size = len(data)
    if size in (1, 2, 4, 8, 16):
        header = bytes([{1: 0xD4, 2: 0xD5, 4: 0xD6, 8: 0xD7, 16: 0xD8}[size]])
    elif size < 2**8:
        header = bytes([0xC7, size])
    elif size < 2**16:
        header = b"\xc8" + size.to_bytes(2, "big")
    else:
        header = b"\xc9" + size.to_bytes(4, "big")
    return header + code.to_bytes(1, "big", signed=True) + data


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 8, 16, 17, 255, 256, 65535, 65536])
@pytest.mark.parametrize("code", [0, 5, 127])
def test_msgpack_reads_an_extension_of_each_header(size: int, code: int) -> None:
    data = ext_bytes(code, b"\xab" * size)
    assert data == msgpack.packb(msgpack.ExtType(code, b"\xab" * size))
    assert msgpack_format.parse(data) == unpack(data) == msgpack_format.Ext(code, b"\xab" * size)


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 255, 65536])
@pytest.mark.parametrize("code", [-128, -2])
def test_msgpack_reads_an_extension_of_a_reserved_type_which_unpackb_rejects(
    size: int, code: int
) -> None:
    data = ext_bytes(code, b"\xab" * size)
    assert msgpack_format.parse(data) == msgpack_format.Ext(code, b"\xab" * size)
    with pytest.raises(ValueError, match="code must be 0~127"):
        unpack(data)


@pytest.mark.parametrize(
    ("seconds", "nanoseconds"),
    [(0, 0), (1, 0), (2**32 - 1, 0), (2**32, 0), (0, 1), (2**34 - 1, 999_999_999)]
    + [(2**34, 0), (2**34, 1), (-1, 0), (-(2**63), 0), (2**63 - 1, 999_999_999)],
)
def test_msgpack_reads_a_timestamp_of_each_layout(seconds: int, nanoseconds: int) -> None:
    stamp = msgpack.Timestamp(seconds, nanoseconds)
    assert_agrees(stamp)
    parsed = msgpack_format.parse(msgpack.packb(stamp))
    assert isinstance(parsed, msgpack_format.Timestamp)
    assert parsed == (seconds, nanoseconds)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 5, 9, 16])
def test_msgpack_rejects_a_timestamp_of_another_length_as_unpackb_does(size: int) -> None:
    data = ext_bytes(-1, bytes(size))
    with pytest.raises(ValueError, match="invalid timestamp data"):
        msgpack_format.parse(data)
    with pytest.raises(ValueError, match="invalid timestamp data"):
        unpack(data)


def test_msgpack_rejects_nanoseconds_of_a_second_or_more() -> None:
    data = ext_bytes(-1, (10**9 << 34).to_bytes(8, "big"))
    with pytest.raises(ValueError, match="nanoseconds"):
        msgpack_format.parse(data)
    with pytest.raises(ValueError, match="nanoseconds"):
        unpack(data)


def test_msgpack_reads_nested_arrays_and_maps() -> None:
    nested = [[[[]], {"a": {"b": [1, {"c": None}]}}], {1: [2, [3, [4]]], b"k": {}}]
    assert_agrees(nested)


def test_msgpack_reads_the_keys_a_map_may_have() -> None:
    assert_agrees({1: "int", "s": "str", b"b": "bytes", 2.5: "float", None: "nil"})


def test_msgpack_reads_a_repeated_key_as_unpackb_does() -> None:
    data = bytes([0x82, 0x01, 0x0A, 0x01, 0x0B])
    assert msgpack_format.parse(data) == unpack(data) == {1: 11}


MSGPACK_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(2**63), 2**64 - 1),
    st.floats(allow_nan=False),
    st.text(max_size=70),
    st.binary(max_size=70),
    st.builds(msgpack.ExtType, st.integers(0, 127), st.binary(max_size=70)),
    st.builds(msgpack.Timestamp, st.integers(-(2**63), 2**63 - 1), st.integers(0, 999_999_999)),
)
MSGPACK_VALUES = st.recursive(
    MSGPACK_SCALARS,
    lambda inner: st.one_of(
        st.lists(inner, max_size=20),
        st.dictionaries(
            st.one_of(st.integers(-(2**63), 2**64 - 1), st.text(max_size=10)), inner, max_size=20
        ),
    ),
    max_leaves=60,
)


@given(MSGPACK_VALUES)
def test_msgpack_agrees_with_unpackb_on_what_packb_makes(value: object) -> None:
    assert_agrees(value)


def failure(data: bytes) -> str:
    with pytest.raises(NoParseError) as e:
        msgpack_format.parse(data)
    return str(e.value)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "got unexpected end of input at offset 0x0, expected: MessagePack value"),
        (b"\xc1", "got unexpected token: 0xc1 at offset 0x0, expected: MessagePack value"),
        (b"\x92\x01\xc1", "got unexpected token: 0xc1 at offset 0x2, expected: MessagePack value"),
        (b"\x92\x01", "got unexpected end of input at offset 0x2, expected: MessagePack value"),
        (b"\x01\x02", "got unexpected token: 0x02 at offset 0x1, expected: end of input"),
        (b"\xa5ab", "length 5 runs past the end: 0x61 at offset 0x1"),
        (b"\xc7\x05\x01ab", "length 5 runs past the end: 0x61 at offset 0x3"),
        (b"\xcd\x01", "not enough bytes: 0x01 at offset 0x1, expected: uint16"),
        (b"\xd9", "got unexpected end of input at offset 0x1, expected: uint8"),
    ],
)
def test_msgpack_errors_name_the_offset_and_what_was_expected(data: bytes, message: str) -> None:
    assert failure(data) == message
    with pytest.raises(ValueError, match=r"^$|Unpack failed|extra data"):
        unpack(data)


def test_msgpack_errors_are_the_same_interpreted_and_compiled() -> None:
    data = b"\x93\x01\x81\xa1k\xc1\x03"
    with pytest.raises(NoParseError) as interpreted:
        msgpack_format.document.parse_bytes(data)
    assert str(interpreted.value) == failure(data)
    assert failure(data) == "got unexpected token: 0xc1 at offset 0x5, expected: MessagePack value"


def test_msgpack_raises_what_unpackb_raises_for_a_string_and_a_key() -> None:
    for data, error in [(b"\xa2\xff\xfe", UnicodeDecodeError), (b"\x81\x90\x01", TypeError)]:
        with pytest.raises(error):
            msgpack_format.parse(data)
        with pytest.raises(error):
            unpack(data)


ROOT = Path(__file__).parents[2]
RECORD = (
    "# 0001 — A title\n\n"
    "- **Status:** Accepted\n"
    "- **Date:** 2026-02-20\n\n"
    "## Context\n\na\n\n"
    "## Decision\n\nb\n\n"
    "## Consequences\n\nc\n"
)


def test_changelog_reads_releases_with_either_dash_and_a_linked_version() -> None:
    releases = changelog.parse(
        "# Changelog\n\nIntro.\n\n"
        "## [Unreleased]\n\nNotes.\n\n### Added\n\n- one\n\n"
        "## [1.0.0] - 2017-06-20\n\n### Fixed\n\n- two\n\n### Security\n\n- three\n\n"
        "## [0.1] — 2016-01-01 [YANKED]\n\n"
        "[1.0.0]: https://example.com/1.0.0\n"
    )
    v100 = changelog.Version((1, 0, 0), None, None, None)
    v01 = changelog.Version((0, 1), None, None, None)
    assert [r.title for r in releases] == [
        changelog.Title(None, None, yanked=False),
        changelog.Title(v100, date(2017, 6, 20), yanked=False),
        changelog.Title(v01, date(2016, 1, 1), yanked=True),
    ]
    assert releases[0].notes == "Notes."
    assert [c.kind for c in releases[1].changes] == ["Fixed", "Security"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.2", changelog.Version((1, 2), None, None, None)),
        ("2.0.0b1", changelog.Version((2, 0, 0), ("b", 1), None, None)),
        ("1.2rc3.post4.dev5", changelog.Version((1, 2), ("rc", 3), 4, 5)),
    ],
)
def test_changelog_reads_a_version(text: str, expected: changelog.Version) -> None:
    assert changelog.version.parse_text(text) == expected


def test_changelog_orders_versions_as_pep_440_does() -> None:
    ordered = ["1.0.dev1", "1.0a1", "1.0b2", "1.0rc1", "1.0", "1.0.post1", "1.1"]
    orders = [changelog.version.parse_text(v).order for v in ordered]
    assert orders == sorted(orders)
    assert len({str(o) for o in orders}) == len(orders)
    assert changelog.version.parse_text("1.0").order == changelog.version.parse_text("1.0.0").order


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("## 1.0.0 - 2020-01-01", "`[version] - date`"),
        ("## [one] - 2020-01-01", "`[version] - date`"),
        ("## [1.0.0]", "`[version] - date`"),
        ("## [1.0.0] - 2020-13-01", "`[version] - date`"),
        ("## [1.0.0] - 2020-1-1", "`[version] - date`"),
        ("## [Unreleased] - 2020-01-01", "`[version] - date`"),
        ("## [1.0.0] - 2020-01-01\n\n### Updated", "Added, Changed"),
    ],
)
def test_changelog_rejects_what_is_not_a_release(section: str, expected: str) -> None:
    with pytest.raises(ParseErrors) as e:
        changelog.parse(f"# Changelog\n\n{section}\n")
    [error] = e.value.exceptions
    assert expected in error.msg


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ("## [1.0.0] - 2020-01-01\n\n### Added\n\n### Added", "Added appears twice in a release"),
        ("## [1.0.0] - 2020-01-01\n\n## [2.0.0] - 2021-01-01", "2.0.0 comes after 1.0.0"),
        ("## [2.0.0] - 2020-01-01\n\n## [1.0.0] - 2021-01-01", "1.0.0 has a later date than 2.0.0"),
        ("## [1.0.0] - 2020-01-01\n\n## [Unreleased]", "Unreleased is not the first release"),
        ("## [1.0.0] - 2020-01-01\n\n## [1.0.0] - 2020-01-01", "1.0.0 comes after 1.0.0"),
        ("## [1.0] - 2020-01-01\n\n## [1.0.0] - 2020-01-01", "1.0.0 comes after 1.0"),
        ("## [1.0b1] - 2020-01-01\n\n## [1.0] - 2019-01-01", "1.0 comes after 1.0b1"),
    ],
)
def test_changelog_reports_a_broken_rule(sections: str, message: str) -> None:
    with pytest.raises(ParseErrors) as e:
        changelog.parse(f"# Changelog\n\n{sections}\n")
    [error] = e.value.exceptions
    assert error.msg == message


def test_changelog_reports_every_error_of_one_release() -> None:
    text = "# Changelog\n\n## [one] - 2020-01-01\n\n### Updated\n\n### Removd\n"
    with pytest.raises(ParseErrors) as e:
        changelog.parse(text)
    assert [error.msg.split(":")[0] for error in e.value.exceptions] == [
        "3,1-3,1",
        "5,1-5,1",
        "7,1-7,1",
    ]


def test_changelog_orders_a_pre_release_after_its_final_version() -> None:
    text = "# Changelog\n\n## [2.0.0] - 2021-01-01\n\n## [2.0.0b1] - 2020-01-01\n"
    assert len(changelog.parse(text)) == 2


def test_changelog_reads_a_linked_version_as_it_is_written() -> None:
    text = "# Changelog\n\n## [1.0] - 2020-01-01\n\n[1.0]: https://example.com\n"
    [release] = changelog.parse(text)
    assert release.title.date == date(2020, 1, 1)


def test_changelog_shows_the_heading_it_rejects() -> None:
    text = "# Changelog\n\n## [1.0] - 2020-13-01\n"
    with pytest.raises(ParseErrors) as e:
        changelog.parse(text)
    [error] = e.value.exceptions
    assert error.msg.startswith("3,1-3,1: got unexpected token: '[1.0] - 2020-13-01', expected:")


def test_changelog_allows_two_releases_on_one_day() -> None:
    text = "# Changelog\n\n## [1.0.1] - 2020-01-01\n\n## [1.0.0] - 2020-01-01\n"
    assert len(changelog.parse(text)) == 2


def test_changelog_takes_whatever_markdown_sits_under_a_heading() -> None:
    text = (
        "# Changelog\n\n<!-- intro -->\n\n### About\n\n"
        "## [1.0.0] - 2020-01-01\n\n<!-- ## notes -->\n\n![logo](l.png) <br>\n\n"
        "### Added\n\n#### Parser\n\n```\n## not a heading\n```\n\n| a |"
    )
    [release] = changelog.parse(text)
    assert release.notes == "<!-- ## notes -->\n![logo](l.png) <br>"
    assert release.changes[0].text == "Parser\n## not a heading\n| a |"


def test_changelog_reports_every_broken_release_where_it_breaks() -> None:
    text = (
        "# Changelog\n\n## [2.0.0] - 2020-13-01\n\n"
        "## [1.0.0] - 2019-01-01\n\n### Updated\n\n"
        "## [0.1] - 2018-13-01\n"
    )
    with pytest.raises(ParseErrors) as e:
        changelog.parse(text)
    assert [error.msg.split(":")[0] for error in e.value.exceptions] == [
        "3,1-3,1",
        "7,1-7,1",
        "9,1-9,1",
    ]


def test_changelog_of_this_repository_follows_the_rules() -> None:
    releases = changelog.parse((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert releases[0].title.version is None
    last = releases[-1].title.version
    assert last is not None
    assert last.release == (0, 1)


def test_adr_reads_a_record() -> None:
    record = adr.parse(RECORD)
    assert (record.number, record.title) == (1, "A title")
    assert record.status == adr.Status("Accepted")
    assert record.date == date(2026, 2, 20)
    assert record.consequences == "c"


@pytest.mark.parametrize("heading", ["# 0001 — A title", "# 1. A title", "# ADR 1: A title"])
def test_adr_reads_each_form_of_title(heading: str) -> None:
    record = adr.parse(RECORD.replace("# 0001 — A title", heading))
    assert (record.number, record.title) == (1, "A title")


def test_adr_reads_the_details_as_adr_tools_writes_them() -> None:
    details = "- **Status:** Accepted\n- **Date:** 2026-02-20\n"
    record = adr.parse(RECORD.replace(details, "Date: 2026-02-20\n\n## Status\n\nProposed\n"))
    assert (record.status, record.date) == (adr.Status("Proposed"), date(2026, 2, 20))


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Proposed", adr.Status("Proposed")),
        ("Deprecated", adr.Status("Deprecated")),
        ("Superseded by ADR 0002", adr.Status("Superseded", 2)),
        ("Superseded by [2. A title](0002-a-title.md)", adr.Status("Superseded", 2)),
    ],
)
def test_adr_reads_each_status(status: str, expected: adr.Status) -> None:
    assert adr.parse(RECORD.replace("Accepted", status)).status == expected


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("# 0001 — A title", "# A title", "`NNNN — Title`"),
        ("Accepted", "accepted", "**Status:** Proposed"),
        ("Accepted", "Superseded by 0002", "**Status:** Proposed"),
        ("2026-02-20", "2026-02-30", "**Date:**"),
        ("**Status:**", "**State:**", "**Status:** Proposed"),
        ("**Date:**", "**On:**", "**Date:**"),
    ],
)
def test_adr_rejects_what_is_not_a_record(old: str, new: str, expected: str) -> None:
    with pytest.raises(ParseErrors) as e:
        adr.parse(RECORD.replace(old, new))
    [error] = e.value.exceptions
    assert expected in error.msg


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("## Decision\n\nb\n\n", "", "## Decision"),
        ("## Consequences", "## Results", "## Consequences"),
    ],
)
def test_adr_requires_every_section(old: str, new: str, expected: str) -> None:
    with pytest.raises(NoParseError) as e:
        adr.parse(RECORD.replace(old, new))
    assert expected in e.value.expected


def test_adr_reports_every_field_of_a_record() -> None:
    broken = RECORD.replace("0001 — ", "").replace("Accepted", "Done").replace("20\n", "30\n")
    with pytest.raises(ParseErrors) as e:
        adr.parse(broken.replace("2026-02-30", "2026-02-31"))
    assert [error.msg.split(":")[0] for error in e.value.exceptions] == [
        "1,1-1,1",
        "3,1-3,1",
        "4,1-4,1",
    ]


def test_adr_takes_whatever_markdown_sits_in_a_section() -> None:
    body = "### Options\n\n<!-- ## c -->\n\n```\n## Consequences\n```\n\n![d](d.png)"
    decision = adr.parse(RECORD.replace("\nb\n", f"\n{body}\n")).decision
    assert decision == "Options\n<!-- ## c -->\n## Consequences\n![d](d.png)"


def test_adr_finds_what_is_wrong_across_records(tmp_path: Path) -> None:
    (tmp_path / "0001-a.md").write_text(RECORD.replace("Accepted", "Superseded by ADR 0006"))
    (tmp_path / "0003-b.md").write_text(RECORD.replace("0001", "0005"))
    assert adr.check(tmp_path) == [
        "the numbers are not 1 to 2 without gaps: [1, 5]",
        "0001-a.md: superseded by ADR 0006, which does not exist",
        "0003-b.md: the title says 0005",
    ]


def test_adr_finds_a_record_superseded_by_itself(tmp_path: Path) -> None:
    (tmp_path / "0001-a.md").write_text(RECORD.replace("Accepted", "Superseded by ADR 0001"))
    assert adr.check(tmp_path) == ["0001-a.md: superseded by itself"]


def test_adr_reports_a_directory_without_records(tmp_path: Path) -> None:
    assert adr.check(tmp_path) == [f"no records in {tmp_path}"]
    assert adr.check(tmp_path / "missing") == [f"no records in {tmp_path / 'missing'}"]


def test_adr_reports_a_broken_record_before_the_rules_across_records(tmp_path: Path) -> None:
    (tmp_path / "0002-a.md").write_text(RECORD.replace("Accepted", "Done"))
    [problem] = adr.check(tmp_path)
    assert problem.startswith("0002-a.md:\n3,1-3,1: got unexpected token: '**Status:** Done'")


def test_adr_records_of_this_repository_follow_the_rules() -> None:
    assert adr.check(ROOT / "docs" / "adr") == []


@pytest.mark.parametrize(
    "text",
    [
        "ftp://ftp.is.co.za/rfc/rfc1808.txt",
        "ldap://[2001:db8::7]/c=GB?objectClass?one",
        "mailto:John.Doe@example.com",
        "news:comp.infosystems.www.servers.unix",
        "tel:+1-816-555-1212",
        "telnet://192.0.2.16:80/",
        "urn:oasis:names:specification:docbook:dtd:xml:4.1.2",
        "g;x?y#s",
        "../../g",
        "//g",
        "?y",
        "",
    ],
)
def test_uri_reads_the_examples_of_the_rfc(text: str) -> None:
    uri.parse(text)


@pytest.mark.parametrize(
    ("text", "kind", "host"),
    [
        ("http://[::1]/", "ipv6", "::1"),
        ("http://[v1.x]/", "ipvfuture", "v1.x"),
        ("http://1.2.3.4/", "ipv4", "1.2.3.4"),
        ("http://1.2.3.4.5/", "name", "1.2.3.4.5"),
        ("http://256.2.3.4/", "name", "256.2.3.4"),
        ("file:///etc/hosts", "name", ""),
    ],
)
def test_uri_says_what_kind_of_host_it_names(text: str, kind: str, host: str) -> None:
    authority = uri.parse(text).authority
    assert authority is not None
    assert (authority.kind, authority.host) == (kind, host)


def test_uri_tells_a_userinfo_from_a_host() -> None:
    assert uri.parse("http://u:p@h:80/").authority == uri.Authority("u:p", "h", "name", 80)
    assert uri.parse("http://h:80/").authority == uri.Authority(None, "h", "name", 80)
    assert uri.parse("http://@h:/").authority == uri.Authority("", "h", "name", None)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("http://a b/", "1,9-1,9: got unexpected token: ' '"),
        ("http://evil.example\\@good.example/", "1,20-1,20: got unexpected token"),
        ("http://a@b@c/", "1,11-1,11: got unexpected token: '@c/'"),
        ("http://[::1/", "1,12-1,12: got unexpected token: '/', expected: ']'"),
        ("1a:b", "1,3-1,3: got unexpected token: ':b'"),
        ("a%zz", "1,2-1,2: got unexpected token: '%zz'"),
    ],
)
def test_uri_rejects_what_urlsplit_may_let_through(text: str, message: str) -> None:
    with pytest.raises(NoParseError, match=re.escape(message)):
        uri.parse(text)


@pytest.mark.parametrize("text", ["http://h:65536/", "http://[1::2::3]/"])
def test_uri_rejects_a_value_out_of_range(text: str) -> None:
    with pytest.raises(ParseErrors):
        uri.parse(text)


URI_CHARS = "ab1:/?#[]@!$&'()*+,;=-._~%2F"


@given(st.sampled_from(["http://", "//", "a:", "a+b:/", ""]), st.text(URI_CHARS, max_size=16))
def test_uri_splits_what_it_accepts_as_urlsplit_does(prefix: str, rest: str) -> None:
    text = prefix + rest
    try:
        parsed = uri.parse(text)
    except (NoParseError, ParseErrors):
        return
    split = urlsplit(text)
    scheme = (parsed.scheme or "").lower()
    assert (scheme, parsed.path, parsed.query or "", parsed.fragment or "") == (
        split.scheme,
        split.path,
        split.query,
        split.fragment,
    )


def test_uri_keeps_a_scheme_as_written_where_urlsplit_lowers_it() -> None:
    assert uri.parse("F:").scheme == "F"
    assert urlsplit("F:").scheme == "f"


def test_uri_query_decodes_each_pair() -> None:
    text = "a=1&b=x+y&flag&e=&c=%E2%82%AC"
    assert uri.parse_query(text) == parse_qsl(text, keep_blank_values=True)
    assert uri.parse_query("") == []


@given(st.text("ab=&+%25Fz", max_size=14))
def test_uri_query_reads_what_it_accepts_as_parse_qsl_does(text: str) -> None:
    try:
        pairs = uri.parse_query(text)
    except (NoParseError, ParseErrors):
        return
    assert pairs == parse_qsl(text, keep_blank_values=True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("a=1&", [("a", "1")]), ("&a=1", [("a", "1")]), ("a&&b", [("a", ""), ("b", "")]), ("&&", [])],
)
def test_uri_query_skips_an_empty_pair(text: str, expected: list[tuple[str, str]]) -> None:
    assert uri.parse_query(text) == expected


@pytest.mark.parametrize("text", ["a=%zz", "=v", "a=1&=v"])
def test_uri_query_rejects_what_parse_qsl_lets_through(text: str) -> None:
    with pytest.raises(NoParseError):
        uri.parse_query(text)


@pytest.mark.parametrize("text", ["a=\ud800", "\udfff=1"])
def test_uri_query_rejects_a_lone_surrogate(text: str) -> None:
    with pytest.raises(NoParseError):
        uri.parse_query(text)
    assert uri.compiled_pairs.try_parse(text) is None


def test_uri_query_rejects_an_escape_that_is_not_utf_8() -> None:
    with pytest.raises(ParseErrors, match="1 parsing error") as e:
        uri.parse_query("a=%FF")
    assert "not UTF-8 once percent-decoded" in str(e.value.exceptions[0])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("text/html", [("text", "html", 1.0)]),
        ("TEXT/HTML;Q=0.5", [("text", "html", 0.5)]),
        ("*/*;q=0", [("*", "*", 0.0)]),
        (",, a/b ,,c/d ,", [("a", "b", 1.0), ("c", "d", 1.0)]),
        ('a/b;x="c,d";q=0.1, e/f;q=0.5', [("a", "b", 0.1), ("e", "f", 0.5)]),
        ('a/b;x="q=0;";q=0.9', [("a", "b", 0.9)]),
        ("", []),
    ],
)
def test_http_accept_reads_each_range(text: str, expected: list[tuple[str, str, float]]) -> None:
    assert [(m.type, m.subtype, m.q) for m in http_headers.accept.parse_text(text)] == expected


@pytest.mark.parametrize(
    "text",
    ["a/b;q=2", "a/b;q=-1", "a/b;q=x", "a/b;q=1.5", "a/b;q=0.1234", "a/b;q=1e-1", "a/b;q=00.5"]
    + ["a/b;q=+0.5", 'a/b;q="0.5"', "a/b;q=", "a/b;q=1.001"],
)
def test_http_accept_rejects_what_is_not_a_weight(text: str) -> None:
    with pytest.raises(NoParseError):
        http_headers.accept.parse_text(text)


@pytest.mark.parametrize(
    ("text", "q"),
    [("a/b;q=0", 0.0), ("a/b;q=0.", 0.0), ("a/b;q=0.125", 0.125), ("a/b;Q=1.000", 1.0)],
)
def test_http_accept_reads_every_form_of_a_weight(text: str, q: float) -> None:
    assert http_headers.accept.parse_text(text)[0].q == q


def test_http_content_type_has_no_weight() -> None:
    media = http_headers.content_type.parse_text("text/plain; q=hello")
    assert media == http_headers.Media("text", "plain", {"q": "hello"})


def test_http_content_type_keeps_what_quotes_protect() -> None:
    media = http_headers.content_type.parse_text('multipart/form-data; boundary="a;b\\"c";;')
    assert media == http_headers.Media("multipart", "form-data", {"boundary": 'a;b"c'})


@pytest.mark.parametrize(
    ("text", "filename"),
    [
        ('attachment; filename="a;b.txt"', "a;b.txt"),
        ("attachment; filename=a.txt", "a.txt"),
        ("attachment; filename*=UTF-8''%e2%82%ac.txt; filename=\"x\"", "€.txt"),
        ("attachment; filename=\"x\"; FileName*=iso-8859-1'en'%E9.txt", "é.txt"),
        ("inline", None),
    ],
)
def test_http_content_disposition_prefers_the_extended_name(
    text: str, filename: str | None
) -> None:
    assert http_headers.content_disposition.parse_text(text).filename == filename


def test_http_content_disposition_rejects_a_quoted_extended_name() -> None:
    with pytest.raises(NoParseError):
        http_headers.content_disposition.parse_text("attachment; filename*=\"UTF-8''%C3%A9\"")


def test_http_content_disposition_rejects_an_escape_of_another_charset() -> None:
    with pytest.raises(ParseErrors):
        http_headers.content_disposition.parse_text("attachment; filename*=UTF-8''%FF")


def test_http_links_read_a_uri_between_brackets() -> None:
    first, second = http_headers.links.parse_text(
        '<https://a.example/x,y;z?page=2>; rel="next", </?page=5>; rel=last'
    )
    assert (first.target.path, first.target.query) == ("/x,y;z", "page=2")
    assert first.parameters == {"rel": "next"}
    assert second.target.authority is None
    assert second.parameters == {"rel": "last"}


def test_http_links_reject_a_target_that_is_not_a_uri() -> None:
    with pytest.raises(NoParseError, match=r"1,10-1,10: got unexpected token: ' '"):
        http_headers.links.parse_text("<http://a b>; rel=x")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bearer", [("Bearer", None, {})]),
        ("Negotiate abc=", [("Negotiate", "abc=", {})]),
        ('Basic realm="a, b"', [("Basic", None, {"realm": "a, b"})]),
        ("Basic realm = x", [("Basic", None, {"realm": "x"})]),
        (
            'Digest realm="x", qop="auth,auth-int", Basic realm="y", Negotiate abc=, Bearer',
            [
                ("Digest", None, {"realm": "x", "qop": "auth,auth-int"}),
                ("Basic", None, {"realm": "y"}),
                ("Negotiate", "abc=", {}),
                ("Bearer", None, {}),
            ],
        ),
    ],
)
def test_http_www_authenticate_tells_a_parameter_from_a_challenge(
    text: str, expected: list[tuple[str, str | None, dict[str, str]]]
) -> None:
    challenges = http_headers.www_authenticate.parse_text(text)
    assert [(c.scheme, c.token68, c.parameters) for c in challenges] == expected


def test_migration_rewrites_each_construct_and_keeps_the_rest() -> None:
    text = "# Title\n\n  [[a|b]]\t@due(2026-11-07)  \n\n\nend\n"
    expected = "# Title\n\n  [b](a)\t*due 2026-11-07*  \n\n\nend\n"
    assert migration.migrate(text) == expected


def test_migration_leaves_a_code_block_alone() -> None:
    text = "```\n[[a|b]] @due(2026-11-07)\n```\n"
    assert migration.migrate(text) == text


def test_migration_of_a_migrated_text_changes_nothing() -> None:
    once = migration.migrate("[[a|b]] @due(2026-11-07)\n")
    assert migration.migrate(once) == once


def test_migration_stops_at_a_broken_link() -> None:
    with pytest.raises(NoParseError, match=r"^2,4-2,4: got unexpected end of input"):
        migration.migrate("fine\n[[a")


def test_migration_stops_at_a_date_the_calendar_does_not_have() -> None:
    with pytest.raises(ParseErrors) as caught:
        migration.migrate("@due(2026-02-30)")
    assert caught.value.exceptions[0].msg == "1,6-1,15: a date"


def test_migration_leaves_a_file_with_a_broken_construct_as_it_was(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("[[a|b]] [[c")
    with pytest.raises(NoParseError):
        migration.migrate_file(path)
    assert path.read_text() == "[[a|b]] [[c"
    path.write_text("[[a|b]]")
    migration.migrate_file(path)
    assert path.read_text() == "[b](a)"


def test_notes_become_alerts_and_the_rest_stays() -> None:
    text = (
        "# Title\n\nIntro  text.\n\nNote: keep it\nshort.\n\n#### Note\n\nOne.\n\nTwo.\n\n## End\n"
    )
    assert notes.migrate(text) == (
        "# Title\n\nIntro  text.\n\n> [!NOTE]\n> keep it\n> short.\n\n"
        "> [!NOTE]\n> One.\n>\n> Two.\n\n## End\n"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("- a\n- **Note:** b\n  c\n- d\n", "- a\n- > [!NOTE]\n  > b\n  > c\n- d\n"),
        ("1. a\n2. Note: b\n   c\n", "1. a\n2. > [!NOTE]\n   > b\n   > c\n"),
        ("> Note: b\n> c\n", "> [!NOTE]\n> b\n> c\n"),
        ("- > Note: b\n  > c\n", "- > [!NOTE]\n  > b\n  > c\n"),
        ("> intro\n>\n> Note: b\n", "> intro\n>\n> > [!NOTE]\n> > b\n"),
    ],
)
def test_a_note_in_a_container_keeps_the_prefix_of_its_lines(text: str, expected: str) -> None:
    assert notes.migrate(text) == expected


def test_a_note_heading_without_a_paragraph_stops_the_migration_at_its_line() -> None:
    with pytest.raises(NoParseError, match="3,1-3,1: got unexpected token: bullet_list_open"):
        notes.migrate("#### Note\n\n- a list\n")


def test_a_migrated_file_migrates_to_itself() -> None:
    once = notes.migrate("Note: a\n\n- Note: b\n")
    assert notes.migrate(once) == once


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Note: a", "> [!NOTE]\n> a"),
        ("x\r\n\r\nNote: a\r\nb\r\n", "x\r\n\r\n> [!NOTE]\r\n> a\r\n> b\r\n"),
        ("x\r\rNote: a\r", "x\r\r> [!NOTE]\r> a\r"),
    ],
)
def test_a_note_keeps_its_line_endings_and_the_end_of_the_file(text: str, expected: str) -> None:
    assert notes.migrate(text) == expected


def test_a_span_of_tokens_without_lines_says_so() -> None:
    with pytest.raises(ValueError, match="no token from 0 to 1 has the lines"):
        notes.lines_of([MarkdownToken("paragraph_close", "", 0)], 0, 1)


JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8),
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=4), inner, max_size=4)
    ),
    max_leaves=12,
)
JSON_DOCUMENTS = st.tuples(
    st.lists(JSON_VALUES, max_size=4)
    | st.dictionaries(st.text(max_size=4), JSON_VALUES, max_size=4),
    st.sampled_from([None, 2]),
    st.booleans(),
).map(lambda t: json.dumps(t[0], indent=t[1], ensure_ascii=t[2]))


def grows(a: Any, b: Any) -> bool:
    """Whether `b` is `a` or `a` gone on: more items or members, or a longer last string."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = list(a)
        if keys != list(b)[: len(keys)] or any(repr(a[k]) != repr(b[k]) for k in keys[:-1]):
            return False
        return not keys or grows(a[keys[-1]], b[keys[-1]])
    if isinstance(a, list) and isinstance(b, list):
        n = len(a)
        if n > len(b) or repr(a[:-1]) != repr(b[: n - 1]):
            return False
        return n == 0 or grows(a[-1], b[n - 1])
    if isinstance(a, str) and isinstance(b, str):
        return b.startswith(a)
    return repr(a) == repr(b)


@given(JSON_DOCUMENTS)
def test_partial_json_reads_a_whole_document_as_json_loads_does(text: str) -> None:
    assert repr(partial_json.parse(text)) == repr(json.loads(text))


@given(JSON_DOCUMENTS)
def test_partial_json_gives_every_prefix_a_value_that_only_grows(text: str) -> None:
    values = [partial_json.parse(text[:n]) for n in range(len(text) + 1)]
    first = next(i for i, v in enumerate(values) if v is not MISSING)
    assert MISSING not in values[first:]
    for a, b in zip(values[first:], values[first + 1 :], strict=False):
        assert grows(a, b), (a, b)


def test_partial_json_reads_every_prefix_of_a_fenced_document() -> None:
    text = '```json\n{"a": [1, "x"]}\n```\n'
    values = [partial_json.parse(text[:n]) for n in range(len(text) + 1)]
    assert values[-1] == {"a": [1, "x"]}
    assert values[text.index("{")] is MISSING


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ('{"a": [1, 2', {"a": [1]}),
        ('{"a": [1, 2,', {"a": [1, 2]}),
        ('{"a": [1.5, 20], "b": -3e', {"a": [1.5, 20]}),
        ('{"a": "Pars', {"a": "Pars"}),
        ('{"a": "x\\u00', {"a": "x"}),
        ('{"a": "\\ud83d', {"a": ""}),
        ('{"a": tr', {}),
        ('{"a": true', {"a": True}),
        ('{"ab', {}),
        ('{"a":', {}),
        ("", MISSING),
        ("```js", MISSING),
    ],
)
def test_partial_json_gives_a_prefix_what_it_has_finished(text: str, value: object) -> None:
    assert partial_json.parse(text) == value


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ('{"a": [1, 2,], "b": 3,}', {"a": [1, 2], "b": 3}),
        ("{'a': 'it\\'s'}", {"a": "it's"}),
        ('{"a": True, "b": False, "c": None}', {"a": True, "b": False, "c": None}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
    ],
)
def test_partial_json_repairs_what_has_one_meaning(text: str, value: object) -> None:
    assert partial_json.parse(text) == value


@pytest.mark.parametrize(
    "text", ["[1 2]", '{"a" 1}', '{"a": 1} trailing', "[01]", "{a: 1}", "12", '"text"', "[1.]"]
)
def test_partial_json_rejects_what_has_more_than_one_meaning(text: str) -> None:
    with pytest.raises((NoParseError, LexerError)):
        partial_json.parse(text)
