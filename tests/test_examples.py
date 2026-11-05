import math
from decimal import Decimal, InvalidOperation
from typing import Any

import msgpack
import pytest
from hypothesis import given
from hypothesis import strategies as st
from ruamel.yaml.error import YAMLError

from examples import kdl, kdl_text, markdown, msgpack_format, yaml
from examples import markdown as md
from examples.calculator import calculate
from examples.kdl import Node, Value
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


def test_kdl_comments_nest_up_to_the_level_of_the_regexp() -> None:
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
