import pytest
from hypothesis import given
from hypothesis import strategies as st
from ruamel.yaml.error import YAMLError

from examples import markdown, yaml
from examples import markdown as md
from examples.calculator import calculate
from funcparserlib.parser import NoParseError


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
        ("a: 1\na: 2\n", "duplicate key 'a' at (2, 1)"),
        ("<<: 1\n", "merge value is not a mapping at (1, 1)"),
    ],
)
def test_yaml_rejects_a_document_without_a_meaning(text: str, message: str) -> None:
    with pytest.raises(yaml.ResolveError) as e:
        yaml.parse(text)
    assert str(e.value) == message


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
