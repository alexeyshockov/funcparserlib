import pytest
from hypothesis import given
from hypothesis import strategies as st

from examples import markdown
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
