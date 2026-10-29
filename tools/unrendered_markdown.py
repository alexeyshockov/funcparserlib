"""Fail on Markdown that reached a built page as text: syntax the renderer did not parse.

Usage: `python tools/unrendered_markdown.py SITE_DIR`; exits 1 listing each page and line.

A site with no page fails too: a check that read nothing has passed nothing.
"""

# Generated; propose changes in an issue rather than editing this file.
#
# A table whose rows were rewrapped into one paragraph, a link or block broken by an edit, a list
# nested at the wrong indent, or an include that did not expand all build without a warning; each
# leaves its syntax in the page's text. Text inside code, preformatted blocks and scripts is not
# read.

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PATTERNS = {
    "table": re.compile(r"\|\s*:?-{3,}:?\s*\||^\s*\|.*\|\s*\|"),
    # A target is a path, URL or anchor: `[1, 2](3)` in prose is not one.
    "link": re.compile(r"\]\([^)\s]*[./#:][^)\s]*\)"),
    "reference": re.compile(r"\]\[[\w.]*\]"),
    "admonition": re.compile(r"^\s*(!!!|\?\?\?\+?) \w"),
    "snippet": re.compile(r"--8<--"),
    "mkdocstrings": re.compile(r"^\s*::: [\w.]+"),
    "emphasis": re.compile(r"(?<![\w*])\*\*\S[^*]*\S\*\*(?![\w*])"),
}
# A heading leaks at the start of a paragraph; at the start of a table cell, `# of retries` is text.
HEADING = re.compile(r"^#{1,6} \S")
# A list marker leaks at the start of a paragraph's or an item's later line: Python-Markdown nests a
# list only at an indent of 4, and joins a `- ` line to the paragraph above it with no blank line.
LIST = re.compile(r"^\s*([-*+]|\d+\.) \S")
SKIP = frozenset({"code", "pre", "script", "style", "kbd", "samp", "textarea"})
# A line break in the text the patterns read; `table` matches within one line.
BLOCKS = frozenset({"p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "div", "br", "dd"})


class _Text(HTMLParser):
    """The text of a page's `<article>`, outside code."""

    def __init__(self) -> None:
        super().__init__()
        # (the block a line starts in, its text)
        self.lines: list[tuple[str, str]] = [("", "")]
        self._skip = 0
        self._article = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._article += tag == "article"
        self._skip += tag in SKIP
        if tag in BLOCKS:
            self.lines.append((tag, ""))

    def handle_endtag(self, tag: str) -> None:
        self._article -= tag == "article" and self._article > 0
        self._skip -= tag in SKIP and self._skip > 0

    def handle_data(self, data: str) -> None:
        if self._article and not self._skip:
            block, text = self.lines[-1]
            self.lines[-1] = (block, text + data)


def leaks(html: str) -> list[tuple[str, str]]:
    """Each line of the page's text holding Markdown syntax, with the kind of syntax."""
    parser = _Text()
    parser.feed(html)
    found = []
    for block, text in parser.lines:
        for index, line in enumerate(text.splitlines()):
            kinds = [kind for kind, pattern in PATTERNS.items() if pattern.search(line)]
            if block == "p" and HEADING.search(line):
                kinds.append("heading")
            if index and block in {"p", "li"} and LIST.search(line):
                kinds.append("list")
            found += [(kind, line.strip()) for kind in kinds]
    return found


def main(argv: list[str]) -> int:
    site = Path(argv[1])
    pages = sorted(site.rglob("*.html"))
    if not pages:
        print(f"no page under {site}: the site was not built there", file=sys.stderr)
        return 1
    count = 0
    for page in pages:
        for kind, line in leaks(page.read_text()):
            count += 1
            print(f"{page.relative_to(site)}: unrendered {kind}: {line[:120]}")
    if count:
        print(f"{count} line(s) of Markdown reached the site as text", file=sys.stderr)
    return 1 if count else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
