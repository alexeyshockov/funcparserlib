"""The Markdown side: inline code the index resolves becomes an optional `<autoref>`."""

# Generated; propose changes in an issue rather than editing this file.

import re
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any
from xml.etree.ElementTree import Element

from markdown import Markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor

from mdautolink._host import current_page_url
from mdautolink._index import NameIndex, build_index, rendered_modules

# `Name`, `mod.Name`, `Name.member`, each optionally one call, whose arguments may hold calls of
# their own: `run()`, `tcp.connect(TCPHost(name, port))`; not an expression that starts and ends
# with one, as `Pool(1) if big else Pool(2)`.
NAME = re.compile(r"^(?P<name>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\((?:[^()]|\([^()]*\))*\))?$")
# Inside these an inline code span is left alone: it is a link already, or a heading.
SKIP = frozenset({"a", "autoref", "pre", "h1", "h2", "h3", "h4", "h5", "h6"})
NOLINK = "nolink"


@cache
def _cached_index(modules: tuple[str, ...], search_paths: tuple[str, ...]) -> NameIndex:
    return build_index(modules, search_paths)


class Linker(Treeprocessor):
    """Wrap each inline `code` element whose text names an indexed object in an `<autoref>`."""

    def __init__(
        self,
        md: Markdown | None,
        index: NameIndex,
        scopes: Sequence[tuple[str, str]] = (),
        page_url: Callable[[], str | None] = current_page_url,
    ) -> None:
        super().__init__(md)
        self.index = index
        self.scopes = scopes
        self.page_url = page_url

    def run(self, root: Element) -> None:
        url = self.page_url()
        prefer = None
        if url is not None:
            scoped = (module for prefix, module in self.scopes if url.startswith(prefix))
            prefer = next(scoped, None)
        self._walk(root, prefer)

    def _walk(self, parent: Element, prefer: str | None) -> None:
        for position, child in enumerate(list(parent)):
            if child.tag in SKIP:
                continue
            if child.tag != "code":
                self._walk(child, prefer)
                continue
            if len(child) or not child.text or NOLINK in child.get("class", "").split():
                continue
            match = NAME.match(child.text)
            target = match and self.index.resolve(match["name"], prefer)
            if not target:
                continue
            # `optional`: an identifier the reference never registered renders as plain code.
            link = Element("autoref", {"identifier": target, "optional": ""})
            link.tail, child.tail = child.tail, None
            link.append(child)
            parent.remove(child)
            parent.insert(position, link)


class AutolinkExtension(Extension):
    def __init__(self, **kwargs: Any) -> None:
        self.config = {
            "modules": [[], "Module paths the reference renders; when empty, the `:::` lines."],
            "docs_dir": ["docs", "The pages read for their `:::` lines when `modules` is empty."],
            "search_paths": [["."], "Where griffe finds the modules' source."],
            "scopes": [[], "[page URL prefix, module] pairs: such a page prefers that module."],
        }
        super().__init__(**kwargs)

    def extendMarkdown(self, md: Markdown) -> None:  # noqa: N802 (Python-Markdown's name)
        modules = tuple(self.getConfig("modules")) or rendered_modules(self.getConfig("docs_dir"))
        index = _cached_index(modules, tuple(self.getConfig("search_paths")))
        scopes = [(prefix, module) for prefix, module in self.getConfig("scopes")]
        # After the inline processor (priority 20): code spans exist as elements by then.
        md.treeprocessors.register(Linker(md, index, scopes), "mdautolink", 4)


def makeExtension(**kwargs: Any) -> AutolinkExtension:  # noqa: N802 (Python-Markdown's name)
    return AutolinkExtension(**kwargs)
