"""CON96x: the glossary, `docs/explanation/glossary.md`, held to the code.

The rules apply where the docs rules do: the repository root holds `zensical.toml`.
"""

import ast
import re
from collections.abc import Iterator

from .._engine import Finding, rule
from .._project import Project

GLOSSARY = "docs/explanation/glossary.md"
# An entry's headword at the head of a line: `**term**`, `**`Name`**`, a code headword linked to its
# reference page, `**[`Name`][pkg.Name]**`, or several code names, `**`A` / `B`**`.
CODE_SPAN = r"\[?`[^`\n]+`(?:\]\[[\w.]+\])?"
HEADWORD = re.compile(rf"^\*\*(?:({CODE_SPAN}(?:\s*[|/]\s*{CODE_SPAN})*)|([^*`\n]+))\*\*", re.M)
CODE_NAME = re.compile(r"`([^`\n]+)`")
FENCED = re.compile(r"^```.*?^```", re.M | re.S)


def defined_names(project: Project) -> set[str]:
    """Every name the packages define: modules, classes, functions, assignments, parameters."""
    names: set[str] = set()
    for module in project.modules:
        names.update(module.name.split("."))
        if (tree := project.tree(module.path)) is None:
            continue
        for node in ast.walk(tree):
            match node:
                case ast.ClassDef() | ast.FunctionDef() | ast.AsyncFunctionDef():
                    names.add(node.name)
                case ast.Name(ctx=ast.Store()):
                    names.add(node.id)
                case ast.arg():
                    names.add(node.arg)
                case _:
                    pass
    return names


def headwords(text: str) -> Iterator[tuple[int, str | None, str | None]]:
    """Each entry's line, its code headword as written, and its prose headword."""
    blanked = FENCED.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    for match in HEADWORD.finditer(blanked):
        line = blanked.count("\n", 0, match.start()) + 1
        yield line, match.group(1), match.group(2)


@rule("CON960", "CON961", "CON962")
def glossary(project: Project) -> Iterator[Finding]:
    """The site has a glossary, and the names it defines exist.

    CON960: A site has `docs/explanation/glossary.md`.

    Reason: a project without a stated vocabulary grows one word with several meanings, each use
    locally reasonable, and nothing notices; the glossary is where a word's one meaning is written
    down and checked before a new name is chosen.

    Fix: add the page, with the project's terms, and link it from the menu.

    CON961: Every code headword in the glossary is a name the packages define.

    Reason: a glossary is read instead of the code, so an entry left behind by a rename misleads
    exactly the reader who looked it up.

    Fix: rename the entry, or drop it. A headword names a pair as `A` / `B` and a union as `A | B`.

    CON962: The glossary defines at least one term.

    Reason: an empty page passes CON961 and states no vocabulary.

    Fix: write the entries: a line starting with a bold headword, `**term**` or `**`Name`**`.
    """
    if not (project.root / "zensical.toml").is_file():
        return
    path = project.root / GLOSSARY
    if not path.is_file():
        yield Finding(GLOSSARY, None, "CON960", "the site has no glossary")
        return
    entries = list(headwords(project.text(path) or ""))
    if not entries:
        yield Finding(GLOSSARY, None, "CON962", "the glossary defines no term")
        return
    defined = defined_names(project)
    for line, code, _ in entries:
        if code is None:
            continue
        for name in CODE_NAME.findall(code):
            leaf = name.rsplit(".", 1)[-1].removesuffix("()")
            if leaf and leaf not in defined:
                yield Finding(
                    GLOSSARY, line, "CON961", f"`{name}` is defined nowhere in the packages"
                )
