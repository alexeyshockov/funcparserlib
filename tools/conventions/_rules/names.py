"""CON1xx: names."""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from .._engine import Finding, rule
from .._project import Project
from ._ast import alias_name, all_value

# `CONVENTIONS.md`'s shared list; a project adds its own in `[tool.conventions] initialisms`.
SHARED_INITIALISMS = ("API", "HTTP", "ID", "RPC", "URL")


def title_case_initialism(project: Project) -> re.Pattern[str]:
    """`Api`, `Http`, ... in a CapWords name, each followed by the end or another word."""
    words = sorted({*SHARED_INITIALISMS, *project.settings.get("initialisms", [])})
    spelled = "|".join(word.capitalize() for word in words)
    return re.compile(rf"(?:{spelled})(?=$|[A-Z0-9])")


def source_roots(project: Project) -> list[Path]:
    """Each package, the `tests/` and `examples/` beside it, and the repository's own."""
    candidates = [project.root / "tests", project.root / "examples"]
    for root in project.package_roots.values():
        candidates += [root, root.parent / "tests", root.parent / "examples"]
    return list(dict.fromkeys(path for path in candidates if path.is_dir()))


@rule("CON100")
def initialisms(project: Project) -> Iterator[Finding]:
    """An initialism in a CapWords name keeps every letter uppercase.

    Reason: PEP 8 CapWords keeps an initialism uppercase, and one spelling per word keeps names
    searchable. A name built from a package's own spelling, such as `Httpx2`, keeps it.

    Fix: write `HTTPClient`, not `HttpClient`. A project's own initialisms go in
    `[tool.conventions] initialisms`.

    Scope: class names and type aliases in each package, stubs included, and in the `tests/` and
    `examples/` beside it and at the repository root.
    """
    pattern = title_case_initialism(project)
    for path in project.under(source_roots(project), frozenset({".py", ".pyi"})):
        tree = project.tree(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            name = node.name if isinstance(node, ast.ClassDef) else alias_name(node)
            if name is not None and pattern.search(name):
                yield Finding(
                    project.relative(path),
                    getattr(node, "lineno", None),
                    "CON100",
                    f"`{name}` spells an initialism in title case",
                )


# An `async def` spelled as a sync function's name with `_async` after it. The `async_` prefix is
# not matched: it is also how an entity is spelled (`async_client`, `async_ok_handler`), and no scan
# can tell a twin action from an entity. A test is not matched either: pytest collects it by its
# `test` prefix, so `atest_...` would stop it running, and its suffix names the half it exercises.
ASYNC_SUFFIXED = re.compile(r"^(?P<bare>\w+?)_async$")


def _exported_elements(tree: ast.Module) -> dict[str, int]:
    """Each name in a module's literal `__all__`, with the line it is listed on."""
    value = all_value(tree)
    if not isinstance(value, ast.List | ast.Tuple):
        return {}
    return {
        e.value: e.lineno
        for e in value.elts
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    }


@rule("CON101")
def async_twins(project: Project) -> Iterator[Finding]:
    """The awaitable twin of an exported sync call is `a` + its name, as `aclose` is to `close`.

    Reason: one spelling for every twin pair, the one the standard library and anyio use, so a
    caller finds the async half without reading the module.

    Fix: rename `emit_async` to `aemit`.

    Scope: the names a public module's `__all__` exports, where a caller reads them. A private
    helper or a test is free. A side marker with no bare partner (`resolve_async` beside
    `resolve_sync`) is not a twin, and the `async_` prefix is left to review: it is also an entity's
    spelling.
    """
    for module in project.modules:
        if not module.public or (tree := project.tree(module.path)) is None:
            continue
        names = _exported_elements(tree)
        for name in sorted(names):
            marked = ASYNC_SUFFIXED.match(name)
            if marked is not None and marked["bare"] in names and not name.startswith("test"):
                yield Finding(
                    project.relative(module.path),
                    names[name],
                    "CON101",
                    f"`{module.name}.{name}`, the twin of an exported sync call: "
                    f"`a{marked['bare']}`",
                )
