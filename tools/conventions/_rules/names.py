"""CON1xx: names."""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from .._engine import Finding, rule
from .._project import Project
from ._ast import alias_name
from ._public import exports

# `CONVENTIONS.md`'s shared list; a project adds its own in `[tool.conventions] initialisms`.
SHARED_INITIALISMS = ("API", "HTTP", "ID", "RPC", "URL")


def title_case_initialism(project: Project) -> re.Pattern[str]:
    """`Api`, `Http`, ... in a CapWords name, each followed by the end or another word."""
    words = sorted({*SHARED_INITIALISMS, *project.settings.get("initialisms", [])})
    spelled = "|".join(word.capitalize() for word in words)
    return re.compile(rf"(?:{spelled})(?=$|[A-Z0-9])")


def source_roots(project: Project) -> list[Path]:
    """Each package, the `typesafety/`, `tests/` and `examples/` beside it, and the root's own."""
    candidates = [project.root / "typesafety", project.root / "tests", project.root / "examples"]
    for root in project.package_roots.values():
        candidates += [
            root,
            root.parent / "typesafety",
            root.parent / "tests",
            root.parent / "examples",
        ]
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


# A twin pair is two definitions in one scope, a sync `bare` and an `async def` beside it, spelled
# other than `bare` / `abare`. Each pattern names the side it matches; the other side is `bare`.
# `async_` alone, with no sync partner, is an entity's spelling (`async_client`, `async_ok_handler`)
# and is not matched. A test is not matched either: pytest collects it by its `test` prefix, so
# `atest_...` would stop it running, and its suffix names the half it exercises.
ASYNC_SIDE = (re.compile(r"^(?P<bare>\w+?)_async$"), re.compile(r"^async_(?P<bare>\w+)$"))
SYNC_SIDE = re.compile(r"^(?P<bare>\w+?)_(?:blocking|sync)$")

Def = ast.FunctionDef | ast.AsyncFunctionDef


def _scopes(tree: ast.Module) -> Iterator[tuple[str, str, list[ast.stmt]]]:
    """The module body and each class body under it whose class name carries no underscore.

    Each comes with its dotted prefix and its owner: the top-level class it sits in, or `""` for the
    module body, where each definition is its own owner.
    """
    yield "", "", tree.body
    stack: list[tuple[str, str, list[ast.stmt]]] = [("", "", tree.body)]
    while stack:
        prefix, owner, body = stack.pop()
        for node in body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                qualified = f"{prefix}{node.name}."
                yield qualified, owner or node.name, node.body
                stack.append((qualified, owner or node.name, node.body))


def _misnamed_twins(defs: dict[str, Def]) -> Iterator[tuple[Def, str]]:
    """Each definition that is half of a twin pair under another spelling, with the pair's fix."""
    for name, node in sorted(defs.items()):
        if name.startswith(("_", "test")):
            continue
        is_async = isinstance(node, ast.AsyncFunctionDef)
        sides = ASYNC_SIDE if is_async else (SYNC_SIDE,)
        for pattern in sides:
            if (marked := pattern.match(name)) is None:
                continue
            bare = marked["bare"]
            partner = defs.get(bare)
            if partner is not None and isinstance(partner, ast.AsyncFunctionDef) != is_async:
                yield node, f"`{bare}` for the sync call and `a{bare}` for its awaitable twin"
                break


def _exported(project: Project) -> set[tuple[str, str]]:
    """The defining module and top-level name of everything a public module's `__all__` lists."""
    return {
        (site.module, site.name)
        for export in exports(project)
        if (site := export.definition) is not None and site.in_packages
    }


@rule("CON101", "CON102")
def async_twins(project: Project) -> Iterator[Finding]:
    """The awaitable twin of a sync call is `a` + its name, as `aclose` is to `close`.

    CON101: A twin pair the public API reaches is spelled `x` / `ax`.

    Reason: one spelling for every twin pair, the one the standard library and anyio use, so a
    caller finds the async half without reading the module.

    Fix: rename `emit_async` or `async_emit` beside `emit` to `aemit`; rename `emit_blocking` beside
    an awaitable `emit` to `emit`, and the awaitable to `aemit`.

    Scope: a pair is two functions or methods in one module or class body; the public API reaches a
    function a public module's `__all__` exports, or a method of a class it exports, wherever either
    is defined. A side marker with no bare partner (`resolve_async` beside `resolve_sync`) is not a
    twin, and neither is a suffix naming where the call is made from (`start_from_thread` beside
    `start`). An `async_` factory beside a sync one of the same name builds an entity, not a twin:
    it carries `# conventions: ignore[CON101]`.

    CON102: A twin pair inside the package is spelled `x` / `ax`.

    Reason: the package's own contributors read one vocabulary, the one its API uses.

    Fix: as CON101.

    Scope: every pair CON101 does not report. A name with a leading underscore, its class's methods
    and a test are free.
    """
    trees = {
        module.name: tree
        for module in project.modules
        if (tree := project.tree(module.path)) is not None
    }
    exported = _exported(project)
    for module in project.modules:
        if (tree := trees.get(module.name)) is None:
            continue
        for prefix, owner, body in _scopes(tree):
            defs = {node.name: node for node in body if isinstance(node, Def)}
            for node, fix in _misnamed_twins(defs):
                public = (module.name, owner or node.name) in exported
                yield Finding(
                    project.relative(module.path),
                    node.lineno,
                    "CON101" if public else "CON102",
                    f"`{module.name}.{prefix}{node.name}` is half of a sync/async twin pair: {fix}",
                )
