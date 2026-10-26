"""CON5xx: source idioms: imports, dataclasses, headers and inline code."""

import ast
import re
from collections.abc import Iterator, Set
from pathlib import Path
from typing import TypeGuard

from .._engine import TEXT_SUFFIXES, Finding, Fixes, fixer, rule
from .._project import Project
from ._ast import bare_names, resolved
from .names import source_roots
from .surface import template_owned


def _sources(project: Project) -> Iterator[tuple[Path, str, ast.Module]]:
    """Each Python file in the source roots that reads and parses, with its text and tree."""
    for path in project.under(source_roots(project), frozenset({".py"})):
        text, tree = project.text(path), project.tree(path)
        if text is not None and tree is not None:
            yield path, text, tree


# --- Imports ---


def rationale_offenders(source: str, tree: ast.Module) -> list[int]:
    """The lines of `if TYPE_CHECKING:` blocks without a comment above, on, or first inside them."""
    lines = source.splitlines()
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        above = node.lineno - 2
        while above >= 0 and not lines[above].strip():
            above -= 1
        commented = (
            (above >= 0 and lines[above].lstrip().startswith("#"))
            or "#" in lines[node.lineno - 1]
            or (node.lineno < len(lines) and lines[node.lineno].lstrip().startswith("#"))
        )
        if not commented:
            found.append(node.lineno)
    return found


@rule("CON500")
def type_checking_rationale(project: Project) -> Iterator[Finding]:
    """A `TYPE_CHECKING` block says which reason it is: a cycle, an optional import, conformance.

    Reason: an import moved under `TYPE_CHECKING` changes what runs; without the reason the next
    reader cannot tell whether it may move back.

    Fix: add a comment above the `if`, on it, or as the first line inside it.

    Scope: each package, and the `tests/` and `examples/` beside it and at the root.
    """
    for path, text, tree in _sources(project):
        for line in rationale_offenders(text, tree):
            yield Finding(
                project.relative(path),
                line,
                "CON500",
                "TYPE_CHECKING block without an adjacent rationale comment",
            )


def sibling_offenders(
    module_name: str, is_package: bool, tree: ast.Module
) -> list[tuple[int, str]]:
    """Absolute imports of a module in the importing module's own directory."""
    own = module_name if is_package else module_name.rpartition(".")[0]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module:
            targets = [node.module]
        elif isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        else:
            continue
        found += [(node.lineno, target) for target in targets if target.rpartition(".")[0] == own]
    return found


@rule("CON501")
def sibling_imports(project: Project) -> Iterator[Finding]:
    """A module imports its siblings relatively.

    Reason: `from ._core import X` within a directory says the two move together; an absolute
    sibling import breaks when the directory is renamed or vendored. An import from another
    directory stays absolute.

    Fix: write `from ._core import X` for a module in the same directory.

    Scope: the modules of each package.
    """
    for module in project.modules:
        tree = project.tree(module.path)
        if tree is None:
            continue
        is_package = module.path.name == "__init__.py"
        for line, target in sibling_offenders(module.name, is_package, tree):
            yield Finding(
                project.relative(module.path), line, "CON501", f"import {target} relatively"
            )


# Layers and consumers. A subpackage is a layer unless `[tool.conventions] consumers` names it: a
# layer is part of the package and imports the modules it builds on directly, and a consumer builds
# on the package from above, as a user would, through its public modules only.


def is_private_path(name: str) -> bool:
    return any(part.startswith("_") and not part.startswith("__") for part in name.split("."))


def imported_modules(module: str, is_package: bool, tree: ast.Module) -> list[tuple[int, str]]:
    """Each module an import may name, absolute: `from a import b` names `a` and `a.b`."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = resolved(module, is_package, node)
            found += [(node.lineno, base)]
            found += [(node.lineno, f"{base}.{alias.name}") for alias in node.names]
    return found


def consumer_of(name: str, consumers: Set[str]) -> str | None:
    return next((c for c in consumers if name == c or name.startswith(c + ".")), None)


def root_import_offenders(package: str, tree: ast.Module) -> list[tuple[int, str]]:
    """`from <package> import X` statements, as `(line, X)`."""
    return [
        (node.lineno, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == package
        for alias in node.names
    ]


def private_prefix(name: str) -> str:
    """`name` up to its first underscore part: `a.b._c.d` is `a.b._c`."""
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part.startswith("_") and not part.startswith("__"):
            return ".".join(parts[: index + 1])
    return name


def private_reach(
    module: str, consumer: str, is_package: bool, tree: ast.Module, builders: Set[str] = frozenset()
) -> list[tuple[int, str]]:
    """The underscore modules of the package outside `consumer` that `module` imports.

    The generated helpers module `<package>._commons` and the modules in `builders` hold message
    builders, which shape an error's prose rather than the package's protocol, and are not counted.
    """
    package = module.partition(".")[0]
    return sorted(
        {
            (line, private_prefix(target))
            for line, target in imported_modules(module, is_package, tree)
            if (target == package or target.startswith(package + "."))
            and not (target == consumer or target.startswith(consumer + "."))
            and is_private_path(target)
            and not template_owned(private_prefix(target))
            and private_prefix(target) not in builders
        }
    )


@rule("CON506", "CON507", "CON508")
def layering(project: Project) -> Iterator[Finding]:
    """A layer imports what it builds on directly; a consumer imports public modules only.

    CON506: A module imports nothing through its package root.

    Reason: `from pkg import X` inside `pkg` is an import cycle, whatever `X` is. For a name it also
    drags in every layer the root imports; for a submodule it works at runtime, through
    `sys.modules`, while the cycle is real.

    Fix: import the module that defines the name: `from pkg._core import X`, or
    `import pkg._core as core`. A subpackage that builds on the package as a user would goes in
    `[tool.conventions] consumers`.

    CON507: A consumer imports no underscore module outside its own subpackage.

    Reason: a consumer is the example of building on the package; reaching a private module says the
    public surface is missing something. The generated helpers module `_commons` and the module
    `[tool.conventions.messages] builders_home` names hold message builders, and are exempt.

    Fix: import the name through a public module, or export it from one.

    CON508: A `consumers` entry names a subpackage of a package.

    Reason: an entry that names nothing exempts nothing, and reads as a decision that holds.

    Fix: correct the dotted name, or remove the entry.

    Scope: the modules of each package.
    """
    consumers = frozenset(project.settings.get("consumers", []))
    pyproject = project.root / "pyproject.toml"
    home = project.settings.get("messages", {}).get("builders_home")
    builders = frozenset(
        module.name for module in project.modules if home and module.path == project.root / home
    )
    for consumer in sorted(consumers):
        path = project.module_path(consumer)
        if path is None or path.name != "__init__.py" or "." not in consumer:
            yield Finding(
                "pyproject.toml",
                project.line_of(pyproject, "consumers"),
                "CON508",
                f"consumer `{consumer}` is not a subpackage of a package",
            )
    for module in project.modules:
        tree = project.tree(module.path)
        if tree is None:
            continue
        package = module.name.partition(".")[0]
        relative = project.relative(module.path)
        is_package = module.path.name in {"__init__.py", "__init__.pyi"}
        consumer = consumer_of(module.name, consumers)
        # A stub never runs, so an import in one makes no cycle.
        if consumer is None and module.name != package and module.path.suffix == ".py":
            for line, name in root_import_offenders(package, tree):
                yield Finding(
                    relative, line, "CON506", f"`{name}` imported through the package root"
                )
        if consumer is not None:
            for line, target in private_reach(module.name, consumer, is_package, tree, builders):
                yield Finding(
                    relative, line, "CON507", f"consumer `{consumer}` imports private `{target}`"
                )


def foreign_private_imports(package: str, tree: ast.Module) -> list[tuple[int, str]]:
    """Plain `import x._y` statements of another top-level package's underscore module."""
    return [
        (node.lineno, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.partition(".")[0] != package and is_private_path(alias.name)
    ]


@rule("CON509")
def foreign_private_modules(project: Project) -> Iterator[Finding]:
    """A package imports no underscore module of another package with a plain `import`.

    Reason: another package's underscore module may change in any release. ruff's PLC2701 holds the
    `from x._y import z` form and does not see `import x._y`.

    Fix: import the name through the other package's public module.

    Scope: the modules of each package.
    """
    for module in project.modules:
        tree = project.tree(module.path)
        if tree is None:
            continue
        for line, name in foreign_private_imports(module.name.partition(".")[0], tree):
            yield Finding(
                project.relative(module.path),
                line,
                "CON509",
                f"`{name}` is another package's private module",
            )


# --- Two dataclass idioms ---
#
# A value is `@valueclass` and an identity object is `@refclass`, both from the package's
# `_commons`. `@dataclass(frozen=True)` is the value role spelled another way; `eq=False` on a
# dataclass is the reference role spelled another way, unless the class takes an `__eq__` from its
# own body or its bases.

# The `collections.abc` classes that define `__eq__`, so a dataclass deriving from one keeps it.
EQ_BASES = frozenset({"Mapping", "MutableMapping", "Set", "MutableSet", "AbstractSet"})


def calls_decorator(deco: ast.expr, name: str) -> TypeGuard[ast.Call]:
    """Whether a decorator is a call of `name`, bare or as a module attribute."""
    if not isinstance(deco, ast.Call):
        return False
    func = deco.func
    return (func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")) == name


def keyword_false(call: ast.Call, name: str) -> bool:
    return any(
        kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is False
        for kw in call.keywords
    )


def keyword_true(call: ast.Call, name: str) -> bool:
    return any(
        kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def defines_eq(node: ast.ClassDef) -> bool:
    return any(isinstance(item, ast.FunctionDef) and item.name == "__eq__" for item in node.body)


def eq_false_dataclasses(tree: ast.Module, eq_bases: Set[str]) -> list[tuple[int, str]]:
    """`@dataclass(..., eq=False)` classes that take no `__eq__` from their body or bases."""
    found = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.ClassDef)
            or defines_eq(node)
            or bare_names(node.bases) & eq_bases
        ):
            continue
        for deco in node.decorator_list:
            if calls_decorator(deco, "dataclass") and keyword_false(deco, "eq"):
                found.append((deco.lineno, node.name))
    return found


def frozen_dataclasses(tree: ast.Module) -> list[tuple[int, str]]:
    """`@dataclass(frozen=True, ...)` classes: values spelled without `@valueclass`."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for deco in node.decorator_list:
            if calls_decorator(deco, "dataclass") and keyword_true(deco, "frozen"):
                found.append((deco.lineno, node.name))
    return found


def unstated_slots_opt_outs(source: str, tree: ast.Module) -> list[tuple[int, str]]:
    """`@refclass(slots=False)` classes with neither a `@cached_property` nor a stated reason.

    The reason is a `# slots=False: <why>` comment on the decorator line.
    """
    lines = source.splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        opts_out = [
            deco
            for deco in node.decorator_list
            if calls_decorator(deco, "refclass") and keyword_false(deco, "slots")
        ]
        cached = any(
            isinstance(member, ast.FunctionDef)
            and "cached_property" in bare_names(member.decorator_list)
            for member in node.body
        )
        stated = any("# slots=False:" in lines[deco.lineno - 1] for deco in opts_out)
        if opts_out and not cached and not stated:
            found.append((node.lineno, node.name))
    return found


@rule("CON502", "CON503", "CON513")
def dataclass_idioms(project: Project) -> Iterator[Finding]:
    """A value is a `@valueclass`; an identity object is a `@refclass`.

    CON502: An identity dataclass is a `@refclass`, not `@dataclass(eq=False)`.

    Reason: `eq=False` is the reference role spelled a second way; one spelling keeps the two roles
    visible at the decorator. A class that takes `__eq__` from its body or its bases (a `Mapping`,
    or a package class defining `__eq__`) is exempt.

    Fix: decorate with `@refclass` from the package's `_commons`.

    CON503: A `@refclass(slots=False)` says why it keeps its `__dict__`.

    Reason: dropping slots costs memory on every instance; the reason is either a `@cached_property`
    the class shows, or it is written down.

    Fix: add `# slots=False: <why>` on the decorator line, or drop `slots=False`.

    CON513: A value class is a `@valueclass`, not `@dataclass(frozen=True)`.

    Reason: `@valueclass` is frozen for the type checkers and builds a non-frozen class, whose
    `__init__` assigns its fields directly; a frozen dataclass's `__init__` sets each field through
    `object.__setattr__`, about three times the cost.

    Fix: decorate with `@valueclass` from the package's `_commons`, keeping the other keywords.

    Scope, CON513: as below, but for the `examples/` directories, which use the public API alone and
    so cannot reach `_commons`.

    Scope: each package, and the `tests/` and `examples/` beside it and at the root.
    """
    eq_classes = {
        node.name
        for module in project.modules
        if (tree := project.tree(module.path)) is not None
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and defines_eq(node)
    }
    eq_bases = EQ_BASES | eq_classes
    for path, text, tree in sorted(_sources(project), key=lambda item: item[0]):
        relative = project.relative(path)
        for line, name in eq_false_dataclasses(tree, eq_bases):
            yield Finding(relative, line, "CON502", f"use @refclass for `{name}`, not eq=False")
        example = "examples" in Path(relative).parts
        for line, name in [] if example else frozen_dataclasses(tree):
            yield Finding(
                relative, line, "CON513", f"use @valueclass for `{name}`, not frozen=True"
            )
        for line, name in unstated_slots_opt_outs(text, tree):
            yield Finding(
                relative, line, "CON503", f"`{name}` opts out of slots without saying why"
            )


# --- Markup ---

# Files whose line comment is `#`, by suffix and by name; Markdown is not one, `#` is a heading.
HASH_COMMENTED = frozenset(
    {".py", ".pyi", ".toml", ".yaml", ".yml", ".env", ".sh", ".ps1", ".just"}
)
HASH_COMMENTED_NAMES = frozenset({"justfile", ".gitignore", ".gitattributes", ".editorconfig"})


def comment_marker(path: Path) -> str | None:
    """The line comment a structural header is written in: `#`, `//` for Rust, else `None`."""
    if path.suffix in HASH_COMMENTED or path.name in HASH_COMMENTED_NAMES:
        return "#"
    return "//" if path.suffix == ".rs" else None


# `<marker> --- Sentence case title ---`, the one spelling of a structural header, and any
# comment that was meant as one: three or more dashes after the marker, or two or more at both
# ends. A `--8<--` snippet marker is neither. `///` and `//!` are Rust doc comments, which are
# Markdown, so the character after `//` has to be a space or a dash.
def header_offenders(text: str, marker: str = "#", names: Set[str] = frozenset()) -> list[int]:
    """The lines of `text` holding a structural header in any shape but the canonical one.

    A title may start lowercase when its first word is in `names`: someone else's identifier.
    """
    quoted = re.escape(marker)
    header = re.compile(rf"\s*{quoted} --- ([^\n]+) ---")
    decorative = re.compile(rf"^\s*{quoted}\s*(?:-{{3,}}.*|-{{2,}}.*-{{2,}}\s*)$")
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if "--8<--" in line or not decorative.match(line):
            continue
        match = header.fullmatch(line)
        if match is None:
            found.append(number)
            continue
        title = match.group(1)
        if title[0].islower() and re.split(r"[\s,:]", title, maxsplit=1)[0] not in names:
            found.append(number)
    return found


@rule("CON504")
def headers(project: Project) -> Iterator[Finding]:
    """A section header is `# --- Sentence case title ---`; a bare separator is not used.

    Reason: one spelling lets a reader and a search find every section the same way.

    Fix: write `# --- Sentence case title ---` (`// --- Sentence case title ---` in Rust), or remove
    the separator. A title that starts with someone else's lowercase identifier, such as `sslmode`,
    keeps it: list the word in `[tool.conventions] header_names`.

    Scope: every file of the repository whose line comment is `#` (Python, TOML, YAML, shell,
    justfiles, the git and editor configs), and Rust's `//`.
    """
    names = frozenset(project.settings.get("header_names", []))
    for path in project.files:
        marker = comment_marker(path)
        relative = project.relative(path)
        if marker is None or (text := project.text(path)) is None:
            continue
        for line in header_offenders(text, marker, names):
            yield Finding(relative, line, "CON504", f"write `{marker} --- Sentence case title ---`")


# reStructuredText markup, which the docs render as stray punctuation: Markdown is the only markup
# here. Two backticks around text that holds none, with no third on either side, is the inline
# literal; Markdown's double backticks wrap code that holds a backtick, which this does not match.
RST_LITERAL = re.compile(r"(?<!`)``(?!`)([^`\n]+?)(?<!`)``(?!`)")
# `:meth:`x``, `:py:func:`x``, with `~`, `!` or an explicit `title <target>` inside.
RST_ROLE = re.compile(r"(?<![\w`]):(?:[a-z]+:)?[a-z]+:`([^`\n]+)`")
RST_FIELD = re.compile(
    r"^\s*:(?:param|type|returns?|rtype|raises?|ivar|cvar|var|keyword|kwarg)\b[^:\n]*:"
)
RST_DIRECTIVE = re.compile(r"^\s*\.\. [a-z][a-z-]*::")
# A paragraph ending in `::`, or a bare `::` line, opens an indented literal block.
RST_LITERAL_BLOCK = re.compile(r"(?:^|[^:])::\s*$")
# Sphinx's attribute-documentation comment, in Python only.
SPHINX_COMMENT = re.compile(r"^(\s*)#:")

RST_FORMS = (
    (RST_LITERAL, "reStructuredText double backticks"),
    (RST_ROLE, "a reStructuredText role"),
    (RST_FIELD, "a reStructuredText field; use a Google-style section"),
    (RST_DIRECTIVE, "a reStructuredText directive"),
    (RST_LITERAL_BLOCK, "a reStructuredText literal block; use a fenced code block"),
)


def _outside_fences(text: str, markdown: bool) -> Iterator[tuple[int, str]]:
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if markdown and line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            yield number, line


def rst_offenders(text: str, markdown: bool, python: bool = False) -> list[tuple[int, str]]:
    """Each line holding reStructuredText markup, once per form, with the form's name."""
    found = []
    for number, line in _outside_fences(text, markdown):
        found += [(number, message) for pattern, message in RST_FORMS if pattern.search(line)]
        if python and SPHINX_COMMENT.match(line):
            found.append((number, "a Sphinx `#:` comment; use a plain `#`"))
    return found


def _role_text(match: re.Match[str]) -> str:
    text = match.group(1)
    if text.endswith(">") and " <" in text:
        text = text.rpartition(" <")[0]
    return f"`{text.lstrip('~!')}`"


def rst_rewritten(text: str, markdown: bool, python: bool) -> str:
    """`text` with each literal, role and `#:` comment written as Markdown and a plain comment."""
    lines = text.split("\n")
    for number, line in _outside_fences(text, markdown):
        line = RST_LITERAL.sub(r"`\1`", line)
        line = RST_ROLE.sub(_role_text, line)
        if python:
            line = SPHINX_COMMENT.sub(r"\1#", line)
        lines[number - 1] = line
    return "\n".join(lines)


def _rst_texts(project: Project) -> Iterator[tuple[Path, str, bool, bool]]:
    for path in project.files:
        if path.suffix not in TEXT_SUFFIXES or (text := project.text(path)) is None:
            continue
        yield path, text, path.suffix == ".md", path.suffix in {".py", ".pyi"}


@rule("CON505")
def inline_code(project: Project) -> Iterator[Finding]:
    """Comments, docstrings and pages are Markdown: no reStructuredText or Sphinx markup.

    Reason: the docs render Markdown, and docstrings follow the Google convention. A
    reStructuredText literal, role, field, directive or literal block renders as stray punctuation,
    and a `#:` comment documents an attribute for a tool nothing here runs.

    Fix: write `x` with single backticks, for a role too; `conventions fix CON505` rewrites
    literals, roles and `#:` comments. A field becomes a Google-style `Args:`, `Returns:` or
    `Raises:` section, a directive plain prose or a Markdown admonition, and a literal block a
    fenced code block with its language. Double backticks stay for code that holds a backtick.

    Scope: every text file git lists, Markdown fences excepted; `#:` comments in Python.
    """
    for path, text, markdown, python in _rst_texts(project):
        for line, message in rst_offenders(text, markdown, python):
            yield Finding(project.relative(path), line, "CON505", message)


@fixer("CON505")
def fix_inline_code(project: Project) -> Fixes:
    """Rewrite reStructuredText literals and roles as Markdown, and `#:` comments as `#`.

    Fields, directives and literal blocks need a person: they are reported, not rewritten.
    """
    rewrites: dict[Path, str] = {}
    for path, text, markdown, python in _rst_texts(project):
        if (new := rst_rewritten(text, markdown, python)) != text:
            rewrites[path] = new
    return Fixes(rewrites)
