"""CON4xx: the surface a module declares, and what its names say about who uses them."""

import ast
import io
import re
import tokenize
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from .._engine import Finding, Fixes, fixer, rule
from .._project import Project
from ._ast import all_value, exported_by, resolved
from ._public import (
    FACTORY,
    INTERNAL_CONSTRUCTOR,
    Export,
    Reach,
    annotations_of,
    definition,
    exports,
    internal_constructor,
    module_level,
    pyrefly_public,
    type_params,
)
from .names import source_roots
from .sealing import class_definition, exception_classes


@rule("CON400", "CON401")
def surface(project: Project) -> Iterator[Finding]:
    """`__all__` marks a public surface, so only a public module declares one.

    CON400: A public module declares `__all__`.

    Reason: `__all__` is the module's API; without it every plain name reads as exported, and the
    visibility rules have no surface to measure against.

    Fix: add `__all__` listing what callers import from the module. A module whose body is empty
    needs none.

    CON401: An internal module declares no `__all__`.

    Reason: the underscore in the module's path already says nothing in it is API; an `__all__`
    there claims a surface the package does not offer.

    Fix: remove `__all__`; export the names from a public module instead.
    """
    for module in project.modules:
        tree = project.tree(module.path)
        if tree is None:
            continue
        value = all_value(tree)
        relative = project.relative(module.path)
        if not module.public and value is not None:
            yield Finding(
                relative,
                value.lineno,
                "CON401",
                f"{module.name} is internal: the underscore already says so",
            )
        elif module.public and tree.body and value is None:
            yield Finding(relative, 1, "CON400", f"{module.name} is public and declares no __all__")


def _all_changes(tree: ast.Module) -> Iterator[ast.stmt]:
    """The module-level statements that bind or change `__all__`."""
    for node in module_level(tree.body):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            yield node
        elif isinstance(node, ast.AnnAssign | ast.AugAssign) and (
            getattr(node.target, "id", None) == "__all__"
        ):
            yield node
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(func := node.value.func, ast.Attribute)
            and getattr(func.value, "id", None) == "__all__"
        ):
            yield node


@rule("CON409", "CON415")
def declared_surface(project: Project) -> Iterator[Finding]:
    """A public module's `__all__` is read from its source, and pyrefly reads the same surface.

    CON409: A public module binds `__all__` once, to a literal list or tuple of strings.

    Reason: every rule that asks what a module exports reads `__all__` from source, without
    importing the package, and so does pyrefly's public-API report. A name added by `+=`,
    `.extend()` or a computed value is missed by one reader or both, silently: pyrefly counts every
    plain name of a module whose `__all__` it cannot evaluate, and no imported one.

    Fix: list every exported name in the one `__all__ = [...]`.

    CON415: Every definition a public module exports is one pyrefly counts as public, and the other
    way round.

    Reason: `just type-coverage` gates the public API on pyrefly's list of it, and every other rule
    on the names `__all__` exports, resolved to their definitions. A definition only one of them
    counts is checked by half the gates.

    Fix: read which side is right. A name re-exported in a way one reader does not follow (a star
    import, an import inside a function) is spelled as a plain `from ... import`; a reader that is
    wrong goes to the backlog with the case that shows it.

    Scope: definitions in the project's packages. A re-exported submodule, and a name defined
    outside the packages, are not definitions pyrefly lists.
    """
    for module in project.modules:
        if not module.public or (tree := project.tree(module.path)) is None:
            continue
        changes = list(_all_changes(tree))
        literal = (
            len(changes) == 1
            and isinstance(changes[0], ast.Assign | ast.AnnAssign)
            and isinstance(value := changes[0].value, ast.List | ast.Tuple)
            and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts)
        )
        if changes and not literal:
            yield Finding(
                project.relative(module.path),
                changes[-1].lineno,
                "CON409",
                f"{module.name} builds __all__ instead of listing it once, literally",
            )
    exported: dict[str, Export] = {}
    for export in exports(project):
        if export.definition is not None and export.definition.in_packages:
            exported.setdefault(export.definition.qualified, export)
    counted = pyrefly_public(project)
    for qualified, export in sorted(exported.items()):
        if qualified not in counted:
            yield Finding(
                project.relative(export.path),
                export.line,
                "CON415",
                f"{export.module} exports {export.name}, defined as {qualified}, which pyrefly"
                " does not count as public",
            )
    for qualified in sorted(counted - exported.keys()):
        module, _, name = qualified.rpartition(".")
        path = project.module_path(module)
        site = definition(project, module, name) if path is not None else None
        yield Finding(
            project.relative(path) if path is not None else module,
            site.node.lineno if site is not None and site.node is not None else None,
            "CON415",
            f"pyrefly counts {qualified} as public, and no public module's __all__ exports it",
        )


# --- The classes a public signature hands out ---


@rule("CON416", "CON417")
def reached_classes(project: Project) -> Iterator[Finding]:
    """A class a public signature hands out is one a caller can import.

    CON416: Every class a public signature names is exported by a public module.

    Reason: a caller who is handed a value, or has to pass one, annotates with its type. A class
    only a private module defines is one the caller cannot name without a private import, and one
    the API reference and the API diff do not show.

    Fix: export the class from the public module whose API hands it out. A constructor users never
    call says so instead (CON417). A type alias is read through, so the classes its value names
    count; the alias itself need not be exported. A private base class is not a signature.

    Scope: the parameters and returns of every exported function, and of every public method of an
    exported class, its bases' in the packages included; the annotations of their public attributes;
    and, in turn, those of every class reported here.

    CON417: A class whose docstring says "its constructor is internal" names, in that paragraph, a
    public factory that returns it.

    Reason: users get such a class from the factory, so its constructor's parameters are not part of
    the API; CON416 and the API diff skip them. The factory is what makes the sentence true.

    Fix: write `` `connect()` `` (or `` `Pool.lease()` ``) in the sentence: a function a public
    module exports, or a public method of an exported class, whose return annotation names the
    class. With no such factory, export the classes the constructor names instead.
    """
    exported = {
        site.qualified: f"{e.module}.{e.name}"
        for e in reversed(exports(project))
        if (site := e.definition) is not None and site.in_packages
    }
    reach = Reach(project, exported)
    pending = [e.definition for e in exports(project) if e.definition and e.definition.in_packages]
    visited: set[str] = set()
    reported: set[tuple[str, int, str]] = set()
    while pending:
        site = pending.pop(0)
        if site.qualified in visited or site.path is None or site.node is None:
            continue
        visited.add(site.qualified)
        if isinstance(site.node, ast.ClassDef):
            mark = internal_constructor(site.node)
            if mark is not None and not any(
                reach.factory_returns(site, f) for f in FACTORY.findall(mark)
            ):
                yield Finding(
                    project.relative(site.path),
                    site.node.lineno,
                    "CON417",
                    f"{reach.spelling(site)} says {INTERNAL_CONSTRUCTOR!r}, and its docstring"
                    " names no public factory returning it",
                )
            members = [(site.module, site.node, type_params(site.node), ""), *reach.members(site)]
        else:
            members = [(site.module, site.node, frozenset(), "")]
        owner = reach.spelling(site)
        for module, member, params, label in members:
            if (
                isinstance(site.node, ast.ClassDef)
                and internal_constructor(site.node) is not None
                and label in ("__init__", "__new__")
            ):
                continue
            scope = params | (
                type_params(member)
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
                else frozenset()
            )
            where = owner if member is site.node else f"{owner}.{label}"
            path = project.module_path(module)
            for annotation in annotations_of(member):
                for named in reach.named(module, annotation, scope):
                    pending.append(named)
                    key = (module, member.lineno, named.qualified)
                    if named.qualified in exported or key in reported:
                        continue
                    reported.add(key)
                    yield Finding(
                        project.relative(path) if path is not None else module,
                        member.lineno,
                        "CON416",
                        f"{where} names {named.qualified}, which no public module exports",
                    )


# --- The exceptions a public module adopts ---

ADOPT = "adopt_exceptions(__name__, __all__, globals())"


def _adopting_call(tree: ast.Module) -> ast.Expr | None:
    """The top-level `adopt_exceptions(__name__, __all__, globals())` statement, if there is one."""
    for node in tree.body:
        if isinstance(node, ast.Expr) and ast.unparse(node.value) == ADOPT:
            return node
    return None


@rule("CON408")
def adopted_exceptions(project: Project) -> Iterator[Finding]:
    """A public module names the exceptions it exports from private modules after itself.

    Reason: a traceback, a class repr and a pickle name a class by its `__module__`. An exception
    defined in a private module and left there reads as that module's, and the path a caller copies
    from a traceback into an `except` is a private import.

    Fix: end the module with `adopt_exceptions(__name__, __all__, globals())`, imported from the
    package's `_commons`; only a `del` may follow it. It adopts the exceptions defined in a private
    module below this one, so one re-exported from a sibling or another public module is not
    counted.
    """
    exceptions = exception_classes(project)
    for module in project.modules:
        if not module.public or (tree := project.tree(module.path)) is None:
            continue
        prefix = f"{module.name}."
        private = sorted(
            name
            for name in exported_by(tree)
            if (hit := class_definition(project, module.name, name)) is not None
            and hit[2].name in exceptions
            and hit[0].startswith(prefix)
            and any(part.startswith("_") for part in hit[0].removeprefix(prefix).split("."))
        )
        if not private:
            continue
        call = _adopting_call(tree)
        if call is not None and all(
            isinstance(node, ast.Delete) for node in tree.body[tree.body.index(call) + 1 :]
        ):
            continue
        anchor = call or all_value(tree)
        yield Finding(
            project.relative(module.path),
            anchor.lineno if anchor is not None else 1,
            "CON408",
            f"{module.name} exports {', '.join(private)} from private modules and does not end"
            f" with {ADOPT}",
        )


# --- The version a package states ---

# A public PEP 440 version: release, optional pre-release, post-release and dev-release segments.
PEP440 = re.compile(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?")
METADATA = "importlib.metadata"


def _version_from_metadata(node: ast.stmt) -> tuple[ast.Try, ast.Constant | None] | None:
    """`node` if it is the `try` reading `__version__` from metadata, with its fallback."""
    if not isinstance(node, ast.Try):
        return None
    reads = any(
        isinstance(stmt, ast.AnnAssign)
        and getattr(stmt.target, "id", None) == "__version__"
        and getattr(stmt.annotation, "id", None) == "str"
        and isinstance(stmt.value, ast.Call)
        and getattr(stmt.value.func, "id", getattr(stmt.value.func, "attr", None)) == "version"
        for stmt in node.body
    )
    if not reads:
        return None
    for handler in node.handlers:
        if (
            getattr(handler.type, "id", getattr(handler.type, "attr", None))
            != "PackageNotFoundError"
        ):
            continue
        for stmt in handler.body:
            target = (
                stmt.targets[0] if isinstance(stmt, ast.Assign) else getattr(stmt, "target", None)
            )
            value = getattr(stmt, "value", None)
            if getattr(target, "id", None) == "__version__" and isinstance(value, ast.Constant):
                return node, value
    return node, None


def _imports_version(tree: ast.Module) -> ast.ImportFrom | None:
    """The `from ._version import __version__` an `__init__.py` re-exports, if it has one."""
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "_version"
            and any(alias.name == "__version__" for alias in node.names)
        ):
            return node
    return None


@rule("CON405", "CON406", "CON407")
def version(project: Project) -> Iterator[Finding]:
    """A package states its version once, read from its installed metadata.

    CON405: A package's `_version.py` reads `__version__: str` from `importlib.metadata.version` in
    a `try` and falls back to a constant on `PackageNotFoundError`; its `__init__.py` imports it
    from there and exports it.

    Reason: callers and tools read `__version__`; one shape in every package is one thing to know,
    and the metadata is the only place the version is written down. It lives in a private module so
    the package's own modules can import it without going through the package root (CON506).

    Fix: `_version.py` holds `try: __version__: str = version("dist-name")` and `except
    PackageNotFoundError: __version__ = "0.0.0.dev0"`; `__init__.py` has `from ._version import
    __version__` and `"__version__"` in `__all__`.

    CON406: The fallback is a valid PEP 440 version.

    Reason: a source checkout has no installed metadata, and a fallback such as `"dev"` is not a
    version: `packaging.version.Version("dev")` raises in whatever parses it.

    Fix: use `"0.0.0.dev0"`.

    CON407: No other module of the package reads the version from metadata.

    Reason: the version is resolved once; a second lookup is a second answer to keep in step.

    Fix: import `__version__` from the package's `_version` module instead.
    """
    homes = set()
    for package, root in project.package_roots.items():
        init, home = root / "__init__.py", root / "_version.py"
        homes.add(home)
        init_tree = project.tree(init)
        home_tree = project.tree(home) if home.is_file() else None
        if init_tree is None:
            continue
        relative = project.relative(init)
        found = (
            None
            if home_tree is None
            else next(filter(None, map(_version_from_metadata, home_tree.body)), None)
        )
        if found is None:
            yield Finding(
                project.relative(home) if home_tree is not None else relative,
                1,
                "CON405",
                f"{package}._version does not read `__version__: str` from importlib.metadata",
            )
        elif found[1] is None:
            yield Finding(
                project.relative(home),
                found[0].lineno,
                "CON405",
                f"{package}._version sets no constant `__version__` on PackageNotFoundError",
            )
        elif not (isinstance(found[1].value, str) and PEP440.fullmatch(found[1].value)):
            yield Finding(
                project.relative(home),
                found[1].lineno,
                "CON406",
                f"{package}'s fallback `__version__` {found[1].value!r} is not a PEP 440 version",
            )
        imported = _imports_version(init_tree)
        if imported is None or "__version__" not in exported_by(init_tree):
            yield Finding(
                relative,
                imported.lineno if imported is not None else 1,
                "CON405",
                f"{package} does not re-export `__version__` from its `_version` module",
            )
    for module in project.modules:
        if module.path in homes:
            continue
        tree = project.tree(module.path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == METADATA
                and any(alias.name == "version" for alias in node.names)
            ):
                yield Finding(
                    project.relative(module.path),
                    node.lineno,
                    "CON407",
                    f"{module.name} reads the version from metadata; import `__version__`",
                )


# --- Visibility ---
#
# Three levels, and a name says which. Public: what a public module's `__all__` exports. Package:
# what another module of the package imports, what a public or package-level signature names, and
# what an entry point names. Module: everything else, which carries a leading underscore. In a
# public module a name is exported or underscored, and a helper other modules need lives in a
# private module. ALL_CAPS constants are exempt, and so is every type variable, whatever its
# spelling, since it stands in for a signature's syntax. Imports from tests do not count.

EXEMPT = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
TYPE_VARIABLES = frozenset({"TypeVar", "ParamSpec", "TypeVarTuple"})


def is_private(module_name: str) -> bool:
    return any(
        part.startswith("_") and not part.startswith("__") for part in module_name.split(".")[1:]
    )


def is_type_variable(node: ast.stmt) -> bool:
    """Whether a statement binds a type variable: `T = TypeVar("T")` and its kin."""
    value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    return (func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")) in TYPE_VARIABLES


def exempt(name: str, node: ast.stmt) -> bool:
    return bool(EXEMPT.match(name)) or is_type_variable(node)


def top_level(tree: ast.Module) -> dict[str, ast.stmt]:
    """The module-level names a module defines, each with the statement that defines it."""
    found: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
        elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            found[node.name.id] = node
    imported = imported_at_module_level(tree)
    return {
        name: node
        for name, node in found.items()
        if not name.startswith("__") and name not in imported
    }


def imported_at_module_level(tree: ast.Module) -> set[str]:
    """Names an import binds outside any function or class, `try` and `if` blocks included.

    `sniffio: Any` beside `import sniffio` declares an import, not a definition.
    """
    found: set[str] = set()
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import | ast.ImportFrom):
            found |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            pending.extend(ast.iter_child_nodes(node))
    return found


def public_member(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def names_in(node: ast.AST) -> set[str]:
    """Every name an expression mentions, string annotations included."""
    found = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            found |= set(re.findall(r"[A-Za-z_]\w*", n.value))
    return found


def signature(node: ast.stmt) -> set[str]:
    """What a definition's public face names: arguments, returns, bases, public attributes."""
    found: set[str] = set()
    if isinstance(node, ast.ClassDef):
        for base in [*node.bases, *(k.value for k in node.keywords)]:
            found |= names_in(base)
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and public_member(
                item.name
            ):
                found |= signature(item)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if not item.target.id.startswith("_"):
                    found |= names_in(item.annotation)
    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = node.args
        for argument in [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        ]:
            if argument is not None and argument.annotation is not None:
                found |= names_in(argument.annotation)
        if node.returns is not None:
            found |= names_in(node.returns)
    elif isinstance(node, ast.AnnAssign):
        # The annotation is the public type; the value is too only for an alias.
        found |= names_in(node.annotation)
        if node.value is not None and ast.unparse(node.annotation).endswith("TypeAlias"):
            found |= names_in(node.value)
    elif isinstance(node, ast.TypeAlias):
        found |= names_in(node.value)
    elif isinstance(node, ast.Assign) and (
        not isinstance(node.value, ast.Call) or is_type_variable(node)
    ):
        # `Alias = A | B` names types a caller writes, and a type variable's bound is what a caller
        # must satisfy; `CONSTANT = Thing(...)` names none.
        found |= names_in(node.value)
    return found


def entry_points(project: Project) -> set[tuple[str, str]]:
    """Every (module, name) an entry point in a package's `pyproject.toml` or the root's names."""
    found: set[tuple[str, str]] = set()
    for pyproject in {
        project.root / "pyproject.toml",
        *(root.parent / "pyproject.toml" for root in project.package_roots.values()),
    }:
        if not pyproject.is_file():
            continue
        with pyproject.open("rb") as f:
            table = tomllib.load(f).get("project", {})
        groups = [
            table.get("scripts", {}),
            table.get("gui-scripts", {}),
            *table.get("entry-points", {}).values(),
        ]
        for group in groups:
            for value in group.values():
                module, _, attribute = value.partition(":")
                if attribute:
                    found.add((module.strip(), attribute.strip().split(".")[0]))
    return found


def imported_names(
    modules: Mapping[str, Path], trees: Mapping[str, ast.Module]
) -> dict[str, set[str]]:
    """For each module, the names another module of the package imports from it or reads off it.

    A submodule named in `from package import module` is a module, not a name the package defines.
    """
    imported: dict[str, set[str]] = {name: set() for name in modules}
    for name, tree in trees.items():
        is_package = modules[name].name == "__init__.py"
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                source = resolved(name, is_package, node)
                for alias in node.names:
                    submodule = f"{source}.{alias.name}"
                    if submodule in modules:
                        aliases[alias.asname or alias.name] = submodule
                    elif source in imported and source != name:
                        imported[source].add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                source = aliases.get(node.value.id)
                if source is not None and source != name:
                    imported[source].add(node.attr)
    return imported


def shared_names(
    modules: Mapping[str, Path],
    trees: Mapping[str, ast.Module],
    extra: set[tuple[str, str]],
) -> dict[str, set[str]]:
    """For each module, the names it defines that are more than its own: package or public."""
    shared: dict[str, set[str]] = {name: set() for name in modules}
    imported = imported_names(modules, trees)
    for name in modules:
        shared[name] |= imported[name]
    for module, name in extra:
        if module in shared:
            shared[module].add(name)
    for name, tree in trees.items():
        if not is_private(name):
            shared[name] |= exported_by(tree)
    grew = True
    while grew:
        grew = False
        for name, tree in trees.items():
            defined = top_level(tree)
            for owned in list(shared[name] & set(defined)):
                for mentioned in signature(defined[owned]) & set(defined):
                    if mentioned not in shared[name]:
                        shared[name].add(mentioned)
                        grew = True
    return shared


def template_owned(module_name: str) -> bool:
    """`<package>._commons` is the generated helpers module, kept whole, used or not."""
    return module_name.partition(".")[2] == "_commons"


@rule("CON402", "CON403", "CON404")
def visibility(project: Project) -> Iterator[Finding]:
    """A name says who uses it: its module, the package, or the public.

    CON402: A name only its own module uses carries a leading underscore.

    Reason: a plain name reads as something other modules may import; the underscore is what lets
    the next reader change it without searching the package.

    Fix: rename `helper` to `_helper` in the module and wherever it is mentioned.

    CON403: A name another module needs carries no leading underscore.

    Reason: a name other modules import, that a shared signature names, or that an entry point
    names, is package-level whatever it is spelled; the underscore misstates it. In a public module
    an underscored name that other code needs is API `__all__` forgot.

    Fix: rename `_helper` to `helper`, and export it if its module is public.

    CON404: A public module's plain names that other code needs are in its `__all__`.

    Reason: a public module's names are its API; a helper other modules need, kept there unexported,
    is API by accident.

    Fix: add the name to `__all__` if it is API, or move it to a private module.

    Scope: the modules of each package. ALL_CAPS constants and type variables are exempt; the
    generated helpers module `<package>._commons` is exempt from CON402 and CON403. Imports from
    tests do not make a name shared; entry points in `[project.scripts]`, `gui-scripts` and
    `entry-points` do.
    """
    analysis = _analysis(project)
    for name, tree in analysis.trees.items():
        relative = project.relative(analysis.modules[name])
        exported = exported_by(tree)
        renames = _renames(name, tree, analysis.shared[name])
        for defined, node in top_level(tree).items():
            if defined in renames:
                new, code = renames[defined]
                yield _rename(relative, node, code, defined, new)
            if (
                not exempt(defined, node)
                and not is_private(name)
                and not defined.startswith("_")
                and defined not in exported
                and defined in analysis.shared[name]
            ):
                yield Finding(
                    relative,
                    node.lineno,
                    "CON404",
                    f"`{name}.{defined}` is needed by other code but not in `__all__`",
                )


@dataclass(frozen=True, slots=True)
class _Analysis:
    modules: Mapping[str, Path]
    trees: Mapping[str, ast.Module]
    shared: Mapping[str, set[str]]


def _analysis(project: Project) -> _Analysis:
    """The package's modules, their trees, and each module's shared names."""

    def compute() -> _Analysis:
        modules = {module.name: module.path for module in project.modules}
        trees = {
            name: tree for name, path in modules.items() if (tree := project.tree(path)) is not None
        }
        return _Analysis(modules, trees, shared_names(modules, trees, entry_points(project)))

    return project.memo("visibility", compute)


def _renames(name: str, tree: ast.Module, shared: set[str]) -> dict[str, tuple[str, str]]:
    """The renames visibility asks of one module: old name to (new name, code)."""
    found: dict[str, tuple[str, str]] = {}
    if template_owned(name):
        return found
    for defined, node in top_level(tree).items():
        if exempt(defined, node):
            continue
        if defined.startswith("_") and defined in shared:
            found[defined] = (defined.lstrip("_"), "CON403")
        elif not defined.startswith("_") and defined not in shared:
            found[defined] = ("_" + defined, "CON402")
    return found


def _rename(relative: str, node: ast.stmt, code: str, old: str, new: str) -> Finding:
    reason = "only its module uses it" if code == "CON402" else "other code needs it"
    return Finding(relative, node.lineno, code, f"`{old}` -> `{new}`: {reason}")


# --- The renamer ---

NAMING_CALLS = frozenset({"TypeVar", "ParamSpec", "TypeVarTuple", "NewType"})
ATTRIBUTE_CALLS = frozenset({"setattr", "getattr", "hasattr", "delattr"})


def _called(node: ast.Call) -> str:
    """The name a call is spelled with: `f` for `f(...)`, `f` for `obj.f(...)`."""
    func = node.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")


def string_references(
    tree: ast.Module, defining: bool, aliases: set[str]
) -> dict[tuple[int, int], str]:
    """Where a string literal refers to a name: `expression` or `exact`, keyed by its position.

    An expression string is an annotation, a `TypeAlias` value, or `cast`'s first argument; the
    names inside it are references. An exact string is `TypeVar("T")` and its kin, or the name in
    `setattr(module, "NAME", ...)` and its kin when the object is the module. Every other string, a
    dictionary key, a message, a path, is text.
    """
    found: dict[tuple[int, int], str] = {}

    def mark(node: ast.AST | None, kind: str) -> None:
        for sub in ast.walk(node) if node is not None else ():
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                found[(sub.lineno, sub.col_offset)] = kind

    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            mark(node.annotation, "expression")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            mark(node.returns, "expression")
        elif isinstance(node, ast.AnnAssign):
            mark(node.annotation, "expression")
            if ast.unparse(node.annotation).endswith("TypeAlias"):
                mark(node.value, "expression")
        elif isinstance(node, ast.TypeAlias):
            mark(node.value, "expression")
        elif isinstance(node, ast.Call) and node.args:
            name = _called(node)
            if name == "cast":
                mark(node.args[0], "expression")
            elif name in NAMING_CALLS:
                mark(node.args[0], "exact")
            elif name in ATTRIBUTE_CALLS and len(node.args) > 1:
                target = node.args[0]
                if isinstance(target, ast.Name) and (target.id in aliases or defining):
                    mark(node.args[1], "exact")
    return found


def renamed(source: str, renames: Mapping[str, str], bare: set[str], aliases: set[str]) -> str:
    """`source` with each name in `renames` rewritten where it means that name.

    A bare name token in `bare`; an attribute token after one of `aliases` (a module bound to a
    name); a string literal that `string_references` says is a reference; and a backticked mention
    in a comment or a docstring.
    """
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    references = string_references(ast.parse(source), bare == set(renames), aliases)
    edits: list[tuple[int, int, str]] = []
    previous: list[tokenize.TokenInfo] = []
    # `import a.b` and the `a.b` of `from a.b import c` are module paths; the `c` is a name.
    module_path = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in ("import", "from"):
            starts = not previous or previous[-1].type in (
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            )
            if starts:
                module_path = True
            elif token.string == "import":
                module_path = False
        elif token.type == tokenize.NEWLINE:
            module_path = False
        start = offsets[token.start[0] - 1] + token.start[1]
        end = offsets[token.end[0] - 1] + token.end[1]
        after_dot = bool(previous) and previous[-1].string == "."
        if token.type == tokenize.NAME and token.string in renames and not module_path:
            owner = previous[-2].string if len(previous) > 1 else ""
            if (not after_dot and token.string in bare) or (after_dot and owner in aliases):
                edits.append((start, end, renames[token.string]))
        elif token.type in (tokenize.STRING, tokenize.COMMENT):
            text = token.string
            kind = references.get(token.start) if token.type == tokenize.STRING else None
            if kind == "exact" and text[1:-1] in renames:
                text = text[0] + renames[text[1:-1]] + text[-1]
            elif kind == "expression":
                text = re.sub(
                    r"(?<![\w.])([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?",
                    lambda m: _expression_name(m, renames, bare, aliases),
                    text,
                )
            for old, new in renames.items():
                text = re.sub(rf"`{re.escape(old)}\b", f"`{new}", text)
            if text != token.string:
                edits.append((start, end, text))
        if token.type not in (tokenize.NL, tokenize.COMMENT):
            previous = [*previous[-1:], token]
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


def _expression_name(
    match: re.Match[str], renames: Mapping[str, str], bare: set[str], aliases: set[str]
) -> str:
    """One name, or `alias.name`, inside an expression string, renamed where it is ours."""
    head, attribute = match.group(1), match.group(2)
    if attribute is None:
        return renames[head] if head in bare else head
    if head in aliases and attribute in renames:
        return f"{head}.{renames[attribute]}"
    if head in bare:
        return f"{renames[head]}.{attribute}"
    return match.group(0)


def collisions(source: str, renames: Mapping[str, str]) -> list[str]:
    """Renames that would shadow or collide: the new name bound where the old one is used."""
    tree = ast.parse(source)
    found = [
        f"{old} -> {new}: {new} is already defined here"
        for old, new in renames.items()
        if new in top_level(tree)
    ]
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        arguments = node.args
        bound = {
            a.arg
            for a in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            ]
            if a is not None
        }
        bound |= {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        used = {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        found += [
            f"{old} -> {new}: `{getattr(node, 'name', 'lambda')}` binds {new} and uses {old}"
            for old, new in renames.items()
            if old in used and new in bound
        ]
    return found


def uses(
    source: str,
    reader: Path,
    module_name: str,
    renames: Mapping[str, str],
    modules: Mapping[str, Path],
) -> tuple[set[str], set[str]]:
    """The renamed names `reader` imports from `module_name`, and names it binds the module to."""
    name = next((n for n, p in modules.items() if p == reader), None)
    is_package = reader.name == "__init__.py"
    parent, _, leaf = module_name.rpartition(".")
    bare: set[str] = set()
    aliases: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            target = resolved(name, is_package, node) if name is not None else node.module or ""
            if target == module_name:
                bare |= {alias.name for alias in node.names if alias.name in renames}
            if target == parent:
                aliases |= {
                    alias.asname or alias.name for alias in node.names if alias.name == leaf
                }
        elif isinstance(node, ast.Import):
            aliases |= {
                alias.asname for alias in node.names if alias.name == module_name and alias.asname
            }
    return bare, aliases


def bound_by_imports(source: str) -> set[str]:
    return {
        (alias.asname or alias.name).split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }


@fixer("CON402", "CON403")
def fix_visibility(project: Project) -> Fixes:
    """Apply the CON402 and CON403 renames across each package and the tests and examples beside it.

    Each module's renames are applied to the module and to every reader that imports a renamed name
    from it or binds the module to a name. A module whose renames would collide with a name already
    bound is refused whole, and none of its files change.
    """
    analysis = _analysis(project)
    readers = sorted(project.under(source_roots(project), frozenset({".py"})))
    current: dict[Path, str] = {}

    def text(path: Path) -> str:
        if path not in current:
            current[path] = project.text(path) or ""
        return current[path]

    refused: list[str] = []
    for module_name, tree in sorted(analysis.trees.items()):
        renames = {
            old: new
            for old, (new, _) in _renames(module_name, tree, analysis.shared[module_name]).items()
        }
        if not renames:
            continue
        path = analysis.modules[module_name]
        problems = [
            f"{project.relative(path)}: {problem}" for problem in collisions(text(path), renames)
        ]
        rewrites = {path: renamed(text(path), renames, set(renames), set())}
        for reader in readers:
            if reader == path or project.tree(reader) is None:
                continue
            source = text(reader)
            bare, aliases = uses(source, reader, module_name, renames, analysis.modules)
            if not (bare or aliases):
                continue
            taken = set(top_level(ast.parse(source))) | bound_by_imports(source)
            problems += [
                f"{project.relative(reader)}: {old} -> {new}: {new} is already a name there"
                for old, new in renames.items()
                if old in bare and new in taken
            ]
            rewrites[reader] = renamed(source, renames, bare, aliases)
        if problems:
            refused += [f"refused, rename by hand: {problem}" for problem in problems]
            continue
        current.update(rewrites)
    changed = {path: new for path, new in current.items() if new != project.text(path)}
    return Fixes(changed, refused)
