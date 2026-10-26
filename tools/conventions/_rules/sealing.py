"""CON2xx: sealed by default."""

import ast
import builtins
import contextlib
import importlib
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import griffe

from .._engine import Finding, rule
from .._project import Project, RunError
from ._ast import bare_names, exported_by, resolved

# A class deriving from one of these is an exception by kind, and open without saying so.
# Resolved by name rather than `issubclass`, since nothing here imports the package.
BUILTIN_EXCEPTIONS = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)

# The phrase a carved-out class's docstring carries, where a reader of the class will see it.
EXTENSIBLE = "Extensible by design"

# A carve-out's reason is long enough for a reader to weigh.
REASON_FLOOR = 80


def class_definition(
    project: Project, module: str, name: str, seen: frozenset[str] = frozenset()
) -> tuple[str, Path, ast.ClassDef] | None:
    """Where `name`, exported by `module`, is defined as a class, through any re-exports."""
    path = project.module_path(module)
    if path is None or module in seen or (tree := project.tree(path)) is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return module, path, node
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            target = resolved(module, path.name == "__init__.py", node)
            if target != module and project.module_path(target) is not None:
                return class_definition(project, target, alias.name, seen | {module})
    return None


def exported_classes(project: Project) -> Mapping[str, tuple[Path, ast.ClassDef]]:
    """Every class a public module exports, keyed by the qualified name of its definition."""
    return project.memo("exported_classes", lambda: _exported_classes(project))


def _exported_classes(project: Project) -> Mapping[str, tuple[Path, ast.ClassDef]]:
    found: dict[str, tuple[Path, ast.ClassDef]] = {}
    for module in project.modules:
        if not module.public or (tree := project.tree(module.path)) is None:
            continue
        for name in sorted(exported_by(tree)):
            if (hit := class_definition(project, module.name, name)) is not None:
                defined_in, path, node = hit
                found[f"{defined_in}.{node.name}"] = (path, node)
    return found


def _imports(tree: ast.Module) -> Mapping[str, str]:
    """The dotted path each name an absolute import binds in a module stands for."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.partition(".")[0]
                bound[alias.asname or top] = alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _dotted(base: ast.expr, bound: Mapping[str, str]) -> str | None:
    """The dotted path a base class expression names through the module's imports, if it does."""
    if isinstance(base, ast.Call):
        base = base.func
    if isinstance(base, ast.Subscript):
        base = base.value
    attrs: list[str] = []
    while isinstance(base, ast.Attribute):
        attrs.append(base.attr)
        base = base.value
    if not isinstance(base, ast.Name) or base.id not in bound:
        return None
    return ".".join([bound[base.id], *reversed(attrs)])


def _stdlib_exception(path: str) -> bool:
    # The standard library is imported to answer: it runs none of the project's code.
    module, _, name = path.rpartition(".")
    try:
        value = getattr(importlib.import_module(module), name)
    except ImportError, AttributeError, ValueError:
        return False
    return isinstance(value, type) and issubclass(value, BaseException)


def _dependency_exception(project: Project, path: str, seen: frozenset[str] = frozenset()) -> bool:
    """Whether `path`, a class outside the packages, derives from a builtin exception.

    A dependency is read from its source in `.venv` by griffe, which imports nothing.
    """
    package, _, rest = path.partition(".")
    if package == "builtins":
        return rest in BUILTIN_EXCEPTIONS
    if package in project.package_roots or path in seen:
        return False
    if package in sys.stdlib_module_names:
        return _stdlib_exception(path)
    loader = project.memo(
        "dependency_loader",
        lambda: griffe.GriffeLoader(
            search_paths=list(project.site_packages), allow_inspection=False
        ),
    )
    if package not in loader.modules_collection.members:
        # A compiled package fails to load on an alias into its extension once its stubs and
        # sources are read, and what was read stays in the collection.
        with contextlib.suppress(ImportError, griffe.GriffeError):
            loader.load(package)
    module = loader.modules_collection.members.get(package)
    if module is None:
        raise RunError(f"{path} is a base class, and griffe cannot read {package} from .venv")
    try:
        found = module[rest] if rest else module
    except KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError:
        return False
    if isinstance(found, griffe.Alias):
        return _dependency_exception(project, found.target_path, seen | {path})
    if not isinstance(found, griffe.Class):
        return False
    for base in found.bases:
        named = base.canonical_path if isinstance(base, griffe.Expr) else base
        if "." not in named:
            if named in BUILTIN_EXCEPTIONS:
                return True
        elif _dependency_exception(project, named, seen | {path}):
            return True
    return False


def exception_classes(project: Project) -> set[str]:
    """Every class in the packages that is an exception by inheritance, transitively.

    A base from outside the packages is followed into the standard library or into the dependency's
    source in `.venv`; a dependency griffe cannot read there is no verdict.
    """
    return project.memo("exception_classes", lambda: _exception_classes(project))


def _exception_classes(project: Project) -> set[str]:
    trees = [t for module in project.modules if (t := project.tree(module.path)) is not None]
    imports = [_imports(tree) for tree in trees]
    known: dict[str, bool] = {}

    def foreign(path: str) -> bool:
        if path not in known:
            known[path] = _dependency_exception(project, path)
        return known[path]

    found: set[str] = set()
    grew = True
    while grew:
        grew = False
        for tree, bound in zip(trees, imports, strict=True):
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef) or node.name in found:
                    continue
                if bare_names(node.bases) & (BUILTIN_EXCEPTIONS | found) or any(
                    (path := _dotted(base, bound)) is not None and foreign(path)
                    for base in node.bases
                ):
                    found.add(node.name)
                    grew = True
    return found


@rule("CON200")
def sealed(project: Project) -> Iterator[Finding]:
    """An exported class is `@final`, or carved out with a reason.

    Reason: a class open to subclassing makes its internals a contract with every subclass. Opening
    one is a decision; leaving `@final` off is not.

    Fix: add `@final`, or name the class in `[tool.conventions.subclassable]` with the reason and
    say "Extensible by design" in its docstring. Exceptions and Protocols are open by kind.
    """
    exceptions = exception_classes(project)
    carved_out = project.settings.get("subclassable", {})
    for qualified, (path, node) in sorted(exported_classes(project).items()):
        bases = bare_names(node.bases)
        open_by_kind = "Protocol" in bases or bool(bases & (BUILTIN_EXCEPTIONS | exceptions))
        if qualified in carved_out or open_by_kind or "final" in bare_names(node.decorator_list):
            continue
        yield Finding(
            project.relative(path),
            node.lineno,
            "CON200",
            f"exported class `{qualified}` is open to subclassing by omission",
        )


@rule("CON201", "CON202", "CON203")
def carve_outs(project: Project) -> Iterator[Finding]:
    """The carve-outs in `[tool.conventions.subclassable]` and the classes agree.

    CON201: A carve-out names a class the package exports.

    Reason: an entry left behind after its class stopped being exported reads as a decision that no
    longer exists.

    Fix: remove the entry, or export the class.

    CON202: A carve-out states a reason a reader can weigh.

    Reason: the entry is the record of why the class is open; a few words do not let the next reader
    decide whether that still holds.

    Fix: write the reason out, over 80 characters.

    CON203: A class is carved out exactly when its docstring says "Extensible by design".

    Reason: the reason in `pyproject.toml` is not where a reader of the class looks, and a docstring
    promising extension without a carve-out is a promise `@final` breaks.

    Fix: add the phrase to the carved-out class's docstring, or the carve-out for the class whose
    docstring has it.
    """
    pyproject = project.root / "pyproject.toml"
    exported = exported_classes(project)
    carved_out: Mapping[str, str] = project.settings.get("subclassable", {})
    for qualified, reason in carved_out.items():
        line = project.line_of(pyproject, f'"{qualified}"')
        if qualified not in exported:
            yield Finding(
                "pyproject.toml", line, "CON201", f"`{qualified}` is carved out but not exported"
            )
            continue
        if len(reason) <= REASON_FLOOR:
            yield Finding(
                "pyproject.toml", line, "CON202", f"`{qualified}`'s carve-out states no reason"
            )
        if EXTENSIBLE not in (ast.get_docstring(exported[qualified][1]) or ""):
            yield Finding(
                "pyproject.toml",
                line,
                "CON203",
                f'`{qualified}` is carved out, but its docstring does not say "{EXTENSIBLE}"',
            )
    for qualified, (path, node) in sorted(exported.items()):
        if qualified not in carved_out and EXTENSIBLE in (ast.get_docstring(node) or ""):
            yield Finding(
                project.relative(path),
                node.lineno,
                "CON203",
                f'`{qualified}` says "{EXTENSIBLE}" but is not carved out',
            )
