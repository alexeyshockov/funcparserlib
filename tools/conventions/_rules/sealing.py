"""CON2xx: sealed by default."""

import ast
from collections.abc import Iterator, Mapping
from pathlib import Path

from .._engine import Finding, rule
from .._project import Project
from ._ast import bare_names, exported_by, resolved

# A class deriving from one of these is an exception by kind, and open without saying so.
# Resolved by name rather than `issubclass`, since nothing here imports the package.
BUILTIN_EXCEPTIONS = frozenset(
    {
        "BaseException",
        "Exception",
        "OSError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "ValueError",
    }
)

# The phrase a carved-out class's docstring carries, where a reader of the class will see it.
EXTENSIBLE = "Extensible by design"

# A carve-out's reason is long enough for a reader to weigh.
REASON_FLOOR = 80


def _definition(
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
                return _definition(project, target, alias.name, seen | {module})
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
            if (hit := _definition(project, module.name, name)) is not None:
                defined_in, path, node = hit
                found[f"{defined_in}.{node.name}"] = (path, node)
    return found


def exception_classes(project: Project) -> set[str]:
    """Every class in the packages that is an exception by inheritance, transitively."""
    trees = [t for module in project.modules if (t := project.tree(module.path)) is not None]
    found: set[str] = set()
    grew = True
    while grew:
        grew = False
        for tree in trees:
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name not in found
                    and bare_names(node.bases) & (BUILTIN_EXCEPTIONS | found)
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
