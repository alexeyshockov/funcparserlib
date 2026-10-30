"""The public surface: what each public module's `__all__` exports, where each name is defined, and
pyrefly's list of the same definitions.
"""

import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .._project import Project, RunError
from .._toolbox import INTERPRETER, distributions, floor
from ._ast import all_value, bare_names, resolved

# The `coverage report` layouts this module reads.
PYREFLY_SCHEMAS = frozenset({"0.2"})


@dataclass(frozen=True, slots=True)
class Definition:
    """Where a name is bound: a statement in the project's packages, a submodule, or elsewhere."""

    module: str
    name: str
    path: Path | None
    """The defining module's file; `None` outside the project's packages."""
    node: ast.stmt | None
    """The statement binding the name; `None` for a submodule or a name defined outside."""

    @property
    def qualified(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def in_packages(self) -> bool:
        return self.node is not None


@dataclass(frozen=True, slots=True)
class Export:
    """One name a public module's `__all__` lists, and what it stands for."""

    module: str
    name: str
    path: Path
    line: int
    """The line of the name inside `__all__`."""
    definition: Definition | None
    """`None` when the module binds no such name."""


def module_level(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements that run when a module is imported, inside `if` and `try` blocks included."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            yield from module_level(node.body)
            yield from module_level(node.orelse)
        elif isinstance(node, ast.Try | ast.TryStar):
            yield from module_level(node.body)
            for handler in node.handlers:
                yield from module_level(handler.body)
            yield from module_level(node.orelse)
            yield from module_level(node.finalbody)


def _binds(node: ast.stmt, name: str) -> bool:
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name == name
    if isinstance(node, ast.TypeAlias):
        return isinstance(node.name, ast.Name) and node.name.id == name
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    return False


def definition(project: Project, module: str, name: str) -> Definition | None:
    """Where `name`, as `module` binds it, is defined: through `from ... import` re-exports, those
    under `if TYPE_CHECKING:` included. A name no module on the way binds is the submodule of that
    name, or `None` when there is none.
    """
    seen: set[tuple[str, str]] = set()
    while (module, name) not in seen:
        seen.add((module, name))
        path = project.module_path(module)
        if path is None:
            return Definition(module, name, None, None)
        if (tree := project.tree(path)) is None:
            return None
        imported: tuple[str, str] | None = None
        for node in module_level(tree.body):
            if _binds(node, name):
                return Definition(module, name, path, node)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        imported = (resolved(module, path.name == "__init__.py", node), alias.name)
        if imported is None:
            break
        module, name = imported
    submodule = project.module_path(f"{module}.{name}")
    return None if submodule is None else Definition(module, name, submodule, None)


def exports(project: Project) -> Sequence[Export]:
    """Every name a public module's literal `__all__` lists, in module order."""
    return project.memo("exports", lambda: _exports(project))


def _exports(project: Project) -> list[Export]:
    found: list[Export] = []
    for module in project.modules:
        if not module.public or (tree := project.tree(module.path)) is None:
            continue
        value = all_value(tree)
        if not isinstance(value, ast.List | ast.Tuple):
            continue
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                found.append(
                    Export(
                        module.name,
                        element.value,
                        module.path,
                        element.lineno,
                        definition(project, module.name, element.value),
                    )
                )
    return found


def pyrefly_public(project: Project) -> frozenset[str]:
    """The top-level definitions pyrefly counts as public, by the module that defines them.

    `pyrefly coverage report --public-only`, the report `just type-coverage` gates on, at the
    interpreter and floor the toolbox gives it. Methods and attributes are left out.

    Raises:
        RunError: pyrefly cannot run, or reports in a layout this module does not read.
    """
    return project.memo("pyrefly_public", lambda: _pyrefly_public(project))


def _pyrefly_public(project: Project) -> frozenset[str]:
    if not exports(project):
        # Nothing to compare: a public module without `__all__` is CON400's.
        return frozenset()
    roots = [project.relative(root) for root in project.package_roots.values()]
    command = [str(Path(sys.executable).parent / "pyrefly"), "coverage", "report", "--public-only"]
    if (project.root / INTERPRETER).is_file():
        command += ["--python-interpreter-path", INTERPRETER]
        command += ["--python-version", floor(project.root, distributions(project.root))]
    try:
        done = subprocess.run(
            [*command, *roots], cwd=project.root, capture_output=True, text=True, check=False
        )
    except OSError as e:
        raise RunError(f"cannot run pyrefly from {command[0]}: {e}") from None
    try:
        report = json.loads(done.stdout)
    except json.JSONDecodeError:
        tail = (done.stderr or done.stdout).strip().splitlines()[-5:]
        raise RunError(f"pyrefly coverage report gave no report: {' / '.join(tail)}") from None
    if (schema := report.get("schema_version")) not in PYREFLY_SCHEMAS:
        raise RunError(
            f"pyrefly's coverage report is schema {schema!r}; the runner reads"
            f" {', '.join(sorted(PYREFLY_SCHEMAS))}"
        )
    found: set[str] = set()
    for module in report["module_reports"]:
        prefix = f"{module['name']}."
        found.update(
            name
            for name in module["names"]
            if name.startswith(prefix) and "." not in name.removeprefix(prefix)
        )
    return frozenset(found)


def _dotted(node: ast.expr) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(parts)]


def spelled(node: ast.expr) -> Iterator[list[str]]:
    """The dotted names an annotation spells, each whole, a string annotation's included.

    `Literal[...]` holds values, not names, and only the first argument of `Annotated[...]` is a
    type.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            parsed = ast.parse(node.value.strip(), mode="eval").body
        except SyntaxError:
            return
        yield from spelled(parsed)
        return
    if (parts := _dotted(node)) is not None:
        yield parts
        return
    if isinstance(node, ast.Subscript) and (parts := _dotted(node.value)) is not None:
        yield parts
        if parts[-1] == "Literal":
            return
        if parts[-1] == "Annotated" and isinstance(node.slice, ast.Tuple) and node.slice.elts:
            yield from spelled(node.slice.elts[0])
            return
        yield from spelled(node.slice)
        return
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            yield from spelled(child)


def resolve(project: Project, module: str, parts: Sequence[str]) -> Definition | None:
    """What the dotted name `parts`, spelled in `module`, is defined as; `None` for an attribute of
    a class or of anything that is not a module.
    """
    site = definition(project, module, parts[0])
    for attr in parts[1:]:
        if site is None or site.path is None or site.node is not None:
            return None
        site = definition(project, site.qualified, attr)
    return site


# --- What a public signature reaches ---

# The sentence a class's docstring carries when users get it from a factory, never by calling it.
INTERNAL_CONSTRUCTOR = "its constructor is internal"
FACTORY = re.compile(r"`([A-Za-z_][\w.]*)\(\)`")


def public_member(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def internal_constructor(node: ast.ClassDef) -> str | None:
    """The paragraph of the class's docstring that says its constructor is internal, if one does."""
    doc = ast.get_docstring(node) or ""
    return next((p for p in doc.split("\n\n") if INTERNAL_CONSTRUCTOR in " ".join(p.split())), None)


@dataclass(frozen=True, slots=True)
class Reach:
    """What a public signature reaches, from the definitions the public modules export."""

    project: Project
    exported: Mapping[str, str]
    """Each exported definition's qualified name, and the public name it is imported by."""

    def spelling(self, site: Definition) -> str:
        return self.exported.get(site.qualified, site.qualified)

    def members(self, site: Definition) -> Iterator[tuple[str, ast.stmt, frozenset[str], str]]:
        """A class's public members, those of its bases in the packages included, each with its
        module, the type parameters in scope and the name it is reached by; a member a nearer class
        defines hides a base's. A dataclass's private fields are its generated `__init__`'s.
        """
        assert isinstance(site.node, ast.ClassDef)
        seen: set[str] = set()
        pending = [site]
        done: set[str] = set()
        while pending:
            cls = pending.pop(0)
            if cls.qualified in done or not isinstance(node := cls.node, ast.ClassDef):
                continue
            done.add(cls.qualified)
            params = type_params(node)
            generated = generates_init(node) and not any(
                member_name(stmt) == "__init__" for stmt in node.body
            )
            for stmt in node.body:
                name = member_name(stmt)
                if name is None:
                    continue
                if public_member(name):
                    if not (isinstance(stmt, ast.AnnAssign) and name in seen):
                        seen.add(name)
                        yield cls.module, stmt, params, name
                elif generated and "__init__" not in seen and init_field(stmt):
                    yield cls.module, stmt, params, "__init__"
            if generated:
                seen.add("__init__")
            for base in node.bases:
                parts = next(iter(spelled(base)), None)
                if parts is not None and (found := resolve(self.project, cls.module, parts)):
                    if found.in_packages:
                        pending.append(found)

    def named(
        self, module: str, annotation: ast.expr | None, params: frozenset[str]
    ) -> Iterator[Definition]:
        """The definitions in the packages `annotation` names, a type alias's value read through."""
        pending = [] if annotation is None else [(module, annotation, params)]
        aliases: set[str] = set()
        while pending:
            module, expr, scope = pending.pop()
            for parts in spelled(expr):
                if len(parts) == 1 and parts[0] in scope:
                    continue
                site = resolve(self.project, module, parts)
                if site is None or not site.in_packages:
                    continue
                if isinstance(site.node, ast.ClassDef):
                    yield site
                elif (value := alias_value(site.node)) is not None:
                    # An exported alias is a name the caller can write; an unexported one is
                    # spelled out by the caller, so the classes in its value are what they need.
                    if site.qualified not in self.exported and site.qualified not in aliases:
                        aliases.add(site.qualified)
                        pending.append((site.module, value, frozenset()))

    def factory_returns(self, cls: Definition, spelled_as: str) -> bool:
        """Whether `spelled_as`, read in the class's module, is a public function or method whose
        return annotation names `cls`.
        """
        head, *rest = spelled_as.split(".")
        owner = resolve(self.project, cls.module, [head])
        if owner is None and spelled_as in self._public_paths():
            owner = self._public_paths()[spelled_as]
            rest = []
        if owner is None or not owner.in_packages:
            return False
        target: ast.stmt | None = owner.node
        if rest:
            if (
                len(rest) != 1
                or not isinstance(target, ast.ClassDef)
                or owner.qualified not in self.exported
                or not public_member(rest[0])
            ):
                return False
            target = next(
                (
                    s
                    for s in target.body
                    if isinstance(s, ast.FunctionDef | ast.AsyncFunctionDef) and s.name == rest[0]
                ),
                None,
            )
        elif owner.qualified not in self.exported:
            return False
        if not isinstance(target, ast.FunctionDef | ast.AsyncFunctionDef):
            return False
        returns = next(iter(spelled(target.returns)), None) if target.returns is not None else None
        if owner.qualified == cls.qualified and returns is not None and returns[-1] == "Self":
            return True
        return any(
            site.qualified == cls.qualified
            for site in self.named(owner.module, target.returns, frozenset())
        )

    def _public_paths(self) -> Mapping[str, Definition]:
        return self.project.memo(
            "public_paths",
            lambda: {
                f"{e.module}.{e.name}": e.definition
                for e in exports(self.project)
                if e.definition is not None
            },
        )


def type_params(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    return frozenset(
        p.name
        for p in node.type_params
        if isinstance(p, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple)
    )


# The decorators that generate `__init__` from the annotated fields of a class body.
DATACLASSES = frozenset({"dataclass", "refclass", "valueclass"})


def generates_init(node: ast.ClassDef) -> bool:
    """Whether a dataclass decorator not told `init=False` generates the class's `__init__`."""
    return any(
        bare_names([deco]) & DATACLASSES
        and not (
            isinstance(deco, ast.Call)
            and any(
                kw.arg == "init" and isinstance(kw.value, ast.Constant) and kw.value.value is False
                for kw in deco.keywords
            )
        )
        for deco in node.decorator_list
    )


def init_field(node: ast.stmt) -> bool:
    """Whether a class-body statement is a dataclass field its generated `__init__` takes."""
    if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
        return False
    if (kind := dotted_name(node.annotation)) is not None and kind[-1] == "ClassVar":
        return False
    value = node.value
    return not (
        isinstance(value, ast.Call)
        and getattr(value.func, "id", getattr(value.func, "attr", None)) == "field"
        and any(
            kw.arg == "init" and isinstance(kw.value, ast.Constant) and kw.value.value is False
            for kw in value.keywords
        )
    )


def member_name(node: ast.stmt) -> str | None:
    """The name a method or an annotated attribute of a class body binds."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def alias_value(node: ast.stmt | None) -> ast.expr | None:
    if isinstance(node, ast.TypeAlias):
        return node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        spelled_as = dotted_name(node.annotation)
        return node.value if spelled_as is not None and spelled_as[-1] == "TypeAlias" else None
    return None


def dotted_name(node: ast.expr) -> list[str] | None:
    return next(iter(spelled(node)), None)


def annotations_of(node: ast.stmt) -> Iterator[ast.expr | None]:
    """The annotations of a function's parameters and return, or of an annotated name."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = node.args
        for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
            yield arg.annotation
        for arg in (arguments.vararg, arguments.kwarg):
            if arg is not None:
                yield arg.annotation
        yield node.returns
    elif isinstance(node, ast.AnnAssign):
        yield node.annotation
