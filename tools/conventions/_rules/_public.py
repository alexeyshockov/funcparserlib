"""The public surface: what each public module's `__all__` exports, where each name is defined, and
pyrefly's list of the same definitions.
"""

import ast
import copy
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
        return f"{self.module}.{self.name}" if self.module else self.name

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
    """The statements that run when a module is imported, inside `if`, `try`, `with` and loop blocks
    included.
    """
    for node in body:
        yield node
        if isinstance(node, ast.If | ast.For | ast.AsyncFor | ast.While):
            yield from module_level(node.body)
            yield from module_level(node.orelse)
        elif isinstance(node, ast.With | ast.AsyncWith):
            yield from module_level(node.body)
        elif isinstance(node, ast.Try | ast.TryStar):
            yield from module_level(node.body)
            for handler in node.handlers:
                yield from module_level(handler.body)
            yield from module_level(node.orelse)
            yield from module_level(node.finalbody)


def _targets(node: ast.expr) -> Iterator[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Tuple | ast.List):
        for element in node.elts:
            yield from _targets(element)
    elif isinstance(node, ast.Starred):
        yield from _targets(node.value)


def _binds(node: ast.stmt, name: str) -> bool:
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name == name
    if isinstance(node, ast.TypeAlias):
        return isinstance(node.name, ast.Name) and node.name.id == name
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    if isinstance(node, ast.Assign):
        return any(name in _targets(t) for t in node.targets)
    if isinstance(node, ast.For | ast.AsyncFor):
        return name in _targets(node.target)
    if isinstance(node, ast.With | ast.AsyncWith):
        return any(
            item.optional_vars is not None and name in _targets(item.optional_vars)
            for item in node.items
        )
    return False


def _imported_module(node: ast.Import, name: str) -> str | None:
    """The module an `import` statement binds to `name`, if it binds it."""
    for alias in node.names:
        if alias.asname == name:
            return alias.name
        if alias.asname is None and alias.name.partition(".")[0] == name:
            return name
    return None


def _lazy_import(
    tree: ast.Module, module: str, is_package: bool, name: str
) -> tuple[str, str] | None:
    """The `from ... import` a module-level `__getattr__` serves `name` by, if it has one."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            for inner in ast.walk(node):
                if isinstance(inner, ast.ImportFrom):
                    for alias in inner.names:
                        if (alias.asname or alias.name) == name:
                            return resolved(module, is_package, inner), alias.name
    return None


def _module_definition(project: Project, dotted: str) -> Definition:
    parent, _, last = dotted.rpartition(".")
    return Definition(parent, last, project.module_path(dotted), None)


def definition(project: Project, module: str, name: str) -> Definition | None:
    """Where `name`, as `module` binds it, is defined: through `from ... import` re-exports, those
    under `if TYPE_CHECKING:` and those a module-level `__getattr__` makes included. A name bound by
    `import` is that module. A name no module on the way binds is the submodule of that name, or
    `None` when there is none.
    """
    seen: set[tuple[str, str]] = set()
    while (module, name) not in seen:
        seen.add((module, name))
        path = project.module_path(module)
        if path is None:
            return Definition(module, name, None, None)
        if (tree := project.tree(path)) is None:
            return None
        is_package = path.name == "__init__.py"
        imported: tuple[str, str] | None = None
        for node in module_level(tree.body):
            if _binds(node, name):
                return Definition(module, name, path, node)
            if isinstance(node, ast.Import) and (target := _imported_module(node, name)):
                return _module_definition(project, target)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        imported = (resolved(module, is_package, node), alias.name)
        if imported is None:
            imported = _lazy_import(tree, module, is_package, name)
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

    The report is read by its layout, whatever `schema_version` it states.

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
    found: set[str] = set()
    try:
        for module in report["module_reports"]:
            prefix = f"{module['name']}."
            found.update(
                name
                for name in module["names"]
                if name.startswith(prefix) and "." not in name.removeprefix(prefix)
            )
    except KeyError, TypeError, AttributeError:
        raise RunError(
            f"pyrefly's coverage report (schema {report.get('schema_version')!r}) has no"
            " module_reports of names, the layout the runner reads"
        ) from None
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


def nested(site: Definition, name: str) -> Definition | None:
    """The class `name` a class body defines, if it defines one."""
    if not isinstance(site.node, ast.ClassDef):
        return None
    for stmt in site.node.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == name:
            return Definition(site.module, f"{site.name}.{name}", site.path, stmt)
    return None


def resolve(
    project: Project, module: str, parts: Sequence[str], within: Definition | None = None
) -> Definition | None:
    """What the dotted name `parts`, spelled in `module`, is defined as: a class nested in a class
    included, and read in the body of the class `within` first. `None` for any other attribute of a
    class, or of anything that is not a module.
    """
    site = (nested(within, parts[0]) if within is not None else None) or definition(
        project, module, parts[0]
    )
    for attr in parts[1:]:
        if site is None:
            return None
        if isinstance(site.node, ast.ClassDef):
            site = nested(site, attr)
            continue
        if site.path is None or site.node is not None:
            return None
        site = definition(project, site.qualified, attr)
    return site


def is_overload(node: ast.stmt) -> bool:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and "overload" in bare_names(
        node.decorator_list
    )


def signatures(project: Project, site: Definition) -> list[ast.stmt]:
    """The statements a caller sees a definition by: each overload of an overloaded top-level
    function, otherwise the definition itself.
    """
    assert site.node is not None and site.path is not None
    if not is_overload(site.node) or (tree := project.tree(site.path)) is None:
        return [site.node]
    return [
        node
        for node in module_level(tree.body)
        if is_overload(node)
        and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == site.name
    ]


# --- What a public signature reaches ---

# The sentence a class's docstring carries when users get it from a factory, never by calling it.
INTERNAL_CONSTRUCTOR = "its constructor is internal"
# A factory is spelled with its call, `connect()`; an accessor without, `Match.target`.
FACTORY = re.compile(r"`([A-Za-z_][\w.]*)(\(\))?`")


def public_member(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def internal_constructor(node: ast.ClassDef) -> str | None:
    """The paragraph of the class's docstring that says its constructor is internal, if one does."""
    doc = ast.get_docstring(node) or ""
    return next((p for p in doc.split("\n\n") if INTERNAL_CONSTRUCTOR in " ".join(p.split())), None)


@dataclass(frozen=True, slots=True)
class Member:
    """One statement a class's public surface is read from."""

    owner: Definition
    """The class whose body holds the statement."""
    node: ast.stmt
    params: frozenset[str]
    """The type parameters in scope."""
    label: str
    """The name the member is reached by: `__init__` for a private field of a generated one."""


@dataclass(frozen=True, slots=True)
class Reach:
    """What a public signature reaches, from the definitions the public modules export."""

    project: Project
    exported: Mapping[str, str]
    """Each exported definition's qualified name, and the public name it is imported by."""

    def spelling(self, site: Definition) -> str:
        outer, inner = site.name, ""
        while f"{site.module}.{outer}" not in self.exported and "." in outer:
            outer, _, last = outer.rpartition(".")
            inner = f".{last}{inner}"
        public = self.exported.get(f"{site.module}.{outer}")
        return site.qualified if public is None else f"{public}{inner}"

    def importable(self, site: Definition) -> bool:
        """Whether a public module exports the class, or a class it is nested in."""
        return self.spelling(site) != site.qualified or site.qualified in self.exported

    def members(self, site: Definition) -> Iterator[Member]:
        """A class's public members, those of its bases in the packages included; a member a nearer
        class defines hides a base's, and an overloaded method is its overloads. The private fields
        of the dataclasses its generated `__init__` collects are that `__init__`'s.
        """
        assert isinstance(site.node, ast.ClassDef)
        hidden: set[str] = set()
        generated: bool | None = None
        pending = [site]
        done: set[str] = set()
        while pending:
            cls = pending.pop(0)
            if cls.qualified in done or not isinstance(node := cls.node, ast.ClassDef):
                continue
            done.add(cls.qualified)
            params = type_params(node)
            names = {name for stmt in node.body if (name := member_name(stmt)) is not None}
            if generated is None and "__init__" in names:
                generated = False
            elif generated is None and generates_init(node):
                generated = True
            overloaded = {
                stmt.name
                for stmt in node.body
                if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef) and is_overload(stmt)
            }
            for stmt in node.body:
                name = member_name(stmt)
                if name is None or name in hidden:
                    continue
                if name in overloaded and not is_overload(stmt):
                    continue
                if public_member(name):
                    yield Member(cls, stmt, params, name)
                elif generated and is_dataclass(node) and init_field(stmt):
                    yield Member(cls, stmt, params, "__init__")
            hidden |= names
            if generated:
                hidden.add("__init__")
            for base in node.bases:
                parts = next(iter(spelled(base)), None)
                if parts is not None and (found := resolve(self.project, cls.module, parts)):
                    if found.in_packages:
                        pending.append(found)

    def named(
        self,
        module: str,
        annotation: ast.expr | None,
        params: frozenset[str],
        within: Definition | None = None,
    ) -> Iterator[Definition]:
        """The definitions in the packages `annotation` names, each type alias's value read through,
        with the names of the class `within` read first.
        """
        pending = [] if annotation is None else [(module, annotation, params, within)]
        aliases: set[str] = set()
        while pending:
            module, expr, scope, owner = pending.pop()
            for parts in spelled(expr):
                if len(parts) == 1 and parts[0] in scope:
                    continue
                site = resolve(self.project, module, parts, owner)
                if site is None or not site.in_packages:
                    continue
                if isinstance(site.node, ast.ClassDef):
                    yield site
                elif (value := alias_value(site.node)) is not None:
                    if site.qualified not in aliases:
                        aliases.add(site.qualified)
                        alias_params = (
                            type_params(site.node)
                            if isinstance(site.node, ast.TypeAlias)
                            else frozenset()
                        )
                        pending.append((site.module, value, alias_params, None))

    def factory_returns(self, cls: Definition, spelled_as: str, called: bool = True) -> bool:
        """Whether `spelled_as`, read in the class's module or a public one exporting it, is a
        public function or method one of whose signatures returns `cls`, or a public attribute of an
        exported class (a property, or an annotated name) whose class's `__call__` returns `cls`:
        `conn.copy()` where `copy` is an object. A class a returned callable takes as a parameter is
        not returned. Not `called`, it is a public attribute of an exported class whose value is
        `cls`: `Match.target`.
        """
        head, *rest = spelled_as.split(".")
        # Read where the class is defined, then where the reference shows it: a public module
        # exporting it, beside the factory.
        homes = [cls.module, *(e.module for e in exports(self.project) if e.definition == cls)]
        owner = next(
            (site for home in homes if (site := resolve(self.project, home, [head])) is not None),
            None,
        )
        if owner is None and spelled_as in self._public_paths():
            owner = self._public_paths()[spelled_as]
            rest = []
        if owner is None or not owner.in_packages:
            return False
        assert owner.node is not None
        targets: list[ast.stmt] = signatures(self.project, owner)
        if rest:
            if (
                len(rest) != 1
                or not isinstance(owner.node, ast.ClassDef)
                or owner.qualified not in self.exported
                or not public_member(rest[0])
            ):
                return False
            found = [m.node for m in self.members(owner) if m.label == rest[0]]
            if not called:
                return bool(found) and self._holds(owner, found[0], cls)
            if found and (callee := self._called(owner.module, found[0])) is not None:
                owner = callee
                found = [m.node for m in self.members(callee) if m.label == "__call__"]
            targets = found
        elif owner.qualified not in self.exported or not called:
            return False
        return any(self._returns(owner, target, cls) for target in targets)

    def _returns(self, owner: Definition, target: ast.stmt, cls: Definition) -> bool:
        if not isinstance(target, ast.FunctionDef | ast.AsyncFunctionDef) or target.returns is None:
            return False
        returns = next(iter(spelled(target.returns)), None)
        if owner.qualified == cls.qualified and returns is not None and returns[-1] == "Self":
            return True
        produced = _Produced().visit(copy.deepcopy(target.returns))
        return any(
            site.qualified == cls.qualified
            for site in self.named(owner.module, produced, frozenset(), owner)
        )

    def _holds(self, owner: Definition, member: ast.stmt, cls: Definition) -> bool:
        """Whether a property or an annotated attribute holds `cls`, in a union or an alias
        included.
        """
        annotation = _attribute_annotation(member)
        if annotation is None:
            return False
        produced = _Produced().visit(copy.deepcopy(annotation))
        return any(
            site.qualified == cls.qualified
            for site in self.named(owner.module, produced, frozenset(), owner)
        )

    def _called(self, module: str, member: ast.stmt | None) -> Definition | None:
        """The class of a property or an annotated attribute, when it is one in the packages."""
        annotation = _attribute_annotation(member)
        parts = next(iter(spelled(annotation)), None) if annotation is not None else None
        site = resolve(self.project, module, parts) if parts is not None else None
        if site is None or not isinstance(site.node, ast.ClassDef):
            return None
        return site

    def _public_paths(self) -> Mapping[str, Definition]:
        return self.project.memo(
            "public_paths",
            lambda: {
                f"{e.module}.{e.name}": e.definition
                for e in exports(self.project)
                if e.definition is not None
            },
        )


def _attribute_annotation(member: ast.stmt | None) -> ast.expr | None:
    """The annotation of an annotated attribute, or the return of a property."""
    if isinstance(member, ast.AnnAssign):
        return member.annotation
    if isinstance(member, ast.FunctionDef) and bare_names(member.decorator_list) & {
        "property",
        "cached_property",
    }:
        return member.returns
    return None


class _Produced(ast.NodeTransformer):
    """`Callable[[...], R]` read as `R`: what calling the annotated value hands out."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        parts = dotted_name(node.value)
        if (
            parts is not None
            and parts[-1] == "Callable"
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
        ):
            return self.visit(node.slice.elts[1])
        return self.generic_visit(node)


def type_params(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.TypeAlias,
) -> frozenset[str]:
    return frozenset(
        p.name
        for p in node.type_params
        if isinstance(p, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple)
    )


# The decorators that generate `__init__` from the annotated fields of a class body.
DATACLASSES = frozenset({"dataclass", "refclass", "valueclass"})


def is_dataclass(node: ast.ClassDef) -> bool:
    """Whether a dataclass decorator collects the class's fields, `init=False` or not."""
    return bool(bare_names(node.decorator_list) & DATACLASSES)


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
    if (kind := dotted_name(node.annotation)) is not None and kind[-1] in ("ClassVar", "KW_ONLY"):
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
    """The name a method, an annotated attribute or a plain assignment of a class body binds; a
    plain assignment to a dunder (`__slots__`, `__match_args__`) is the class's machinery.
    """
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(target := node.targets[0], ast.Name)
        and not (target.id.startswith("__") and target.id.endswith("__"))
    ):
        return target.id
    return None


def _type_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return all(
            (isinstance(side, ast.Constant) and side.value is None) or _type_expression(side)
            for side in (node.left, node.right)
        )
    return isinstance(node, ast.Name | ast.Attribute | ast.Subscript)


def alias_value(node: ast.stmt | None) -> ast.expr | None:
    """The value a type alias names: a `type` statement's, a `TypeAlias`-annotated name's, or a
    plain assignment's whose value is a type expression (`Handler = Callable[[Event], None]`).
    """
    if isinstance(node, ast.TypeAlias):
        return node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        spelled_as = dotted_name(node.annotation)
        return node.value if spelled_as is not None and spelled_as[-1] == "TypeAlias" else None
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _type_expression(node.value)
    ):
        return node.value
    return None


def dotted_name(node: ast.expr) -> list[str] | None:
    return next(iter(spelled(node)), None)


def annotations_of(node: ast.stmt) -> Iterator[ast.expr | None]:
    """The annotations of a function's parameters and return, or of an annotated name, the value of
    a type alias, and the bounds and constraints of a class's, a function's or an alias's type
    parameters.
    """
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.TypeAlias):
        for param in node.type_params:
            if isinstance(param, ast.TypeVar):
                yield param.bound
    if (value := alias_value(node)) is not None:
        yield value
        return
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
