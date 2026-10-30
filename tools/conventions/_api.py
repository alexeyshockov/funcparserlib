"""The public API as a caller uses it, and what changed in it between two trees.

The surface is `_rules._public`'s: what the public modules' `__all__` export, each class's public
members and those of its bases in the packages, with each annotation as written. A constructor whose
class says it is internal is not API, and neither is an addition, which breaks no caller.
"""

import ast
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from ._project import Project, RunError
from ._rules._public import (
    Definition,
    Reach,
    alias_value,
    exports,
    generates_init,
    init_field,
    internal_constructor,
    member_name,
    resolve,
    spelled,
)

POSITIONAL_ONLY, EITHER, KEYWORD_ONLY, VAR_POSITIONAL, VAR_KEYWORD = (
    "positional-only",
    "positional-or-keyword",
    "keyword-only",
    "*args",
    "**kwargs",
)


@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    kind: str
    default: bool
    annotation: str


@dataclass(frozen=True, slots=True)
class Item:
    """One public name: what it is, and for a callable its parameters and return."""

    kind: str
    """`function`, `class`, `attribute`, `alias`, `module` or `imported`."""
    text: str = ""
    """An attribute's annotation, if it has one; an alias's value; an imported name's source."""
    parameters: tuple[Parameter, ...] = ()
    returns: str = ""


class _Strings(ast.NodeTransformer):
    """String annotations parsed in place, so `"Foo"` and `Foo` read the same; `Literal` kept."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if (parts := next(iter(spelled(node.value)), None)) and parts[-1] == "Literal":
            return node
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            try:
                return self.visit(ast.parse(node.value.strip(), mode="eval").body)
            except SyntaxError:
                return node
        return node


def _text(node: ast.expr | None) -> str:
    return "" if node is None else ast.unparse(_Strings().visit(node))


def _callable(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Item:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [False] * (len(positional) - len(args.defaults)) + [True] * len(args.defaults)
    params = [
        Parameter(
            arg.arg,
            POSITIONAL_ONLY if index < len(args.posonlyargs) else EITHER,
            defaults[index],
            _text(arg.annotation),
        )
        for index, arg in enumerate(positional)
    ]
    if args.vararg is not None:
        params.append(
            Parameter(args.vararg.arg, VAR_POSITIONAL, True, _text(args.vararg.annotation))
        )
    params += [
        Parameter(arg.arg, KEYWORD_ONLY, default is not None, _text(arg.annotation))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    ]
    if args.kwarg is not None:
        params.append(Parameter(args.kwarg.arg, VAR_KEYWORD, True, _text(args.kwarg.annotation)))
    return Item("function", parameters=tuple(params), returns=_text(node.returns))


def _generated_init(project: Project, site: Definition) -> Item | None:
    """A dataclass's generated `__init__`: its fields, its bases' first; `None` for a class that
    writes its own, or is no dataclass.
    """
    chain: list[ast.ClassDef] = []
    pending: list[Definition] = [site]
    while pending:
        cls = pending.pop(0)
        if not isinstance(cls.node, ast.ClassDef) or cls.node in chain:
            continue
        if any(member_name(stmt) == "__init__" for stmt in cls.node.body):
            return None
        chain.append(cls.node)
        for base in cls.node.bases:
            parts = next(iter(spelled(base)), None)
            if parts is not None and (found := resolve(project, cls.module, parts)) is not None:
                if found.in_packages:
                    pending.append(found)
    if not generates_init(chain[0]):
        return None
    params: dict[str, Parameter] = {}
    for node in reversed(chain):
        for stmt in node.body:
            if init_field(stmt) and isinstance(stmt, ast.AnnAssign):
                assert isinstance(stmt.target, ast.Name)
                params[stmt.target.id] = Parameter(
                    stmt.target.id, EITHER, stmt.value is not None, _text(stmt.annotation)
                )
    return Item("function", parameters=(Parameter("self", EITHER, False, ""), *params.values()))


def snapshot(project: Project) -> Mapping[str, Item]:
    """Every public name, by the dotted path a caller imports it by, and each exported class's
    public members, by that path and the member's name.
    """
    exported = {
        site.qualified: f"{e.module}.{e.name}"
        for e in reversed(exports(project))
        if (site := e.definition) is not None and site.in_packages
    }
    reach = Reach(project, exported)
    found: dict[str, Item] = {m.name: Item("module") for m in project.modules if m.public}
    for export in exports(project):
        path, site = f"{export.module}.{export.name}", export.definition
        if site is None:
            continue
        if site.path is None:
            found[path] = Item("imported", site.qualified)
        elif site.node is None:
            found[path] = Item("module")
        elif isinstance(site.node, ast.ClassDef):
            found[path] = Item("class")
            internal = internal_constructor(site.node) is not None
            for _, member, _, label in reach.members(site):
                if label in ("__init__", "__new__") and internal:
                    continue
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    found.setdefault(f"{path}.{label}", _callable(member))
                elif isinstance(member, ast.AnnAssign) and label != "__init__":
                    found.setdefault(f"{path}.{label}", Item("attribute", _text(member.annotation)))
            if not internal and (init := _generated_init(project, site)) is not None:
                found.setdefault(f"{path}.__init__", init)
        elif isinstance(site.node, ast.FunctionDef | ast.AsyncFunctionDef):
            found[path] = _callable(site.node)
        elif (value := alias_value(site.node)) is not None:
            found[path] = Item("alias", _text(value))
        elif isinstance(site.node, ast.AnnAssign):
            found[path] = Item("attribute", _text(site.node.annotation))
        else:
            found[path] = Item("attribute")
    return found


def _parameter_changes(old: Item, new: Item) -> Iterator[str]:
    by_name = {p.name: p for p in new.parameters}
    positional = [p for p in new.parameters if p.kind in (POSITIONAL_ONLY, EITHER)]
    takes_args = any(p.kind == VAR_POSITIONAL for p in new.parameters)
    takes_kwargs = any(p.kind == VAR_KEYWORD for p in new.parameters)
    old_positional = [p for p in old.parameters if p.kind in (POSITIONAL_ONLY, EITHER)]
    for index, param in enumerate(old_positional):
        if index >= len(positional) and not takes_args:
            yield f"parameter `{param.name}` can no longer be passed by position"
    for param in old.parameters:
        if param.kind in (VAR_POSITIONAL, VAR_KEYWORD):
            if not any(p.kind == param.kind for p in new.parameters):
                yield f"no longer takes `{param.kind}`"
            continue
        now = by_name.get(param.name)
        if param.kind in (EITHER, KEYWORD_ONLY) and (
            now is None or now.kind not in (EITHER, KEYWORD_ONLY)
        ):
            if not takes_kwargs:
                yield f"parameter `{param.name}` can no longer be passed by keyword"
            continue
        if now is None:
            continue
        if param.default and not now.default:
            yield f"parameter `{param.name}` no longer has a default"
        if param.annotation != now.annotation:
            yield f"parameter `{param.name}` was `{param.annotation}`, is `{now.annotation}`"
    old_names = {p.name for p in old.parameters}
    for param in new.parameters:
        if param.name not in old_names and not param.default:
            yield f"new required parameter `{param.name}`"
    if old.returns != new.returns:
        yield f"returns `{new.returns}`, was `{old.returns}`"


def changes(old: Mapping[str, Item], new: Mapping[str, Item]) -> Iterator[str]:
    """What a caller of the `old` API meets in the `new` one, one line per change."""
    removed: list[str] = []
    for path, was in sorted(old.items()):
        now = new.get(path)
        if now is None:
            # Everything under a removed name goes with it.
            if not any(path.startswith(f"{gone}.") for gone in removed):
                removed.append(path)
                yield f"{path}: removed"
        elif now.kind != was.kind:
            yield f"{path}: was {_a(was.kind)}, is {_a(now.kind)}"
        elif was.kind == "function":
            yield from (f"{path}: {change}" for change in _parameter_changes(was, now))
        elif was.text and was.text != now.text:
            # An annotation added where there was none narrows nothing a caller relied on.
            yield f"{path}: was `{was.text}`, is `{now.text}`"


def _a(kind: str) -> str:
    return f"an {kind}" if kind[0] in "aeiou" else f"a {kind}"


def at_revision(root: Path, revision: str, into: Path) -> Project:
    """The tree of `revision`, extracted into `into` as a repository of its own."""
    try:
        archive = subprocess.run(
            ["git", "archive", "--format=tar", revision],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(into)], input=archive, check=True)
        subprocess.run(["git", "init", "-q"], cwd=into, check=True)
    except subprocess.CalledProcessError as e:
        message = e.stderr.decode(errors="replace").strip() if e.stderr else str(e)
        raise RunError(f"cannot read {revision}: {message}") from None
    return Project(into)


def diff(root: Path, base: str) -> list[str]:
    """What changed in the public API from `base` to the working tree."""
    with tempfile.TemporaryDirectory(prefix="api-diff-") as scratch:
        old = snapshot(at_revision(root, base, Path(scratch)))
    return list(changes(old, snapshot(Project(root))))
