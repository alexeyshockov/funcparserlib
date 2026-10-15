"""AST readers shared by the rule families."""

import ast


def alias_name(node: ast.AST) -> str | None:
    """The name a type alias binds, in either spelling, or `None` if `node` is not one."""
    if isinstance(node, ast.TypeAlias):
        return node.name.id if isinstance(node.name, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        annotation = node.annotation
        spelled = (
            annotation.id if isinstance(annotation, ast.Name) else getattr(annotation, "attr", None)
        )
        return node.target.id if spelled == "TypeAlias" else None
    return None


def all_value(tree: ast.Module) -> ast.expr | None:
    """The value a module binds to `__all__`, plain or annotated, or `None` if it binds none."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "__all__"
            and node.value is not None
        ):
            return node.value
    return None


def exported_by(tree: ast.Module) -> set[str]:
    """A module's literal `__all__`."""
    value = all_value(tree)
    if not isinstance(value, ast.List | ast.Tuple):
        return set()
    return {e.value for e in value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}


def bare_names(nodes: list[ast.expr]) -> set[str]:
    """The bare names of decorators or bases: calls and generic subscripts unwrapped."""
    found: set[str] = set()
    for node in nodes:
        target = node
        if isinstance(target, ast.Call):
            target = target.func
        if isinstance(target, ast.Subscript):
            target = target.value
        found.add(target.id if isinstance(target, ast.Name) else getattr(target, "attr", ""))
    return found


def resolved(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    """The absolute module an `ImportFrom` in `module` names."""
    if not node.level:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    for _ in range(node.level - 1):
        package = package.rpartition(".")[0]
    return f"{package}.{node.module}" if node.module else package


def line_at(text: str, offset: int) -> int:
    """The 1-based line holding `offset` in `text`."""
    return text.count("\n", 0, offset) + 1
