"""CON51x: the time bounds a test suite sets."""

import ast
import io
import tokenize
import tomllib
from collections.abc import Iterator, Mapping, Sequence, Set
from pathlib import Path

from .._engine import Finding, rule
from .._project import Project
from .names import source_roots

# A method that waits. The scope is drawn by what is called rather than by a keyword's name: the
# same word configures a pool where it is passed to a constructor and bounds an assertion where it
# is passed to a checkout.
WAITING_METHODS = frozenset(
    {"receive", "areceive", "wait", "acquire", "poll", "get", "recv", "join", "result"}
)

# Functions that suspend, however they are imported: `time.sleep`, `anyio.sleep`,
# `anyio.move_on_after`, `asyncio.timeout`. `within(seconds)` constructs a budget
# that expires after `seconds`.
WAITING_FUNCTIONS = frozenset({"sleep", "move_on_after", "fail_after", "timeout", "within"})

# The calls whose expiry fails the test with an error of its own, which a backstop that fires at
# the same moment replaces with its stack dump.
FAILING_BOUNDS = frozenset({"fail_after", "move_on_after", "within"})

# A `pytest.raises` over one of these says a bound expired. A project adds its own in
# `[tool.conventions] timeout_exceptions`.
TIMEOUT_EXCEPTIONS = frozenset({"TimeoutError"})

# A second or more cannot fire spuriously on a machine running the suite: the point at which a
# bound stops being a guess about scheduling.
GENEROUS_SECONDS = 1.0


def test_roots(project: Project) -> list[Path]:
    """The `tests/` beside each package, and at the repository root."""
    return [path for path in source_roots(project) if path.name == "tests"]


def _test_files(project: Project) -> Iterator[tuple[Path, str, ast.Module]]:
    for path in project.under(test_roots(project), frozenset({".py"})):
        text, tree = project.text(path), project.tree(path)
        if text is not None and tree is not None:
            yield path, text, tree


def called_name(call: ast.Call) -> str:
    match call.func:
        case ast.Attribute(attr=attr):
            return attr
        case ast.Name(id=name):
            return name
        case _:
            return ""


def module_seconds(tree: ast.Module) -> dict[str, float]:
    """Module-level `NAME = <number>` bindings, including ones derived from another.

    CON512 asks for a sub-second bound to be named, and CON511 refuses a control whose bound is
    wider than the assertion it vouches for; without resolving names, obeying the first would switch
    the second off.
    """
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and (value := seconds_of(node.value, found)) is not None:
            found[target.id] = value
    return found


def seconds_of(node: ast.expr, constants: Mapping[str, float]) -> float | None:
    """`node` as a number of seconds, resolving names and the arithmetic over them."""
    match node:
        case ast.Constant(value=bool()):
            return None
        case ast.Constant(value=int() | float() as value):
            return float(value)
        case ast.Name(id=name):
            return constants.get(name)
        case ast.BinOp(left=left, op=ast.Mult() | ast.Add() | ast.Sub() | ast.Div(), right=right):
            a, b = seconds_of(left, constants), seconds_of(right, constants)
            if a is None or b is None:
                return None
            match node.op:
                case ast.Mult():
                    return a * b
                case ast.Add():
                    return a + b
                case ast.Sub():
                    return a - b
                case _:
                    return a / b if b else None
        case _:
            return None


def literal_seconds(node: ast.expr) -> float | None:
    """`node` as a literal number of seconds; a name is not one, which is how naming satisfies
    CON512.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return None if isinstance(node.value, bool) else float(node.value)
    return None


def bound_expression(call: ast.Call) -> ast.expr | None:
    """The expression a waiting call's bound is spelled with, literal or not.

    A positional first argument to something that suspends; a `timeout=` keyword on something that
    waits; or the offset in `deadline=<clock>() + <seconds>`, on any call.
    """
    name = called_name(call)
    for keyword in call.keywords:
        if keyword.arg == "deadline":
            return offset_of(keyword.value)
    if name in WAITING_FUNCTIONS and call.args:
        return call.args[0]
    if name not in WAITING_FUNCTIONS and name not in WAITING_METHODS:
        return None
    for keyword in call.keywords:
        if keyword.arg == "timeout":
            return keyword.value
    return None


def offset_of(node: ast.expr) -> ast.expr | None:
    """The addend in `<clock>() + <seconds>`."""
    match node:
        case ast.BinOp(op=ast.Add(), left=ast.Call(), right=right):
            return right
        case ast.BinOp(op=ast.Add(), left=left, right=ast.Call()):
            return left
        case _:
            return None


def comment_lines(source: str) -> frozenset[int]:
    """Line numbers holding a comment token; a `#` inside a string is not one."""
    found: set[int] = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                found.add(token.start[0])
    except tokenize.TokenError, IndentationError:
        pass
    return frozenset(found)


def is_explained(lines: Sequence[str], comments: Set[int], first: int, last: int) -> bool:
    """A comment within the statement's span, or in the comment block directly above it."""
    if any(line in comments for line in range(first, last + 1)):
        return True
    previous = first - 1
    while previous >= 1 and not lines[previous - 1].strip():
        previous -= 1
    return previous >= 1 and previous in comments and lines[previous - 1].lstrip().startswith("#")


def enclosing_statement(tree: ast.AST, node: ast.AST) -> ast.stmt | None:
    """The innermost statement containing `node`: where a comment about a wrapped call goes."""
    innermost: ast.stmt | None = None
    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.stmt):
            continue
        for inner in ast.walk(candidate):
            if inner is node:
                if innermost is None or candidate.lineno >= innermost.lineno:
                    innermost = candidate
                break
    return innermost


def test_functions(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test_"
        ):
            yield node


# --- CON510 ---


def backstop(project: Project, root: Path) -> float | None:
    """pytest-timeout's `timeout` for the tests under `root`: the nearest `pyproject.toml`'s."""
    for candidate in (root.parent / "pyproject.toml", project.root / "pyproject.toml"):
        if candidate.is_file():
            try:
                data = tomllib.loads(candidate.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError:
                return None
            options = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
            if "timeout" in options:
                return float(options["timeout"])
    return None


def own_timeout(function: ast.FunctionDef | ast.AsyncFunctionDef) -> float | None:
    """A `@pytest.mark.timeout(n)` on the test, which replaces the backstop for it."""
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and ast.unparse(decorator.func).endswith("mark.timeout")
            and decorator.args
            and (limit := literal_seconds(decorator.args[0])) is not None
        ):
            return limit
    return None


def backstop_offenders(tree: ast.Module, ceiling: float) -> list[tuple[int, float, float]]:
    """Each literal failing bound at or above the limit it runs under: `(line, bound, limit)`."""
    limits: dict[int, float] = {}
    for function in ast.walk(tree):
        if (
            isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
            and (limit := own_timeout(function)) is not None
        ):
            limits |= {id(inner): limit for inner in ast.walk(function)}
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and called_name(node) in FAILING_BOUNDS and node.args):
            continue
        bound, limit = literal_seconds(node.args[0]), limits.get(id(node), ceiling)
        if bound is not None and bound >= limit:
            found.append((node.lineno, bound, limit))
    return found


@rule("CON510")
def under_the_backstop(project: Project) -> Iterator[Finding]:
    """A bound a test sets stays below pytest-timeout's backstop.

    Reason: a `fail_after(n)` that ties the backstop races it, and when the backstop wins it ends
    the process before pytest reports the test's own `TimeoutError`: the run keeps only the stacks
    the backstop dumps.

    Fix: lower the bound, or raise the test's own limit with `@pytest.mark.timeout(n)`.

    Scope: the `tests/` beside each package and at the root, under the `timeout` of the nearest
    `pyproject.toml`'s `[tool.pytest.ini_options]`; nothing where none is set.
    """
    for root in test_roots(project):
        if (ceiling := backstop(project, root)) is None:
            continue
        for path in project.under([root], frozenset({".py"})):
            if (tree := project.tree(path)) is None:
                continue
            for line, bound, limit in backstop_offenders(tree, ceiling):
                yield Finding(
                    project.relative(path),
                    line,
                    "CON510",
                    f"a {bound:g}s bound under a {limit:g}s limit races the backstop",
                )


# --- CON511 ---


def receiver_of(call: ast.Call) -> str | None:
    """The object a method is called on, as source, or `None` for a bare call."""
    return ast.unparse(call.func.value) if isinstance(call.func, ast.Attribute) else None


def _unawait(node: ast.expr) -> ast.expr:
    return node.value if isinstance(node, ast.Await) else node


def negated_call(test: ast.expr) -> ast.Call | None:
    """The call inside `assert not f(...)` or `assert f(...) is None`."""
    match test:
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            inner = _unawait(operand)
        case ast.Compare(
            left=left, ops=[ast.Is() | ast.Eq()], comparators=[ast.Constant(value=None)]
        ):
            inner = _unawait(left)
        case _:
            return None
    return inner if isinstance(inner, ast.Call) else None


def raises_timeout(item: ast.withitem, exceptions: Set[str]) -> bool:
    """Whether a `with` item is a `pytest.raises` over an expired-bound exception."""
    call = item.context_expr
    if not isinstance(call, ast.Call) or called_name(call) != "raises" or not call.args:
        return False
    named = call.args[0]
    elements = named.elts if isinstance(named, ast.Tuple) else [named]
    return any(
        (isinstance(e, ast.Name) and e.id in exceptions)
        or (isinstance(e, ast.Attribute) and e.attr in exceptions)
        for e in elements
    )


def timeout_expectations(
    function: ast.AST, exceptions: Set[str]
) -> Iterator[tuple[ast.stmt, ast.Call]]:
    """`with pytest.raises(TimeoutError): <a bounded wait>`, a negative in another shape."""
    for node in ast.walk(function):
        if isinstance(node, ast.With | ast.AsyncWith) and any(
            raises_timeout(item, exceptions) for item in node.items
        ):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and bound_expression(inner) is not None:
                    yield node, inner


def has_control(
    function: ast.AST,
    receiver: str,
    seconds: float,
    inside: Set[int],
    constants: Mapping[str, float],
) -> bool:
    """Whether the same object is used elsewhere in the test in a way that can succeed.

    A call counts only where its result is used, and with a bound no wider than the one it vouches
    for, a name resolved to its value. A plain mention counts only where the object is passed to
    something, as in `next(iter(notifications))`.
    """
    discarded = {
        id(node.value)
        for node in ast.walk(function)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    }
    used_as_receiver: set[int] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or id(node) in inside or receiver_of(node) != receiver:
            continue
        assert isinstance(node.func, ast.Attribute), "a receiver implies an attribute call"
        used_as_receiver.add(id(node.func.value))
        if id(node) in discarded:
            continue
        bound = bound_expression(node)
        if bound is None:
            return True
        if (value := seconds_of(bound, constants)) is not None and value <= seconds:
            return True
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or id(node) in inside:
            continue
        for argument in [*node.args, *(k.value for k in node.keywords)]:
            if (
                id(argument) not in used_as_receiver
                and isinstance(argument, ast.Name | ast.Attribute)
                and isinstance(argument.ctx, ast.Load)
                and ast.unparse(argument) == receiver
            ):
                return True
    return False


def uncontrolled_negatives(source: str, tree: ast.Module, exceptions: Set[str]) -> list[int]:
    """The lines of time-bounded negative assertions with nothing showing the object delivers."""
    lines = source.splitlines()
    comments = comment_lines(source)
    constants = module_seconds(tree)
    found = []
    for function in test_functions(tree):
        candidates: list[tuple[ast.stmt, ast.Call]] = [
            (node, call)
            for node in ast.walk(function)
            if isinstance(node, ast.Assert) and (call := negated_call(node.test)) is not None
        ]
        candidates += timeout_expectations(function, exceptions)
        negatives = []
        for node, call in candidates:
            bound, receiver = bound_expression(call), receiver_of(call)
            if bound is None or (seconds := literal_seconds(bound)) is None or receiver is None:
                continue
            # A type, or a module's bound constructor (`anyio.fail_after`, `asyncio.timeout`), names
            # the bound, not the thing waited on, so it has no control to look for.
            if receiver.rpartition(".")[2][:1].isupper() or called_name(call) in WAITING_FUNCTIONS:
                continue
            negatives.append((node, receiver, seconds))
        inside = {id(n) for statement, _, _ in negatives for n in ast.walk(statement)}
        for statement, receiver, seconds in negatives:
            if is_explained(lines, comments, statement.lineno, statement.end_lineno or 0):
                continue
            if not has_control(function, receiver, seconds, inside, constants):
                found.append(statement.lineno)
    return found


@rule("CON511")
def positive_controls(project: Project) -> Iterator[Finding]:
    """A time-bounded negative assertion is shown that the same object can deliver.

    Reason: `assert rx.get(timeout=0.05) is None` passes when nothing arrived, and passes just as
    well when the mechanism is broken and nothing ever could; a bound too small makes it pass on a
    slow machine. Waiting longer cannot tell the two apart, so the bound's size does not exempt it.

    Fix: use the same object in the test in a way that succeeds, with a bound no wider than the
    negative's (a receive that returns, or the object passed to something that reads it), or say in
    a comment why the absence is the promise itself. The negative shapes are
    `assert not x.wait(...)`, `assert x.get(...) is None`, and `pytest.raises(TimeoutError)` around
    a bounded wait; a project's own timeout exceptions go in
    `[tool.conventions] timeout_exceptions`.

    Scope: the test functions in the `tests/` beside each package and at the root.
    """
    exceptions = TIMEOUT_EXCEPTIONS | set(project.settings.get("timeout_exceptions", []))
    for path, text, tree in _test_files(project):
        for line in uncontrolled_negatives(text, tree, exceptions):
            yield Finding(
                project.relative(path),
                line,
                "CON511",
                "negative bound with no positive control on the same object",
            )


# --- CON512 ---


def unexplained_bounds(source: str, tree: ast.Module) -> list[tuple[int, float, str]]:
    """Sub-second literal bounds in waiting calls with neither a name nor a comment."""
    lines = source.splitlines()
    comments = comment_lines(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or (expression := bound_expression(node)) is None:
            continue
        seconds = literal_seconds(expression)
        if seconds is None or seconds == 0 or seconds >= GENEROUS_SECONDS:
            continue
        statement = enclosing_statement(tree, node)
        first = statement.lineno if statement is not None else node.lineno
        last = (statement.end_lineno if statement is not None else node.end_lineno) or first
        if not is_explained(lines, comments, first, last):
            found.append((expression.lineno, seconds, called_name(node)))
    return found


@rule("CON512")
def sub_second_bounds(project: Project) -> Iterator[Finding]:
    """A sub-second bound in a test is a named constant or says why it is safe.

    Reason: a number below a second is a guess about scheduling until it says otherwise, and an
    unexplained one is copied to its neighbours with its assumptions. A bound of zero is a
    checkpoint or an expired deadline, not a guess, and a second or more cannot fire spuriously.

    Fix: name the value as a module constant where it expresses a relationship or repeats
    (`SETTLE = 0.05`), or put a comment on or above the statement saying why it is safe at that
    size.

    Scope: the waiting calls (`sleep`, `move_on_after`, `fail_after`, `timeout`, `within`, and a
    `timeout=` to `get`, `wait`, `receive` and their kin, or a `deadline=` offset) in the `tests/`
    beside each package and at the root.
    """
    for path, text, tree in _test_files(project):
        for line, seconds, name in unexplained_bounds(text, tree):
            yield Finding(
                project.relative(path),
                line,
                "CON512",
                f"{seconds:g}s in {name}() is neither named nor explained",
            )
