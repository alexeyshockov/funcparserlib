"""`Parser.compile()`: a grammar as generated Python functions, for the parses that succeed.

The generated code keeps a position in a local variable and tracks no error state, see ADR 0002. A
parser becomes nested `if` statements: code for what follows a parser runs inside the test that it
succeeded, and a failure falls out of the block. Each position and value has a name of its own, so
backtracking needs no reset.

Every function takes the tokens `T`, their types `Y` and a position. `Y` ends in `None`, so testing
it past the last token fails. A function returns `(value, position)` or `None`, or for a parser
whose value nothing reads, the position or `-1`.

The code is built as `ast` nodes, see ADR 0002. In an expression a `str` is a name of generated
code, and a `str` constant is `ast.Constant`.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from importlib import import_module
from operator import attrgetter, index
from pathlib import Path
from re import IGNORECASE, Pattern
from typing import Any, Literal

from ._commons import valueclass
from .lexer import Token
from .parser import IgnoredValue, Parser, ParsingFailure, State

type _Kind = Literal["ignored", "plain", "tuple", "unknown"]
"""What `+` needs to know of a value: an `IgnoredValue`, a `_Tuple`, neither, or any of them."""

type _Next = Callable[[str, _Value | None], None]
"""Emits what follows a parser, given the position after it and its value, if asked for."""

type _Expr = ast.expr | str | int | None
"""An expression of generated code: a `str` is a name, an `int` or `None` a constant."""

type _First = tuple[frozenset[str] | None, bool]
"""The token types a parser can start with, `None` when unknown, and whether it can read none."""

_INLINE_SIZE = 10
"""A parser used more than once gets a function of its own when it is larger than this."""

_MAX_DEPTH = 60
"""Code nested deeper goes to a function of its own: CPython's compiler recurses over the nesting.
"""

_MAX_LOOPS = 15
"""Loops nested deeper go to a function of their own: CPython compiles 20 nested blocks."""

_GROUP_SIZE = 32
"""A longer sequence is split into functions of this many parsers, as each nests the next."""

_MAX_START_TESTS = 16
"""A `commit()` with more token tests at its start leaves its failures to the interpreted parser."""

_MAX_CLASS_RANGE = 8192
"""A wider range of a character class leaves the regex without a first-character test."""

_READS_INPUT = ("text", "text_end", "byte", "fixed", "previous")
"""The forms of parsers that read text or bytes, and no tokens."""

_FILENAME = "<funcparserlib grammar>"
"""The file name of generated code, in its code objects and tracebacks."""

_TOKEN_CLASS = {Token}
_token_type = attrgetter("type")


@valueclass
class Internals:
    """What generated code shares with the parsers of `funcparserlib.parser`."""

    tuple_class: type[tuple[Any, ...]]
    ignored: IgnoredValue
    ignore: Callable[[Any], IgnoredValue]
    token_value: Callable[[Any], Any]
    undecided: type[Exception]


class _Declined(Exception):
    """The grammar stays interpreted."""


@valueclass
class _Chars:
    """The characters of `chars`, or with `negated` every other character."""

    chars: frozenset[str]
    negated: bool = False


_NO_CHARS = _Chars(frozenset())


def _union(left: _Chars | None, right: _Chars | None) -> _Chars | None:
    """The characters of both, `None` when either is unknown."""
    if left is None or right is None:
        return None
    a, b = left.chars, right.chars
    if left.negated:
        return _Chars(a & b if right.negated else a - b, negated=True)
    return _Chars(b - a, negated=True) if right.negated else _Chars(a | b)


def _class_chars(items: Sequence[tuple[Any, Any]], names: Any) -> _Chars | None:
    """The characters of a character class of `re._parser`, `None` for a category such as `\\d`."""
    chars: set[str] = set()
    negated = False
    for op, arg in items:
        if op is names.NEGATE:
            negated = True
        elif op is names.LITERAL:
            chars.add(chr(arg))
        elif op is names.RANGE and arg[1] - arg[0] <= _MAX_CLASS_RANGE:
            chars.update(map(chr, range(arg[0], arg[1] + 1)))
        else:
            return None
    return _Chars(frozenset(chars), negated=negated)


def _node_chars(op: Any, arg: Any, names: Any) -> tuple[_Chars | None, bool]:
    """What a node of `re._parser` can start with, and whether it can read nothing.

    A node this function does not know gives `None`.
    """
    found: tuple[_Chars | None, bool] = None, False
    if op is names.LITERAL:
        found = _Chars(frozenset(chr(arg))), False
    elif op is names.NOT_LITERAL:
        found = _Chars(frozenset(chr(arg)), negated=True), False
    elif op is names.IN:
        found = _class_chars(arg, names), False
    elif op in (names.ASSERT, names.ASSERT_NOT, names.AT):
        found = _NO_CHARS, True
    elif op is names.SUBPATTERN and not arg[1] & IGNORECASE:
        found = _sequence_chars(arg[3], names)
    elif op is names.ATOMIC_GROUP:
        found = _sequence_chars(arg, names)
    elif op is names.BRANCH:
        branches = [_sequence_chars(branch, names) for branch in arg[1]]
        chars: _Chars | None = _NO_CHARS
        for branch, _ in branches:
            chars = _union(chars, branch)
        found = chars, any(nullable for _, nullable in branches)
    elif op in (names.MAX_REPEAT, names.MIN_REPEAT, names.POSSESSIVE_REPEAT):
        chars, nullable = _sequence_chars(arg[2], names)
        found = chars, nullable or arg[0] == 0
    return found


def _sequence_chars(nodes: Iterable[tuple[Any, Any]], names: Any) -> tuple[_Chars | None, bool]:
    """What a sequence of nodes of `re._parser` can start with, and whether it can read nothing."""
    chars: _Chars | None = _NO_CHARS
    for op, arg in nodes:
        first, nullable = _node_chars(op, arg, names)
        chars = _union(chars, first)
        if not nullable:
            return chars, False
    return chars, True


def _regex_chars(regex: Pattern[str]) -> _Chars | None:
    """The characters a non-empty match of `regex` can start with, `None` when unknown.

    `re._parser` is private, and a construct or a version it does not read as this function expects
    gives `None`: the regex then runs at every position.
    """
    if regex.flags & IGNORECASE:
        return None
    try:
        parser, names = import_module("re._parser"), import_module("re._constants")
        return _sequence_chars(parser.parse(regex.pattern, regex.flags), names)[0]
    except Exception:  # noqa: BLE001 — a private module may change in any way
        return None


def _node(value: _Expr) -> ast.expr:
    if isinstance(value, ast.expr):
        return value
    return ast.Name(value, ast.Load()) if isinstance(value, str) else ast.Constant(value)


def _store(name: str) -> ast.Name:
    return ast.Name(name, ast.Store())


def _call(function: _Expr, *args: _Expr) -> ast.Call:
    return ast.Call(_node(function), [_node(arg) for arg in args], [])


def _attr(value: _Expr, name: str) -> ast.Attribute:
    return ast.Attribute(_node(value), name, ast.Load())


def _item(value: _Expr, index: _Expr) -> ast.Subscript:
    return ast.Subscript(_node(value), _node(index), ast.Load())


def _slice(value: _Expr, start: _Expr, end: _Expr) -> ast.Subscript:
    return ast.Subscript(_node(value), ast.Slice(_node(start), _node(end)), ast.Load())


def _plus(left: _Expr, right: _Expr) -> ast.BinOp:
    return ast.BinOp(_node(left), ast.Add(), _node(right))


def _compare(left: _Expr, op: ast.cmpop, right: _Expr) -> ast.Compare:
    return ast.Compare(_node(left), [op], [_node(right)])


def _class_is(value: _Expr, name: str) -> ast.Compare:
    return _compare(_attr(value, "__class__"), ast.Is(), name)


def _is_found(match: _Expr) -> ast.Compare:
    return _compare(match, ast.IsNot(), None)


def _length() -> ast.Call:
    return _call("len", "T")


def _not(test: _Expr) -> ast.UnaryOp:
    return ast.UnaryOp(ast.Not(), _node(test))


def _all(*tests: ast.expr) -> ast.expr:
    return ast.BoolOp(ast.And(), list(tests)) if len(tests) > 1 else tests[0]


def _any(*tests: ast.expr) -> ast.expr:
    return ast.BoolOp(ast.Or(), list(tests)) if len(tests) > 1 else tests[0]


def _tuple(*items: _Expr) -> ast.Tuple:
    return ast.Tuple([_node(item) for item in items], ast.Load())


def _star(value: ast.expr) -> ast.Starred:
    return ast.Starred(value, ast.Load())


def _undecided() -> ast.Raise:
    return ast.Raise(_node("Undecided"))


def _function(name: str, body: list[ast.stmt]) -> ast.FunctionDef:
    """`def name(T, Y, p0):`."""
    args = [ast.arg("T"), ast.arg("Y"), ast.arg("p0")]
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(posonlyargs=[], args=args, kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body,
        decorator_list=[],
        type_params=[],
    )


@valueclass
class _Value:
    """A value in generated code: an expression, or the expressions of a tuple to build."""

    kind: _Kind
    code: _Expr = None
    parts: tuple[ast.expr, ...] | None = None

    @property
    def expr(self) -> ast.expr:
        """The expression of the value."""
        if self.kind == "ignored":
            return _node("IGNORED")
        if self.parts is not None:
            return _call("_T", _tuple(*self.parts))
        return _node(self.code)


def _read(value: _Value | None) -> ast.expr:
    """The expression of `value`, of an `IgnoredValue` without one."""
    return (value or _Value("ignored")).expr


def _form(p: Parser[Any, Any]) -> tuple[Any, ...]:
    """What `p` is built from; a parser with no known form is run as it is."""
    if p._form is not None:
        return p._form
    return ("test",) if p._test is not None else ("opaque",)


def _target(p: Parser[Any, Any]) -> Parser[Any, Any]:
    """The parser `p` stands for, through `forward_decl()` and `Parser(p)`."""
    seen = set()
    while _form(p)[0] == "ref":
        if id(p) in seen:
            raise _Declined(f"{p.name} is defined as itself")
        seen.add(id(p))
        p = _form(p)[1]
    return p


def _parts(p: Parser[Any, Any]) -> list[Parser[Any, Any]]:
    """The parsers `p` runs."""
    found: list[Parser[Any, Any]]
    match _form(p):
        case ("seq" | "alt" | "but_not" | "counted", left, right):
            found = [left, right]
        case ("map" | "starmap" | "when", inner, _):
            found = [inner]
        case ("ignore" | "many" | "commit" | "context", inner):
            found = [inner]
        case ("oneplus", inner):
            # Generated code runs `inner` from two places: once, then in the loop.
            found = [inner, inner]
        case ("longest", parsers):
            found = list(parsers)
        case ("prefixed", length, body):
            found = [length] if body is None else [length, body]
        case ("switch", key, cases):
            found = [key, *cases.values()]
        case _:
            found = []
    return [_target(q) for q in found]


class _Compiler:
    """Generates the functions of the grammar `root`."""

    def __init__(
        self, root: Parser[Any, Any], internals: Internals, *, values: bool = False
    ) -> None:
        self.root = _target(root)
        self.internals = internals
        self.values = values
        self.names: dict[str, Any] = {
            "IGNORED": internals.ignored,
            "IGV": IgnoredValue,
            "INDEX": index,
            "ParsingFailure": ParsingFailure,
            "State": State,
            "Undecided": internals.undecided,
            "_T": internals.tuple_class,
        }
        self.constants: dict[int, str] = {}
        self.functions: dict[tuple[int, bool], str] = {}
        self.pending: list[tuple[Callable[[_Next], None], bool, str]] = []
        self.tables: dict[str, dict[Any, str]] = {}
        self.body: list[ast.stmt] = []
        self.depth = 1
        self.loops = 0
        self.counter = 0
        self.tests_token_types = False
        self.tests_chars = False
        self.held_chars: dict[str, str] = {}
        self.returns: _Next | None = None

        self.nodes: dict[int, Parser[Any, Any]] = {}
        self.uses: dict[int, int] = {}
        self.cycle_heads: set[int] = set()
        self._walk(self.root, set())
        self.first = self._first_sets()
        self.chars = self._first_chars()
        self.kinds: dict[int, _Kind] = {}
        self.sizes: dict[int, int] = {}
        for p in self.nodes.values():
            # A first-token test, of the loop or around it, can skip the `ValueError` it raises.
            if _form(p)[0] in ("many", "oneplus") and self.first[id(_target(_form(p)[1]))][1]:
                raise _Declined(f"{p.name} repeats a parser that matches no tokens")

    # --- Analysis ---

    def _walk(self, p: Parser[Any, Any], path: set[int]) -> None:
        i = id(p)
        self.uses[i] = self.uses.get(i, 0) + 1
        if i in path:
            self.cycle_heads.add(i)
        elif i not in self.nodes:
            self.nodes[i] = p
            path.add(i)
            for part in _parts(p):
                self._walk(part, path)
            path.discard(i)

    def _first_sets(self) -> dict[int, _First]:
        """The least fixed point of `_first_of`, which recursive grammars need."""
        first: dict[int, _First] = dict.fromkeys(self.nodes, (frozenset(), False))
        changed = True
        while changed:
            changed = False
            for i, p in self.nodes.items():
                new = self._first_of(p, first)
                if new != first[i]:
                    first[i] = new
                    changed = True
        return first

    @staticmethod
    def _first_of(p: Parser[Any, Any], known: dict[int, _First]) -> _First:
        def of(q: Parser[Any, Any]) -> _First:
            return known[id(_target(q))]

        def either(left: _First, right: _First) -> frozenset[str] | None:
            return None if left[0] is None or right[0] is None else left[0] | right[0]

        result: _First
        match _form(p):
            case ("test",):
                token = p._test.token if p._test is not None else None
                result = (frozenset({token[0]}) if token else None), False
            case ("seq", left, right):
                first = of(left)
                result = (either(first, of(right)), of(right)[1]) if first[1] else (first[0], False)
            case ("alt", left, right):
                result = either(of(left), of(right)), of(left)[1] or of(right)[1]
            case ("many", inner):
                result = of(inner)[0], True
            case ("pure", _):
                result = frozenset(), True
            case ("finished",) | ("opaque",) | ("text_end", _):
                result = None, True
            case ("text", _, _, _) | ("byte",) | ("fixed", _, _):
                result = None, False
            case ("previous", _):
                result = None, True
            case ("prefixed" | "counted" | "switch", length, _):
                result = None, of(length)[1]
            case ("longest", parsers):
                found = [of(q) for q in parsers]
                keys = [f for f, _ in found]
                union = None if None in keys else frozenset().union(*(k for k in keys if k))
                result = union, any(nullable for _, nullable in found)
            case (_, inner, *_):
                result = of(inner)
            case _:
                raise AssertionError(f"no first set for form {_form(p)!r}")
        return result

    def _first_chars(self) -> dict[int, _Chars | None]:
        """The characters each parser can start with, the least fixed point of `_chars_of`."""
        regexes: dict[int, _Chars | None] = {}
        for p in self.nodes.values():
            match _form(p):
                case ("text", match, None, _) if isinstance(match.__self__.pattern, str):
                    regexes[id(p)] = _regex_chars(match.__self__)
                case _:
                    pass
        known: dict[int, _Chars | None] = dict.fromkeys(self.nodes, _NO_CHARS)
        changed = True
        while changed:
            changed = False
            for i, p in self.nodes.items():
                new = self._chars_of(p, known, regexes)
                if new != known[i]:
                    known[i] = new
                    changed = True
        return known

    def _chars_of(
        self,
        p: Parser[Any, Any],
        known: dict[int, _Chars | None],
        regexes: dict[int, _Chars | None],
    ) -> _Chars | None:
        def of(q: Parser[Any, Any]) -> _Chars | None:
            return known[id(_target(q))]

        result: _Chars | None = None
        match _form(p):
            case ("text", _, str(literal), _):
                result = _Chars(frozenset(literal[0]))
            case ("text", _, None, _):
                result = regexes.get(id(p))
            case ("seq", left, right):
                nullable = self.first[id(_target(left))][1]
                result = _union(of(left), of(right)) if nullable else of(left)
            case ("alt", left, right):
                result = _union(of(left), of(right))
            case ("longest", parsers):
                result = _NO_CHARS
                for q in parsers:
                    result = _union(result, of(q))
            case ("pure", _):
                result = _NO_CHARS
            case ("many" | "oneplus" | "ignore" | "commit" | "context", inner):
                result = of(inner)
            case ("map" | "starmap" | "when" | "but_not", inner, _):
                result = of(inner)
            case _:
                pass
        return result

    def kind(self, p: Parser[Any, Any]) -> _Kind:
        """What `+` needs to know of the value of `p`."""
        p = _target(p)
        i = id(p)
        if i not in self.kinds:
            self.kinds[i] = "unknown"  # What a parser reached again through a cycle gives.
            self.kinds[i] = self._kind_of(p)
        return self.kinds[i]

    def _kind_of(self, p: Parser[Any, Any]) -> _Kind:
        kind: _Kind = "unknown"
        match _form(p):
            case ("test",) if p._test is not None:
                convs = p._test.convs
                if convs and convs[-1] is self.internals.ignore:
                    kind = "ignored"
                elif all(c is self.internals.token_value for c in convs):
                    kind = "plain"
            case ("ignore", _):
                kind = "ignored"
            case ("many" | "oneplus", _) | ("text", _, _, _) | ("byte",) | ("previous", _):
                kind = "plain"
            case ("fixed", _, _) | ("counted", _, _) | ("prefixed", _, None):
                kind = "plain"
            case ("longest", parsers):
                kinds = {self.kind(q) for q in parsers}
                kind = kinds.pop() if len(kinds) == 1 else "unknown"
            case ("pure", x) if not isinstance(x, (IgnoredValue, self.internals.tuple_class)):
                kind = "plain"
            case ("context" | "commit", inner) | ("when" | "but_not", inner, _):
                kind = self.kind(inner)
            case ("alt", left, right) if self.kind(left) == self.kind(right):
                kind = self.kind(left)
            case ("seq", left, right):
                kind = self._kind_of_sequence(self.kind(left), self.kind(right))
            case _:
                pass
        return kind

    @staticmethod
    def _kind_of_sequence(left: _Kind, right: _Kind) -> _Kind:
        if left == "ignored":
            return right
        if right == "ignored":
            return left
        return "unknown" if "unknown" in (left, right) else "tuple"

    def size(self, p: Parser[Any, Any]) -> int:
        """The parsers `p` inlines, up to the cycles of the grammar."""
        i = id(p)
        if i not in self.sizes:
            self.sizes[i] = 1
            parts = (q for q in _parts(p) if id(q) not in self.cycle_heads)
            self.sizes[i] = 1 + sum(self.size(q) for q in parts)
        return self.sizes[i]

    def has_function(self, p: Parser[Any, Any]) -> bool:
        """Whether code calls `p` rather than inlining it."""
        i = id(p)
        return i in self.cycle_heads or (self.uses[i] > 1 and self.size(p) > _INLINE_SIZE)

    # --- Emission ---

    def add(self, statement: ast.stmt) -> None:
        self.body.append(statement)

    def assign(self, name: str, value: _Expr) -> None:
        self.add(ast.Assign([_store(name)], _node(value)))

    def unpack(self, value: str, at: str, pair: _Expr) -> None:
        """`value, at = pair`."""
        self.add(ast.Assign([ast.Tuple([_store(value), _store(at)], ast.Store())], _node(pair)))

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def constant(self, value: object, prefix: str = "F") -> str:
        """A name of the generated module bound to `value`."""
        name = self.constants.get(id(value))
        if name is None:
            name = self.constants[id(value)] = self.fresh(prefix)
            self.names[name] = value
        return name

    def nested(self, body: Callable[[], None]) -> list[ast.stmt]:
        """The statements `body` emits, one level deeper."""
        outer, self.body = self.body, []
        self.depth += 1
        body()
        self.depth -= 1
        statements, self.body = self.body, outer
        return statements or [ast.Pass()]

    def block(
        self, test: _Expr, body: Callable[[], None], otherwise: Callable[[], None] | None = None
    ) -> None:
        """An `if` running `body`, and `otherwise` in its `else`."""
        then = self.nested(body)
        self.add(ast.If(_node(test), then, self.nested(otherwise) if otherwise else []))

    def loop(self, test: _Expr, body: Callable[[], None]) -> None:
        self.loops += 1
        statements = self.nested(body)
        self.loops -= 1
        self.add(ast.While(_node(test), statements, []))

    def undecided_if(self, test: _Expr) -> None:
        self.add(ast.If(_node(test), [_undecided()], []))

    def literal(self, value: object) -> _Expr:
        """An expression of `value`: a constant for a `str`, which a subclass may override."""
        return ast.Constant(value) if type(value) is str else self.constant(value, "C")

    def bound(self, value: _Value) -> _Value:
        """`value` under a name, for code that reads it twice."""
        if value.kind == "ignored" or (value.parts is None and not isinstance(value.code, ast.AST)):
            return value
        name = self.fresh("v")
        self.assign(name, value.expr)
        return _Value(value.kind, name)

    def start_test(self, p: Parser[Any, Any], pos: str) -> ast.expr | None:
        """The test that `p` can start at `pos`, when it saves running `p`."""
        first, nullable = self.first[id(p)]
        if nullable or _form(p)[0] == "test":
            return None
        if first is None:
            return self.char_test(p, pos)
        if len(first) == 1:
            return _compare(_item("Y", pos), ast.Eq(), self.literal(next(iter(first))))
        return _compare(_item("Y", pos), ast.In(), self.constant(first, "S"))

    def has_char_test(self, p: Parser[Any, Any]) -> bool:
        """Whether a test of the first character saves running `p`, as a literal is as cheap."""
        form = _form(p)
        literal = form[0] == "text" and form[2] is not None
        return self.chars[id(p)] is not None and not self.first[id(p)][1] and not literal

    def char_test(self, p: Parser[Any, Any], pos: str) -> ast.expr | None:
        """The test that the character at `pos` can start `p`, when it saves running `p`."""
        p = _target(p)
        chars = self.chars[id(p)]
        if chars is None or not self.has_char_test(p):
            return None
        self.tests_chars = True
        char: _Expr = self.held_chars.get(pos) or _slice("T", pos, _plus(pos, 1))
        op = ast.NotIn() if chars.negated else ast.In()
        return _compare(char, op, self.constant(chars.chars, "S"))

    @contextmanager
    def holding_char(self, parsers: Iterable[Parser[Any, Any]], pos: str) -> Generator[None]:
        """Names the character at `pos` for the tests of `parsers` in the block."""
        if pos in self.held_chars or not any(self.has_char_test(_target(q)) for q in parsers):
            yield
            return
        name = self.held_chars[pos] = self.fresh("c")
        self.assign(name, _slice("T", pos, _plus(pos, 1)))
        yield
        del self.held_chars[pos]

    def test_condition(self, p: Parser[Any, Any], pos: str) -> ast.expr:
        """Whether the token at `pos` passes the token test `p`."""
        test = p._test
        if test is None:
            raise AssertionError(f"{p.name} tests no token")
        if test.token is None:
            if self.values:
                raise _Declined(f"{p.name} tests a token, and there are only its values")
            read = _call(self.constant(test.pred), _item("T", pos))
            return _all(_compare(pos, ast.Lt(), _length()), read)
        self.tests_token_types = True
        type_, value = test.token
        condition = _compare(_item("Y", pos), ast.Eq(), self.literal(type_))
        if value is None:
            return condition
        return _all(condition, _compare(self.token_value(pos), ast.Eq(), self.literal(value)))

    def token_value(self, pos: str) -> ast.expr:
        """The value of the token at `pos`."""
        return _item("T", pos) if self.values else _attr(_item("T", pos), "value")

    def leaf_start_tests(self, p: Parser[Any, Any], pos: str) -> list[ast.expr] | None:
        """What `start_tests()` gives for a parser that reads the input itself."""
        tests: list[ast.expr] | None = None
        match _form(p):
            case ("test",):
                tests = [self.test_condition(p, pos)]
            case ("text", _, literal, _) if literal is not None:
                tests = [_call(_attr("T", "startswith"), self.literal(literal), pos)]
            case ("text", match, _, _):
                tests = [_is_found(_call(self.constant(match, "R"), "T", pos))]
            case ("text_end", None):
                tests = []
            case _:
                # A parser that runs as it is, or an end moving past what it skips, which counts as
                # reading.
                pass
        return tests

    def start_tests(self, p: Parser[Any, Any], pos: str, seen: set[int]) -> list[ast.expr] | None:
        """The token tests `p` can run at `pos` before it reads a token, `None` when unknown."""
        p = _target(p)
        if id(p) in seen:
            return []
        seen.add(id(p))
        parts: list[Parser[Any, Any]] = []
        match _form(p):
            case ("test",) | ("opaque",) | ("text", _, _, _) | ("text_end", _):
                return self.leaf_start_tests(p, pos)
            case ("byte",) | ("fixed", _, _) | ("prefixed" | "counted" | "switch", _, _):
                return None
            case ("previous", _):
                return []
            case ("longest", found):
                parts = list(found)
            case ("seq", left, right):
                parts = [left, right] if self.first[id(_target(left))][1] else [left]
            case ("alt" | "but_not", left, right):
                parts = [left, right]
            case ("map" | "starmap" | "when", inner, _):
                parts = [inner]
            case ("ignore" | "many" | "oneplus" | "commit" | "context", inner):
                parts = [inner]
            case _:
                pass
        tests: list[ast.expr] = []
        for part in parts:
            found = self.start_tests(part, pos, seen)
            if found is None:
                return None
            tests += found
        return tests

    def emit(
        self, p: Parser[Any, Any], pos: str, want: bool, then: _Next, *, inline: bool = False
    ) -> None:
        """Code running `p` at `pos`; on a success `then` emits what follows, inside the test."""
        p = _target(p)
        too_deep = self.depth > _MAX_DEPTH or self.loops >= _MAX_LOOPS
        if not inline and (self.has_function(p) or (too_deep and _form(p)[0] != "test")):
            self.emit_call(p, pos, want, then)
        else:
            getattr(self, f"emit_{_form(p)[0]}")(p, pos, want, then)

    def after_skip(self, skip: Pattern[str] | None, pos: str) -> str:
        """The position after what `skip` matches at `pos`, `pos` when it is `None`."""
        if skip is None:
            return pos
        m, at = self.fresh("m"), self.fresh("p")
        self.assign(m, _call(self.constant(skip.match, "K"), "T", pos))
        self.assign(at, ast.IfExp(_is_found(m), _call(_attr(m, "end")), _node(pos)))
        return at

    def emit_text(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, match, literal, skip = _form(p)

        def finish(end: str, value: _Expr) -> None:
            then(self.after_skip(skip, end), _Value("plain", value) if want else None)

        if literal is not None:
            text = self.literal(literal)
            size = len(literal)
            self.block(
                _call(_attr("T", "startswith"), text, pos),
                lambda: finish(self.named_position(_plus(pos, size)), text),
            )
            return
        m = self.fresh("m")
        self.assign(m, _call(self.constant(match, "R"), "T", pos))
        self.block(
            _all(_is_found(m), _compare(_call(_attr(m, "end")), ast.Gt(), pos)),
            lambda: finish(self.named_position(_call(_attr(m, "end"))), _call(_attr(m, "group"))),
        )

    def named_position(self, expr: _Expr) -> str:
        at = self.fresh("p")
        self.assign(at, expr)
        return at

    def emit_text_end(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        start = self.after_skip(_form(p)[1], pos)
        self.block(
            _compare(start, ast.GtE(), _length()), lambda: then(start, _Value("plain", None))
        )

    def emit_byte(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        def body() -> None:
            at = self.named_position(_plus(pos, 1))
            then(at, _Value("plain", _item("T", pos)) if want else None)

        self.block(_compare(pos, ast.Lt(), _length()), body)

    def emit_previous(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        before = _item("T", ast.BinOp(_node(pos), ast.Sub(), _node(1)))
        value = _Value("plain", ast.BinOp(before, ast.BitAnd(), _node(self.literal(_form(p)[1]))))
        self.block(_compare(pos, ast.Gt(), 0), lambda: then(pos, value if want else None))

    def emit_fixed(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, unpack, size = _form(p)

        def body() -> None:
            at = self.named_position(_plus(pos, size))
            if want:
                then(at, _Value("plain", _item(_call(self.constant(unpack, "U"), "T", pos), 0)))
            else:
                then(at, None)

        self.block(_compare(_plus(pos, size), ast.LtE(), _length()), body)

    def function_for(self, p: Parser[Any, Any], want: bool) -> str:
        """The name of the function of `p`, which `pending` gets on the first call."""
        p = _target(p)
        key = (id(p), want)
        function = self.functions.get(key)
        if function is None:
            function = self.functions[key] = self.fresh("f")

            def body(result: _Next) -> None:
                self.emit(p, "p0", want, result, inline=True)

            self.pending.append((body, want, function))
        return function

    def emit_prefixed(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, length, body = _form(p)
        if self.values:
            raise _Declined(f"{p.name} reads tokens, and there are only their values")

        def after(at: str, value: _Value | None) -> None:
            size = self.bound(value or _Value("ignored")).expr
            end = self.named_position(_plus(at, size))

            def read() -> None:
                if body is None:
                    then(end, _Value("plain", _slice("T", at, end)) if want else None)
                    return
                function = self.function_for(body, True)
                res, name, stop = self.fresh("r"), self.fresh("v"), self.fresh("p")
                types = _plus(_slice("Y", at, end), ast.List([_node(None)], ast.Load()))
                self.assign(res, _call(function, _slice("T", at, end), types, 0))

                def done() -> None:
                    self.unpack(name, stop, res)
                    kind = self.kind(body)
                    self.block(
                        _compare(stop, ast.Eq(), size),
                        lambda: then(end, _Value(kind, name) if want else None),
                    )

                self.block(_compare(res, ast.IsNot(), None), done)

            fits = _compare(end, ast.LtE(), _length())
            self.block(_all(_compare(size, ast.GtE(), 0), fits), read)

        self.emit(length, pos, True, after)

    def emit_counted(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, length, item = _form(p)

        def after(start: str, value: _Value | None) -> None:
            size, at, acc, count, read = (self.fresh(x) for x in ("n", "p", "a", "c", "ok"))
            # What `range()` does to a count that is no integer.
            self.assign(size, _call("INDEX", _read(value)))
            self.assign(at, start)
            self.assign(acc, ast.List([], ast.Load()))
            self.assign(count, 0)
            self.assign(read, _compare(size, ast.GtE(), 0))
            nullable = self.first[id(_target(item))][1]

            def body() -> None:
                self.add(ast.AugAssign(_store(count), ast.Add(), _node(1)))

                def store(end: str, value: _Value | None) -> None:
                    if nullable:
                        # What `counted()` raises for an item that reads no tokens.
                        self.undecided_if(_compare(end, ast.Eq(), at))
                    self.add(ast.Expr(_call(_attr(acc, "append"), _read(value))))
                    self.assign(at, end)
                    self.add(ast.Continue())

                self.emit(item, at, True, store)
                self.assign(read, False)
                self.add(ast.Break())

            self.loop(_compare(count, ast.Lt(), size), body)
            self.block(read, lambda: then(at, _Value("plain", acc) if want else None))

        self.emit(length, pos, True, after)

    def emit_switch(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, key, table = _form(p)
        name = self.fresh("J")
        self.tables[name] = {k: self.function_for(q, True) for k, q in table.items()}

        def after(end: str, value: _Value | None) -> None:
            k = self.bound(value or _Value("ignored")).expr
            function, res, name_v, at = (self.fresh(x) for x in ("g", "r", "v", "p"))
            self.assign(function, _call(_attr(name, "get"), k))

            def called() -> None:
                self.assign(res, _call(function, "T", "Y", end))

                def got() -> None:
                    self.unpack(name_v, at, res)
                    then(at, _Value("unknown", name_v) if want else None)

                self.block(_compare(res, ast.IsNot(), None), got)

            self.block(_compare(function, ast.IsNot(), None), called)

        self.emit(key, pos, True, after)

    def emit_longest(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        best, name = self.fresh("p"), self.fresh("v")
        self.assign(best, -1)

        def keep(end: str, value: _Value | None) -> None:
            def store() -> None:
                self.assign(best, end)
                if want:
                    self.assign(name, _read(value))

            self.block(_compare(end, ast.Gt(), best), store)

        def attempt(parser: Parser[Any, Any]) -> None:
            test = self.char_test(parser, pos)
            if test is None:
                self.emit(parser, pos, want, keep)
            else:
                self.block(test, lambda: self.emit(parser, pos, want, keep))

        with self.holding_char(_form(p)[1], pos):
            for parser in _form(p)[1]:
                attempt(parser)
        kind = self.kind(p)
        self.block(
            _compare(best, ast.GtE(), 0), lambda: then(best, _Value(kind, name) if want else None)
        )

    def emit_map(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, f = _form(p)
        self.emit_conversion(inner, self.constant(f), pos, want, then)

    def emit_starmap(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, f = _form(p)
        self.emit_conversion(inner, self.constant(f), pos, want, then, star=True)

    def emit_ignore(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.emit(_form(p)[1], pos, False, lambda at, _: then(at, _Value("ignored")))

    def emit_context(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.emit(_form(p)[1], pos, want, then)

    def emit_many(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.emit_loop(_form(p)[1], pos, want, then, None)

    def emit_oneplus(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        inner = _form(p)[1]

        def after(at: str, value: _Value | None) -> None:
            head = (value or _Value("ignored")) if want else None
            self.emit_loop(inner, at, want, then, head)

        self.emit(inner, pos, want, after)

    def emit_finished(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.block(_compare(pos, ast.GtE(), _length()), lambda: then(pos, _Value("plain", None)))

    def emit_call(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        want = want and self.kind(p) != "ignored"
        function = self.function_for(p, want)
        kind = self.kind(p)
        self.emit_function_call(
            function, pos, want, lambda at, value: then(at, _Value(kind, value) if value else None)
        )

    def emit_function_call(
        self, function: str, pos: str, want: bool, then: Callable[[str, str | None], None]
    ) -> None:
        """A call of `function`; `then` gets the position after it and the name of its value."""
        at = self.fresh("p")
        if not want:
            self.assign(at, _call(function, "T", "Y", pos))
            self.block(_compare(at, ast.GtE(), 0), lambda: then(at, None))
            return
        res, value = self.fresh("r"), self.fresh("v")
        self.assign(res, _call(function, "T", "Y", pos))

        def body() -> None:
            self.unpack(value, at, res)
            then(at, value)

        self.block(_compare(res, ast.IsNot(), None), body)

    def emit_test(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        test = p._test
        if test is None:
            raise AssertionError(f"{p.name} tests no token")

        def body() -> None:
            at = self.named_position(_plus(pos, 1))
            expr: _Expr = _item("T", pos)
            kind: _Kind = "plain" if not test.convs else "unknown"
            for conv in test.convs:
                if conv is self.internals.token_value:
                    expr, kind = self.token_value(pos), "plain"
                elif conv is self.internals.ignore:
                    expr, kind = "IGNORED", "ignored"
                else:
                    name = self.fresh("v")
                    self.assign(name, _call(self.constant(conv), expr))
                    expr, kind = name, "unknown"
            then(at, _Value(kind, expr) if want else None)

        self.block(self.test_condition(p, pos), body)

    def emit_pure(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        x = _form(p)[1]
        then(pos, _Value(self.kind(p), None if x is None else self.constant(x, "C")))

    def emit_conversion(
        self,
        inner: Parser[Any, Any],
        function: str,
        pos: str,
        want: bool,
        then: _Next,
        *,
        star: bool = False,
    ) -> None:
        def after(at: str, value: _Value | None) -> None:
            name = self.fresh("v")
            self.assign(name, _call(function, _star(_read(value)) if star else _read(value)))
            then(at, _Value("unknown", name) if want else None)

        self.emit(inner, pos, True, after)

    def emit_when(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, pred = _form(p)

        def after(at: str, value: _Value | None) -> None:
            value = self.bound(value or _Value("ignored"))
            test = _call(self.constant(pred), value.expr)
            self.block(test, lambda: then(at, value if want else None))

        self.emit(inner, pos, True, after)

    def emit_commit(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        inner = _form(p)[1]
        # A failure is final once `inner` read a token, which is when one of its token tests passed
        # at `pos`. The interpreted parser decides that failure, as it has what the error needs.
        done, at, name = self.fresh("ok"), self.fresh("p"), self.fresh("v")
        self.assign(done, False)

        def store(end: str, value: _Value | None) -> None:
            self.assign(at, end)
            if want:
                self.assign(name, _read(value))
            self.assign(done, True)

        self.emit(inner, pos, want, store)
        kind = self.kind(inner)
        body = self.nested(lambda: then(at, _Value(kind, name) if want else None))
        tests = self.start_tests(inner, pos, set())
        orelse: list[ast.stmt] = []
        if tests is None or len(tests) > _MAX_START_TESTS:
            orelse = [_undecided()]
        elif tests:
            distinct = {ast.dump(test): test for test in tests}
            orelse = [ast.If(_any(*distinct.values()), [_undecided()], [])]
        self.add(ast.If(_node(done), body, orelse))

    def emit_but_not(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, other = _form(p)

        def after(at: str, value: _Value | None) -> None:
            kept = self.bound(value) if value is not None else None
            matched = self.fresh("ok")
            self.assign(matched, False)
            self.emit(other, pos, False, lambda _at, _v: self.assign(matched, True))
            self.block(_not(matched), lambda: then(at, kept))

        self.emit(inner, pos, want, after)

    def emit_opaque(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        if self.values:
            raise _Declined(f"{p.name} reads tokens, and there are only their values")
        res = self.fresh("r")
        state = _call("State", pos, pos)
        self.assign(res, _call(_attr(self.constant(p, "P"), "run"), "T", state))

        def failed() -> None:
            self.undecided_if(_attr(res, "committed"))

        def succeeded() -> None:
            self.undecided_if(_attr(_attr(res, "state"), "problems"))
            at, value = self.fresh("p"), self.fresh("v")
            self.assign(at, _attr(_attr(res, "state"), "pos"))
            self.assign(value, _attr(res, "value"))
            then(at, _Value("unknown", value))

        self.block(_class_is(res, "ParsingFailure"), failed, succeeded)

    def emit_seq(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        items: list[Parser[Any, Any]] = []
        left = p
        while _form(left)[0] == "seq":
            items.append(_target(_form(left)[2]))
            left = _target(_form(left)[1])
            if self.has_function(left):
                break
        items.append(left)
        items.reverse()

        if len(items) > _GROUP_SIZE:
            self.emit_groups(items, pos, want, then)
            return

        def step(i: int, at: str, value: _Value | None) -> None:
            if i == len(items):
                then(at, (value or _Value("ignored")) if want else None)
                return

            def after(next_at: str, item: _Value | None) -> None:
                step(i + 1, next_at, self.joined(value, item) if want else None)

            self.emit(items[i], at, want and self.kind(items[i]) != "ignored", after)

        step(0, pos, None)

    def emit_groups(self, items: list[Parser[Any, Any]], pos: str, want: bool, then: _Next) -> None:
        """`items` in sequence, through functions giving the values of `_GROUP_SIZE` of them."""
        groups = [items[i : i + _GROUP_SIZE] for i in range(0, len(items), _GROUP_SIZE)]

        def step(i: int, at: str, value: _Value | None) -> None:
            if i == len(groups):
                then(at, (value or _Value("ignored")) if want else None)
                return
            group = groups[i]
            kinds = [self.kind(item) for item in group if want and self.kind(item) != "ignored"]
            function = self.fresh("f")
            self.pending.append((self.group_body(group, want), bool(kinds), function))

            def after(next_at: str, values: str | None) -> None:
                joined = value
                for n, kind in enumerate(kinds):
                    joined = self.joined(joined, _Value(kind, _item(values, n)))
                step(i + 1, next_at, joined)

            self.emit_function_call(function, at, bool(kinds), after)

        step(0, pos, None)

    def group_body(self, group: list[Parser[Any, Any]], want: bool) -> Callable[[_Next], None]:
        """Code of a function running `group`, which gives the values `+` has to join."""

        def body(result: _Next) -> None:
            def step(i: int, at: str, values: tuple[ast.expr, ...]) -> None:
                if i == len(group):
                    result(at, _Value("plain", _tuple(*values)) if values else None)
                    return
                takes = want and self.kind(group[i]) != "ignored"

                def after(next_at: str, value: _Value | None) -> None:
                    step(i + 1, next_at, (*values, value.expr) if takes and value else values)

                self.emit(group[i], at, takes, after)

            step(0, "p0", ())

        return body

    def joined(self, left: _Value | None, right: _Value | None) -> _Value:
        """The value of `+` over two values, as `ParsingSuccess.__add__()` makes it."""
        right = right or _Value("ignored")
        if left is None or left.kind == "ignored":
            return right
        if right.kind == "ignored":
            return left
        if left.kind != "unknown" and right.kind != "unknown":
            return _Value("tuple", parts=(*self.tuple_parts(left), right.expr))
        b = self.bound(right).expr
        value: ast.expr
        if left.kind == "unknown":
            a = self.bound(left).expr
            extended = _call("_T", _tuple(_star(a), b))
            value = ast.IfExp(_class_is(a, "_T"), extended, _call("_T", _tuple(a, b)))
            value = ast.IfExp(_class_is(a, "IGV"), b, value)
        else:
            a = left.expr
            value = _Value("tuple", parts=(*self.tuple_parts(left), b)).expr
        if right.kind == "unknown":
            value = ast.IfExp(_class_is(b, "IGV"), a, value)
        name = self.fresh("v")
        self.assign(name, value)
        return _Value("unknown", name)

    @staticmethod
    def tuple_parts(value: _Value) -> tuple[ast.expr, ...]:
        """The items a value known to be plain or a tuple gives the tuple `+` builds from it."""
        if value.parts is not None:
            return value.parts
        return (_star(value.expr),) if value.kind == "tuple" else (value.expr,)

    def alternatives(self, parts: tuple[Parser[Any, Any], ...]) -> list[Parser[Any, Any]]:
        """`parts` and, in order, the alternatives inside them that are inlined."""
        found: list[Parser[Any, Any]] = []
        for part in map(_target, parts):
            match _form(part):
                case ("alt", left, right) if not self.has_function(part):
                    found += self.alternatives((left, right))
                case _:
                    found.append(part)
        return found

    def emit_alt(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        alts = self.alternatives(_form(p)[1:])
        kind = self.kind(p)
        at, name = self.fresh("p"), self.fresh("v")

        def store(flag: str | None) -> _Next:
            def stored(end: str, value: _Value | None) -> None:
                self.assign(at, end)
                if want:
                    self.assign(name, _read(value))
                if flag:
                    self.assign(flag, True)

            return stored

        def result() -> None:
            then(at, _Value(kind, name) if want else None)

        def attempt(alt: Parser[Any, Any], flag: str | None, *conditions: ast.expr) -> None:
            def run() -> None:
                self.emit(alt, pos, want, store(flag))

            if conditions:
                self.block(_all(*conditions), run)
            else:
                run()

        last = alts[-1]
        with self.holding_char(alts, pos):
            if then is self.returns:
                # A success leaves the function, so reaching an alternative means the others failed.
                for alt in alts:
                    start = self.start_test(alt, pos)
                    self.emit_returning(alt, pos, want, then, start)
                return
            if len(alts) == 2 and _form(last)[0] == "pure":
                # `maybe(p)`: the default, which `p` overwrites when it succeeds.
                self.emit_pure(last, pos, want, store(None))
                start = self.start_test(alts[0], pos)
                attempt(alts[0], None, *([start] if start else []))
                result()
                return
            done = self.fresh("ok")
            self.assign(done, False)
            for i, alt in enumerate(alts):
                start = self.start_test(alt, pos)
                attempt(alt, done, *([_not(done)] if i else []), *([start] if start else []))
            self.block(done, result)

    def emit_returning(
        self, alt: Parser[Any, Any], pos: str, want: bool, then: _Next, test: ast.expr | None
    ) -> None:
        """`alt` at `pos`, under `test` if there is one; `then` leaves the function."""
        if test is None:
            self.emit(alt, pos, want, then)
        else:
            self.block(test, lambda: self.emit(alt, pos, want, then))

    def emit_loop(
        self, inner: Parser[Any, Any], pos: str, want: bool, then: _Next, head: _Value | None
    ) -> None:
        item = _target(inner)
        at, acc = self.fresh("p"), self.fresh("a")
        self.assign(at, pos)
        if want:
            self.assign(acc, ast.List([head.expr] if head else [], ast.Load()))

        def body() -> None:
            def after(end: str, value: _Value | None) -> None:
                if want:
                    self.add(ast.Expr(_call(_attr(acc, "append"), _read(value))))
                self.assign(at, end)
                self.add(ast.Continue())

            self.emit(item, at, want and self.kind(item) != "ignored", after)
            self.add(ast.Break())

        self.loop(True, body)
        then(at, _Value("plain", acc) if want else None)

    def module(self) -> tuple[ast.Module, str]:
        """The functions as a module, and the name of the one that runs the grammar."""
        entry = self.fresh("f")
        self.functions[id(self.root), True] = entry

        def root(result: _Next) -> None:
            self.emit(self.root, "p0", True, result, inline=True)

        self.pending.append((root, True, entry))
        functions: list[ast.stmt] = []
        while self.pending:
            body, want, name = self.pending.pop()
            self.body, self.depth, self.loops = [], 1, 0

            def result(at: str, value: _Value | None, *, want: bool = want) -> None:
                self.add(ast.Return(_tuple(_read(value), at) if want else _node(at)))

            self.returns = result
            body(result)
            self.add(ast.Return(_node(None if want else -1)))
            functions.append(_function(name, self.body))
        return ast.Module(functions, []), entry


def _number_lines(statements: list[ast.stmt], line: int) -> int:
    """Give `statements` the lines `ast.unparse()` writes them on; the line after them."""
    for statement in statements:
        statement.lineno = statement.end_lineno = line
        line += 1
        match statement:
            case ast.If(body=body, orelse=orelse):
                line = _number_lines(body, line)
                # An `else` around a lone `if` is an `elif`, which has no line of its own.
                if orelse and not (len(orelse) == 1 and isinstance(orelse[0], ast.If)):
                    line += 1
                line = _number_lines(orelse, line)
            case ast.While(body=body) | ast.FunctionDef(body=body):
                line = _number_lines(body, line)
            case _:
                pass
    return line


def _with_lines(module: ast.Module, line: int = 1) -> ast.Module:
    """`module` with the lines of its source from `line`, a blank one between its functions."""
    for function in module.body:
        line = _number_lines([function], line) + 1
    return ast.fix_missing_locations(module)


def _functions(
    root: Parser[Any, Any], internals: Internals, *, values: bool, filename: str | None, line: int
) -> tuple[Callable[[Any, list[str | None], int], Any], bool, bool, str | None] | None:
    """The function running `root` and its source for a `filename`, `None` when it cannot be.

    Between them, whether it reads token types and whether it tests characters of a `str`.
    """
    try:
        compiler = _Compiler(root, internals, values=values)
        module, entry = compiler.module()
        code = compile(_with_lines(module, line), filename or _FILENAME, "exec")
        source = None if filename is None else ast.unparse(module)
    except (_Declined, RecursionError):
        return None
    names = compiler.names
    exec(code, names)  # noqa: S102 — the code is built from the grammar, see ADR 0002
    for table, functions in compiler.tables.items():
        names[table] = {key: names[function] for key, function in functions.items()}
    return names[entry], compiler.tests_token_types, compiler.tests_chars, source


def compile_grammar[A, B](
    root: Parser[A, B], internals: Internals, filename: str | None, line: int
) -> tuple[Callable[[Sequence[A]], tuple[B, int] | None] | None, str | None]:
    """A function giving the value and the end position of a parse, and its source for a `filename`.

    The function returns `None` when the interpreted parser has to decide. Both are `None` for a
    grammar that stays interpreted. Its frames report `filename`, and lines from `line`.
    """
    functions = _functions(root, internals, values=False, filename=filename, line=line)
    if functions is None:
        return None, None
    run, tests_token_types, _, source = functions

    def fast(tokens: Sequence[A]) -> tuple[B, int] | None:
        if tests_token_types:
            if not set(map(type, tokens)) <= _TOKEN_CLASS:
                return None
            types: list[str | None] = list(map(_token_type, tokens))
            types.append(None)
        else:
            types = [None] * (len(tokens) + 1)
        try:
            result: tuple[B, int] | None = run(tokens, types, 0)
        except Exception:  # noqa: BLE001 — whatever it was, the interpreted parser raises it again
            return None
        return result

    return fast, source


def line_after(source: str | None) -> int:
    """The line a program starts on in a file that holds `source` before it."""
    return 1 if source is None else source.count("\n") + 3


def write_source(filename: str | None, *sources: str | None) -> None:
    """Write `sources` to `filename`, a blank line between them; nothing when there are none."""
    texts = [source for source in sources if source is not None]
    if filename is not None and texts:
        Path(filename).write_text("\n\n".join(texts) + "\n", encoding="utf-8")


def reads_text(root: Parser[Any, Any]) -> bool:
    """Whether the grammar reads text or bytes, through `lit()`, `regex()`, a `Text` or
    `funcparserlib.binary`.
    """
    seen: set[int] = set()
    stack = [root]
    try:
        while stack:
            p = _target(stack.pop())
            if id(p) not in seen:
                seen.add(id(p))
                if _form(p)[0] in _READS_INPUT:
                    return True
                stack += _parts(p)
    except _Declined:
        return False
    return False


def leading_skip(root: Parser[Any, Any]) -> Pattern[str] | None:
    """The first `skip` of a `Text` in the grammar, which text may start with."""
    seen: set[int] = set()
    stack = [root]
    try:
        while stack:
            p = _target(stack.pop())
            if id(p) not in seen:
                seen.add(id(p))
                match _form(p):
                    case ("text", _, _, skip) | ("text_end", skip) if skip is not None:
                        found: Pattern[str] = skip
                        return found
                    case ("prefixed", length, _):
                        stack.append(length)
                    case _:
                        stack += reversed(_parts(p))
    except _Declined:
        return None
    return None


def reads_tokens(root: Parser[Any, Any]) -> bool:
    """Whether the grammar reads tokens through `tok()`, which text does not have."""
    seen: set[int] = set()
    stack = [root]
    try:
        while stack:
            p = _target(stack.pop())
            if id(p) not in seen:
                seen.add(id(p))
                if p._test is not None and p._test.token is not None:
                    return True
                stack += _parts(p)
    except _Declined:
        return True
    return False


def compile_text[B](
    root: Parser[Any, B],
    internals: Internals,
    scan: Callable[[str], tuple[list[str], list[str | None]] | None] | None,
    filename: str | None,
) -> tuple[Callable[[str], tuple[B, int] | None] | None, str | None]:
    """A function giving the value and the end position of a text, and its source for a `filename`.

    The function returns `None` when the text does not parse, and raises `internals.undecided` when
    the interpreted parser has to decide. Both are `None` for a grammar that stays interpreted.

    `scan` splits the text into the values and the types of its tokens, without `Token` objects: a
    grammar that reads a token any other way than through `tok()` then stays interpreted. Without
    `scan` the grammar reads the text itself. Its frames report `filename`.
    """
    functions = _functions(root, internals, values=scan is not None, filename=filename, line=1)
    if functions is None:
        return None, None
    run, _, tests_chars, source = functions
    skip = leading_skip(root)

    def fast(text: str) -> tuple[B, int] | None:
        # A character test is false for `bytes`, where the parser raises `TypeError`.
        if tests_chars and type(text) is not str:
            raise internals.undecided
        try:
            if scan is None:
                m = skip.match(text) if skip is not None else None
                result: tuple[B, int] | None = run(text, [], m.end() if m else 0)
            else:
                scanned = scan(text)
                result = None if scanned is None else run(scanned[0], scanned[1], 0)
        except Exception as e:
            raise internals.undecided from e
        return result

    return fast, source
