"""`Parser.compile()`: a grammar as generated Python functions, for the parses that succeed.

The generated code keeps a position in a local variable and tracks no error state, see ADR 0002. A
parser becomes nested `if` statements: code for what follows a parser runs inside the test that it
succeeded, and a failure falls out of the block. Each position and value has a name of its own, so
backtracking needs no reset.

Every function takes the tokens `T`, their types `Y` and a position. `Y` ends in `None`, so testing
it past the last token fails. A function returns `(value, position)` or `None`, or for a parser
whose value nothing reads, the position or `-1`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from operator import attrgetter
from typing import Any, Literal

from ._commons import valueclass
from .lexer import Token
from .parser import IgnoredValue, Parser, ParsingError, State

type _Kind = Literal["ignored", "plain", "tuple", "unknown"]
"""What `+` needs to know of a value: an `IgnoredValue`, a `_Tuple`, neither, or any of them."""

type _Next = Callable[[str, _Value | None], None]
"""Emits what follows a parser, given the position after it and its value, if asked for."""

type _First = tuple[frozenset[str] | None, bool]
"""The token types a parser can start with, `None` when unknown, and whether it can read none."""

_INLINE_SIZE = 10
"""A parser used more than once gets a function of its own when it is larger than this."""

_MAX_INDENT = 60
"""Code nested deeper goes to a function of its own: CPython compiles 100 levels of indentation."""

_MAX_LOOPS = 15
"""Loops nested deeper go to a function of their own: CPython compiles 20 nested blocks."""

_GROUP_SIZE = 32
"""A longer sequence is split into functions of this many parsers, as each nests the next."""

_MAX_START_TESTS = 16
"""A `commit()` with more token tests at its start leaves its failures to the interpreted parser."""

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
class _Value:
    """A value in generated code: an expression, or the expressions of a tuple to build."""

    kind: _Kind
    text: str | None = None
    parts: tuple[str, ...] | None = None

    @property
    def expr(self) -> str:
        """The expression of the value."""
        if self.kind == "ignored":
            return "IGNORED"
        if self.parts is not None:
            return f"_T(({', '.join(self.parts)},))"
        return str(self.text)


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
    match _form(p):
        case ("seq" | "alt" | "but_not", left, right):
            return [_target(left), _target(right)]
        case ("map" | "starmap" | "check" | "when", inner, _):
            return [_target(inner)]
        case ("ignore" | "many" | "oneplus" | "commit" | "context", inner):
            return [_target(inner)]
        case ("longest", parsers):
            return [_target(q) for q in parsers]
        case _:
            return []


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
            "ParsingError": ParsingError,
            "State": State,
            "Undecided": internals.undecided,
            "_T": internals.tuple_class,
        }
        self.constants: dict[int, str] = {}
        self.functions: dict[tuple[int, bool], str] = {}
        self.pending: list[tuple[Callable[[_Next], None], bool, str]] = []
        self.lines: list[str] = []
        self.indent = 1
        self.loops = 0
        self.counter = 0
        self.tests_token_types = False

        self.nodes: dict[int, Parser[Any, Any]] = {}
        self.uses: dict[int, int] = {}
        self.cycle_heads: set[int] = set()
        self._walk(self.root, set())
        self.first = self._first_sets()
        self.kinds: dict[int, _Kind] = {}
        self.sizes: dict[int, int] = {}
        for p in self.nodes.values():
            if _form(p)[0] in ("many", "oneplus"):
                first, nullable = self.first[id(_target(_form(p)[1]))]
                # A first-token test can skip the loop, and the `ValueError` it raises.
                if nullable and first is not None:
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
            case ("text", _, _, _):
                result = None, False
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
            case ("many" | "oneplus", _) | ("text", _, _, _):
                kind = "plain"
            case ("longest", parsers):
                kinds = {self.kind(q) for q in parsers}
                kind = kinds.pop() if len(kinds) == 1 else "unknown"
            case ("pure", x) if not isinstance(x, (IgnoredValue, self.internals.tuple_class)):
                kind = "plain"
            case ("context" | "commit", inner) | ("check" | "when" | "but_not", inner, _):
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

    def line(self, code: str) -> None:
        self.lines.append("    " * self.indent + code)

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

    def block(self, head: str, body: Callable[[], None]) -> None:
        self.line(head)
        self.indent += 1
        before = len(self.lines)
        body()
        if len(self.lines) == before:
            self.line("pass")
        self.indent -= 1

    def literal(self, value: object) -> str:
        """An expression of `value`: `repr()` for a `str`, which a subclass may override."""
        return repr(value) if type(value) is str else self.constant(value, "C")

    def bound(self, value: _Value) -> _Value:
        """`value` under a name, for code that reads it twice."""
        if value.kind == "ignored":
            return value
        if value.text is not None and value.text.isidentifier():
            return value
        name = self.fresh("v")
        self.line(f"{name} = {value.expr}")
        return _Value(value.kind, name)

    def start_test(self, p: Parser[Any, Any], pos: str) -> str | None:
        """The test that `p` can start at `pos`, when it saves running `p`."""
        first, nullable = self.first[id(p)]
        if first is None or nullable or _form(p)[0] == "test":
            return None
        if len(first) == 1:
            return f"Y[{pos}] == {self.literal(next(iter(first)))}"
        return f"Y[{pos}] in {self.constant(first, 'S')}"

    def test_condition(self, p: Parser[Any, Any], pos: str) -> str:
        """Whether the token at `pos` passes the token test `p`."""
        test = p._test
        if test is None:
            raise AssertionError(f"{p.name} tests no token")
        if test.token is None:
            if self.values:
                raise _Declined(f"{p.name} tests a token, and there are only its values")
            return f"{pos} < len(T) and {self.constant(test.pred)}(T[{pos}])"
        self.tests_token_types = True
        type_, value = test.token
        condition = f"Y[{pos}] == {self.literal(type_)}"
        if value is not None:
            condition += f" and {self.token_value(pos)} == {self.literal(value)}"
        return condition

    def token_value(self, pos: str) -> str:
        """The value of the token at `pos`."""
        return f"T[{pos}]" if self.values else f"T[{pos}].value"

    def start_tests(self, p: Parser[Any, Any], pos: str, seen: set[int]) -> list[str] | None:
        """The token tests `p` can run at `pos` before it reads a token, `None` when unknown."""
        p = _target(p)
        if id(p) in seen:
            return []
        seen.add(id(p))
        parts: list[Parser[Any, Any]] = []
        match _form(p):
            case ("test",):
                return [self.test_condition(p, pos)]
            case ("opaque",) | ("text", _, _, _):
                return None
            case ("longest", found):
                parts = list(found)
            case ("seq", left, right):
                parts = [left, right] if self.first[id(_target(left))][1] else [left]
            case ("alt" | "but_not", left, right):
                parts = [left, right]
            case ("map" | "starmap" | "check" | "when", inner, _):
                parts = [inner]
            case ("ignore" | "many" | "oneplus" | "commit" | "context", inner):
                parts = [inner]
            case _:
                pass
        tests: list[str] = []
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
        too_deep = self.indent > _MAX_INDENT or self.loops >= _MAX_LOOPS
        if not inline and (self.has_function(p) or (too_deep and _form(p)[0] != "test")):
            self.emit_call(p, pos, want, then)
        else:
            getattr(self, f"emit_{_form(p)[0]}")(p, pos, want, then)

    def after_skip(self, skip: object, pos: str) -> str:
        """The position after what `skip` matches at `pos`, `pos` when it is `None`."""
        if skip is None:
            return pos
        m, at = self.fresh("m"), self.fresh("p")
        self.line(f"{m} = {self.constant(skip, 'K')}(T, {pos})")
        self.line(f"{at} = {m}.end() if {m} is not None else {pos}")
        return at

    def emit_text(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, match, literal, skip = _form(p)
        start = self.fresh("p")
        # Nothing has skipped what comes before the first token.
        self.line(f"{start} = {pos}")
        if skip is not None:
            self.block(
                f"if {pos} == 0:", lambda: self.line(f"{start} = {self.after_skip(skip, pos)}")
            )

        def finish(end: str, value: str) -> None:
            at = self.after_skip(skip, end)
            then(at, _Value("plain", value) if want else None)

        if literal is not None:
            text = self.literal(literal)
            size = len(literal)
            self.block(
                f"if T.startswith({text}, {start}):",
                lambda: finish(self.named_position(f"{start} + {size}"), text),
            )
            return
        m = self.fresh("m")
        self.line(f"{m} = {self.constant(match, 'R')}(T, {start})")
        self.block(
            f"if {m} is not None and {m}.end() > {start}:",
            lambda: finish(self.named_position(f"{m}.end()"), f"{m}.group()"),
        )

    def named_position(self, expr: str) -> str:
        at = self.fresh("p")
        self.line(f"{at} = {expr}")
        return at

    def emit_text_end(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        skip = _form(p)[1]
        start = self.fresh("p")
        self.line(f"{start} = {pos}")
        if skip is not None:
            self.block(
                f"if {pos} == 0:", lambda: self.line(f"{start} = {self.after_skip(skip, pos)}")
            )
        self.block(f"if {start} >= len(T):", lambda: then(start, _Value("plain", "None")))

    def emit_longest(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        best, name = self.fresh("p"), self.fresh("v")
        self.line(f"{best} = -1")

        def keep(end: str, value: _Value | None) -> None:
            def store() -> None:
                self.line(f"{best} = {end}")
                if want:
                    self.line(f"{name} = {value.expr if value else 'IGNORED'}")

            self.block(f"if {end} > {best}:", store)

        for parser in _form(p)[1]:
            self.emit(parser, pos, want, keep)
        kind = self.kind(p)
        self.block(f"if {best} >= 0:", lambda: then(best, _Value(kind, name) if want else None))

    def emit_map(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, f = _form(p)
        self.emit_conversion(inner, self.constant(f) + "({})", pos, want, then)

    def emit_starmap(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, f = _form(p)
        self.emit_conversion(inner, self.constant(f) + "(*{})", pos, want, then)

    def emit_ignore(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.emit(_form(p)[1], pos, False, lambda at, _: then(at, _Value("ignored")))

    def emit_context(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.emit(_form(p)[1], pos, want, then)

    def emit_many(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.emit_loop(_form(p)[1], pos, want, then, None)

    def emit_oneplus(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        inner = _form(p)[1]
        self.emit(inner, pos, want, lambda at, v: self.emit_loop(inner, at, want, then, v))

    def emit_finished(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        self.block(f"if {pos} >= len(T):", lambda: then(pos, _Value("plain", "None")))

    def emit_call(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        want = want and self.kind(p) != "ignored"
        key = (id(p), want)
        function = self.functions.get(key)
        if function is None:
            function = self.functions[key] = self.fresh("f")

            def body(result: _Next) -> None:
                self.emit(p, "p0", want, result, inline=True)

            self.pending.append((body, want, function))
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
            self.line(f"{at} = {function}(T, Y, {pos})")
            self.block(f"if {at} >= 0:", lambda: then(at, None))
            return
        res, value = self.fresh("r"), self.fresh("v")
        self.line(f"{res} = {function}(T, Y, {pos})")

        def body() -> None:
            self.line(f"{value}, {at} = {res}")
            then(at, value)

        self.block(f"if {res} is not None:", body)

    def emit_test(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        test = p._test
        if test is None:
            raise AssertionError(f"{p.name} tests no token")

        def body() -> None:
            at = self.fresh("p")
            self.line(f"{at} = {pos} + 1")
            expr = f"T[{pos}]"
            kind: _Kind = "plain" if not test.convs else "unknown"
            for conv in test.convs:
                if conv is self.internals.token_value:
                    expr, kind = self.token_value(pos), "plain"
                elif conv is self.internals.ignore:
                    expr, kind = "IGNORED", "ignored"
                else:
                    name = self.fresh("v")
                    self.line(f"{name} = {self.constant(conv)}({expr})")
                    expr, kind = name, "unknown"
            then(at, _Value(kind, expr) if want else None)

        self.block(f"if {self.test_condition(p, pos)}:", body)

    def emit_pure(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        x = _form(p)[1]
        then(pos, _Value(self.kind(p), "None" if x is None else self.constant(x, "C")))

    def emit_conversion(
        self, inner: Parser[Any, Any], call: str, pos: str, want: bool, then: _Next
    ) -> None:
        def after(at: str, value: _Value | None) -> None:
            name = self.fresh("v")
            self.line(f"{name} = {call.format(value.expr if value else 'IGNORED')}")
            then(at, _Value("unknown", name) if want else None)

        self.emit(inner, pos, True, after)

    def emit_check(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, pred = _form(p)

        def after(at: str, value: _Value | None) -> None:
            value = self.bound(value or _Value("ignored"))
            condition = f"if not {self.constant(pred)}({value.expr}):"
            self.block(condition, lambda: self.line("raise Undecided"))
            then(at, value if want else None)

        self.emit(inner, pos, True, after)

    def emit_when(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, pred = _form(p)

        def after(at: str, value: _Value | None) -> None:
            value = self.bound(value or _Value("ignored"))
            condition = f"if {self.constant(pred)}({value.expr}):"
            self.block(condition, lambda: then(at, value if want else None))

        self.emit(inner, pos, True, after)

    def emit_commit(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        inner = _form(p)[1]
        # A failure is final once `inner` read a token, which is when one of its token tests passed
        # at `pos`. The interpreted parser decides that failure, as it has what the error needs.
        done, at, name = self.fresh("ok"), self.fresh("p"), self.fresh("v")
        self.line(f"{done} = False")

        def store(end: str, value: _Value | None) -> None:
            self.line(f"{at} = {end}")
            if want:
                self.line(f"{name} = {value.expr if value else 'IGNORED'}")
            self.line(f"{done} = True")

        self.emit(inner, pos, want, store)
        kind = self.kind(inner)
        self.block(f"if {done}:", lambda: then(at, _Value(kind, name) if want else None))
        tests = self.start_tests(inner, pos, set())
        if tests is None or len(tests) > _MAX_START_TESTS:
            self.block("else:", lambda: self.line("raise Undecided"))
        elif tests:
            read = " or ".join(f"({test})" for test in dict.fromkeys(tests))
            self.block(f"elif {read}:", lambda: self.line("raise Undecided"))

    def emit_but_not(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        _, inner, other = _form(p)

        def after(at: str, value: _Value | None) -> None:
            kept = self.bound(value) if value is not None else None
            matched = self.fresh("ok")
            self.line(f"{matched} = False")
            self.emit(other, pos, False, lambda _at, _v: self.line(f"{matched} = True"))
            self.block(f"if not {matched}:", lambda: then(at, kept))

        self.emit(inner, pos, want, after)

    def emit_opaque(self, p: Parser[Any, Any], pos: str, want: bool, then: _Next) -> None:
        if self.values:
            raise _Declined(f"{p.name} reads tokens, and there are only their values")
        res = self.fresh("r")
        self.line(f"{res} = {self.constant(p, 'P')}.run(T, State({pos}, {pos}))")

        def failed() -> None:
            self.block(f"if {res}.committed:", lambda: self.line("raise Undecided"))

        def succeeded() -> None:
            self.block(f"if {res}.state.problems:", lambda: self.line("raise Undecided"))
            at, value = self.fresh("p"), self.fresh("v")
            self.line(f"{at} = {res}.state.pos")
            self.line(f"{value} = {res}.value")
            then(at, _Value("unknown", value))

        self.block(f"if {res}.__class__ is ParsingError:", failed)
        self.block("else:", succeeded)

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
                    joined = self.joined(joined, _Value(kind, f"{values}[{n}]"))
                step(i + 1, next_at, joined)

            self.emit_function_call(function, at, bool(kinds), after)

        step(0, pos, None)

    def group_body(self, group: list[Parser[Any, Any]], want: bool) -> Callable[[_Next], None]:
        """Code of a function running `group`, which gives the values `+` has to join."""

        def body(result: _Next) -> None:
            def step(i: int, at: str, values: tuple[str, ...]) -> None:
                if i == len(group):
                    tuple_ = "".join(f"{value}, " for value in values)
                    result(at, _Value("plain", f"({tuple_})") if values else None)
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
        if left.kind == "unknown":
            a = self.bound(left).expr
            value = f"_T((*{a}, {b})) if {a}.__class__ is _T else _T(({a}, {b}))"
            value = f"{b} if {a}.__class__ is IGV else {value}"
        else:
            a = left.expr
            value = _Value("tuple", parts=(*self.tuple_parts(left), b)).expr
        if right.kind == "unknown":
            value = f"{a} if {b}.__class__ is IGV else {value}"
        name = self.fresh("v")
        self.line(f"{name} = {value}")
        return _Value("unknown", name)

    @staticmethod
    def tuple_parts(value: _Value) -> tuple[str, ...]:
        """The items a value known to be plain or a tuple gives the tuple `+` builds from it."""
        if value.parts is not None:
            return value.parts
        return (f"*{value.expr}",) if value.kind == "tuple" else (value.expr,)

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
                self.line(f"{at} = {end}")
                if want:
                    self.line(f"{name} = {value.expr if value else 'IGNORED'}")
                if flag:
                    self.line(f"{flag} = True")

            return stored

        def result() -> None:
            then(at, _Value(kind, name) if want else None)

        def attempt(alt: Parser[Any, Any], flag: str | None, *conditions: str) -> None:
            def run() -> None:
                self.emit(alt, pos, want, store(flag))

            if conditions:
                self.block(f"if {' and '.join(conditions)}:", run)
            else:
                run()

        last = alts[-1]
        if len(alts) == 2 and _form(last)[0] == "pure":
            # `maybe(p)`: the default, which `p` overwrites when it succeeds.
            self.emit_pure(last, pos, want, store(None))
            start = self.start_test(alts[0], pos)
            attempt(alts[0], None, *([start] if start else []))
            result()
            return
        done = self.fresh("ok")
        self.line(f"{done} = False")
        for i, alt in enumerate(alts):
            start = self.start_test(alt, pos)
            attempt(alt, done, *([f"not {done}"] if i else []), *([start] if start else []))
        self.block(f"if {done}:", result)

    def emit_loop(
        self, inner: Parser[Any, Any], pos: str, want: bool, then: _Next, head: _Value | None
    ) -> None:
        item = _target(inner)
        at, acc = self.fresh("p"), self.fresh("a")
        self.line(f"{at} = {pos}")
        if want:
            self.line(f"{acc} = [{head.expr if head else ''}]")
        nullable = self.first[id(item)][1]

        def body() -> None:
            def after(end: str, value: _Value | None) -> None:
                if nullable:
                    # What `many()` raises for a parser that reads no tokens.
                    self.block(f"if {end} == {at}:", lambda: self.line("raise Undecided"))
                if want:
                    self.line(f"{acc}.append({value.expr if value else 'IGNORED'})")
                self.line(f"{at} = {end}")
                self.line("continue")

            self.emit(item, at, want and self.kind(item) != "ignored", after)
            self.line("break")

        self.loops += 1
        self.block("while True:", body)
        self.loops -= 1
        then(at, _Value("plain", acc) if want else None)

    def source(self) -> tuple[str, str]:
        """The source of the functions, and the name of the one that runs the grammar."""
        entry = self.fresh("f")
        self.functions[id(self.root), True] = entry

        def root(result: _Next) -> None:
            self.emit(self.root, "p0", True, result, inline=True)

        self.pending.append((root, True, entry))
        functions: list[str] = []
        while self.pending:
            body, want, name = self.pending.pop()
            self.lines, self.indent, self.loops = [], 1, 0

            def result(at: str, value: _Value | None, *, want: bool = want) -> None:
                self.line(f"return ({value.expr}, {at})" if want and value else f"return {at}")

            body(result)
            self.line("return None" if want else "return -1")
            functions.append(f"def {name}(T, Y, p0):\n" + "\n".join(self.lines))
        return "\n\n".join(functions) + "\n", entry


def _functions(
    root: Parser[Any, Any], internals: Internals, *, values: bool
) -> tuple[Callable[[Any, list[str | None], int], Any], bool] | None:
    """The function running `root`, and whether it reads token types; `None` when it cannot be."""
    try:
        compiler = _Compiler(root, internals, values=values)
        source, entry = compiler.source()
        code = compile(source, "<funcparserlib grammar>", "exec")
    except (_Declined, SyntaxError, RecursionError, MemoryError):
        # A `SyntaxError` or `MemoryError` is nesting past what CPython compiles.
        return None
    exec(code, compiler.names)  # noqa: S102 — the source is built from the grammar, see ADR 0002
    return compiler.names[entry], compiler.tests_token_types


def compile_grammar[A, B](
    root: Parser[A, B], internals: Internals
) -> Callable[[Sequence[A]], tuple[B, int] | None] | None:
    """A function giving the value and the end position of a parse the generated code decides.

    The function returns `None` when the interpreted parser has to decide. `compile_grammar` returns
    `None` for a grammar that stays interpreted.
    """
    functions = _functions(root, internals, values=False)
    if functions is None:
        return None
    run, tests_token_types = functions

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

    return fast


def reads_text(root: Parser[Any, Any]) -> bool:
    """Whether the grammar reads text through `funcparserlib.text.Text`."""
    seen: set[int] = set()
    stack = [root]
    try:
        while stack:
            p = _target(stack.pop())
            if id(p) not in seen:
                seen.add(id(p))
                if _form(p)[0] in ("text", "text_end"):
                    return True
                stack += _parts(p)
    except _Declined:
        return False
    return False


def compile_text[B](
    root: Parser[Any, B],
    internals: Internals,
    scan: Callable[[str], tuple[list[str], list[str | None]] | None] | None,
) -> Callable[[str], tuple[B, int] | None] | None:
    """A function giving the value and the end position of a text, `None` when it does not parse.

    `scan` splits the text into the values and the types of its tokens, without `Token` objects:
    this returns `None` for a grammar that reads a token any other way than through `tok()`. Without
    `scan` the grammar reads the text itself. The function raises `internals.undecided` when the
    interpreted parser has to decide.
    """
    functions = _functions(root, internals, values=scan is not None)
    if functions is None:
        return None
    run, _ = functions

    def fast(text: str) -> tuple[B, int] | None:
        try:
            if scan is None:
                result: tuple[B, int] | None = run(text, [], 0)
            else:
                scanned = scan(text)
                result = None if scanned is None else run(scanned[0], scanned[1], 0)
        except Exception as e:
            raise internals.undecided from e
        return result

    return fast
