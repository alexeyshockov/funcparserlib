# Copyright © 2009/2023 Andrey Vlasovskikh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Functional parsing combinators.

Parsing combinators define an internal domain-specific language (DSL) for describing the parsing
rules of a grammar. The DSL allows you to start with a few primitive parsers, then combine your
parsers to get more complex ones, and finally cover the whole grammar you want to parse.

The structure of the language:

* Class `Parser`
    * All the primitives and combinators of the language return `Parser` objects
    * It defines the main `Parser.parse(tokens)` method
* Primitive parsers
    * `tok(type, value)`, `a(value)`, `some(pred)`, `forward_decl()`, `finished`
* Parser combinators
    * `p1 + p2`, `p1 | p2`, `p >> f`, `-p`, `maybe(p)`, `many(p)`, `oneplus(p)`, `skip(p)`
* Abstraction
    * Use regular Python variables `p = ...  # Expression of type Parser` to define new rules
      (non-terminals) of your grammar

Every time you apply one of the combinators, you get a new `Parser` object. In other words, the set
of `Parser` objects is closed under the means of combination.

!!! Note

    We took the parsing combinators language from the book [Introduction to Functional
    Programming][1] and translated it from ML into Python.

  [1]: https://www.cl.cam.ac.uk/teaching/Lectures/funprog-jrh-1996/
"""

from __future__ import annotations

__all__ = [
    "some",
    "pure",
    "a",
    "maybe",
    "many",
    "oneplus",
    "skip",
    "finished",
    "forward_decl",
    "anything_but",
    "tok",
    "Parser",
    "ForwardDeclParser",
    "State",
    "NoParseError",
    "ParsingResult",
    "ParsingSuccess",
    "ParsingError",
    "IgnoredValue",
    "IgnoredParser",
    "EndOfInput",
    "IgnoredEndOfInput",
    "Tuple2Parser",
    "Tuple3Parser",
    "Tuple4Parser",
    "Tuple5Parser",
    "ParserFn",
    "ParserObjOrFn",
    "parser",
    "commit",
    "TokenDescription",
    "ParseErrors",
    "Expectation",
    "CheckFailure",
    "debug",
]

import logging
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NoReturn, Protocol, Self, TypeGuard, cast, final, overload

from ._commons import valueclass, with_hint
from .lexer import Place, Token

_log = logging.getLogger("funcparserlib")

debug = False


@final
@valueclass
class TokenDescription:
    """How an error shows a token: its `text`, and where it is when the token knows.

    `start` and `end` are the (_line_, _column_) of its first and last characters, as
    `funcparserlib.lexer.Token` has them. `Parser.parse(tokens, describe=...)` takes a function from
    a token to its description; `NoParseError.render()` uses the positions to point at it.
    """

    text: str
    start: Place | None = None
    end: Place | None = None


def _describe(token: object) -> TokenDescription:
    """The default description: the `value`, `start` and `end` a token has, the token otherwise."""
    value = getattr(token, "value", token)
    text = repr(value) if isinstance(value, str) else str(value)
    start, end = getattr(token, "start", None), getattr(token, "end", None)
    if _is_place(start) and _is_place(end):
        return TokenDescription(text, start, end)
    return TokenDescription(text)


def _is_place(x: object) -> TypeGuard[Place]:
    return isinstance(x, tuple) and len(x) == 2 and all(isinstance(n, int) for n in x)


type Expectation = tuple[Parser[Any, Any], tuple[str, ...]]
"""A parser that failed at the furthest position, with the `Parser.context()` labels around it."""


@final
@valueclass
class CheckFailure:
    """A failed `Parser.check()`: its message, the tokens it covers (`start` to `end`, exclusive)
    and the `Parser.context()` labels around it.
    """

    message: str
    start: int
    end: int
    context: tuple[str, ...]


@final
@valueclass(repr=False)
class State:
    """Parsing state: where parsing is, and what an error message needs.

    `pos` is the position in the sequence being parsed, and `max` the furthest position a parser
    reached. `expected` holds the parsers that failed at `max`, each with the `Parser.context()`
    labels around it; `context` holds the labels around `pos`; `problems` holds the non-fatal errors
    `Parser.check()` recorded on the way here.
    """

    pos: int
    max: int
    expected: tuple[Expectation, ...] = ()
    context: tuple[str, ...] = ()
    problems: tuple[CheckFailure, ...] = ()

    @property
    def parser(self) -> Parser[Any, Any] | None:
        """The last parser that failed at `max`, if any."""
        return self.expected[-1][0] if self.expected else None

    def expect(self, p: Parser[Any, Any]) -> State:
        """This state with `p` among the parsers that failed, unless a parser got further."""
        if self.pos == self.max:
            return State(
                self.pos, self.max, (*self.expected, (p, self.context)), self.context, self.problems
            )
        return self

    def reset_pos_to(self, s: State) -> State:
        """`s`, with the furthest position and what failed there from this state."""
        # dataclasses.replace() hits performance, as it performs additional validation
        return State(s.pos, self.max, self.expected, s.context, s.problems)

    def reset_pos(self, pos: int) -> State:
        """This state at `pos`; reaching past `max` moves it, and forgets what failed there."""
        if pos > self.max:
            return State(pos, pos, (), self.context, self.problems)
        return State(pos, self.max, self.expected, self.context, self.problems)

    def replace_rest_from(self, s: State) -> State:
        """This state, with the furthest position and what failed there from `s`."""
        return State(self.pos, s.max, s.expected, self.context, self.problems)

    def __str__(self) -> str:
        return str((self.pos, self.max))

    def __repr__(self) -> str:
        return f"State({self.pos!r}, {self.max!r})"


class NoParseError(Exception):
    """Raised by `Parser.parse()` when the tokens do not match the grammar.

    `msg` is what `str()` returns: where parsing stopped, what it found and what it expected.
    `token` is the token parsing stopped at, `None` at the end of input; `expected` names the
    parsers that failed there, in the order they were tried; `context` names the `Parser.context()`
    labels around them, innermost first; `description` is the token's `TokenDescription`, `None`
    when there is none; `state` is where parsing stopped.

    `render(source)` adds the source line and a marker under the token, when the token's description
    has a position.
    """

    _result: ParsingError | None = None

    def __init__(
        self,
        msg: str,
        state: State,
        *,
        token: object = None,
        expected: tuple[str, ...] = (),
        context: tuple[str, ...] = (),
        description: TokenDescription | None = None,
    ) -> None:
        self.msg: str = msg
        self.state: State = state
        self.token: object = token
        self.expected: tuple[str, ...] = expected
        self.context: tuple[str, ...] = context
        self.description: TokenDescription | None = description

    def __str__(self) -> str:
        return self.msg

    def render(self, source: str | None = None) -> str:
        """The message, then the line of `source` the token is on with `^` under the token.

        Without `source`, or without a position for the token, it is the message alone.
        """
        d = self.description
        if source is None or d is None or d.start is None:
            return self.msg
        line, column = d.start
        lines = source.split("\n")
        if not 1 <= line <= len(lines):
            return self.msg
        text = lines[line - 1].rstrip("\r")
        last = d.end[1] if d.end is not None and d.end[0] == line else len(text)
        gutter = f"{line} | "
        marker = " " * (len(gutter) + column - 1) + "^" * max(1, last - column + 1)
        return f"{self.msg}\n{gutter}{text}\n{marker}"


@final
class ParseErrors(ExceptionGroup[NoParseError]):
    """Raised by `Parser.parse()` when `Parser.check()` recorded errors: all of them, by position.

    A parse that also failed has its `NoParseError` among them. `except* NoParseError` handles a
    `ParseErrors` and a single `NoParseError` alike.
    """

    @property
    def exceptions(self) -> tuple[NoParseError, ...]:
        """The errors, ordered by position; a `ParseErrors` does not nest."""
        return cast("tuple[NoParseError, ...]", super().exceptions)

    def derive(self, excs: Sequence[NoParseError]) -> ParseErrors:  # type: ignore[override]
        return ParseErrors(self.message, excs)

    def render(self, source: str | None = None) -> str:
        """Each error's `NoParseError.render()`, in order, separated by blank lines."""
        return "\n\n".join(e.render(source) for e in self.exceptions)


class ParsingResult[R](Iterable[Any], Protocol):
    """Result monad for parsing combinators.

    Immutable (object's data should not be changed after creation). `ParsingSuccess` and
    `ParsingError` implement it structurally.
    """

    @property
    def value(self) -> R:
        """The parsed value; raises `NoParseError` for a failure."""
        ...

    @property
    def state(self) -> State:
        """The parsing state after this result."""
        ...

    def map[C](self, f: Callable[[R], C]) -> ParsingResult[C]:
        """Apply `f` to the parsed value of a success; a failure is returned as it is."""
        ...

    def starmap[C](self, f: Callable[..., C]) -> ParsingResult[C]:
        """Assume that the value is a tuple and apply the function to its elements."""
        ...

    def bind[C](self, f: Callable[[R, State], ParsingResult[C]]) -> ParsingResult[C]:
        """Continue a success with `f`, given its value and state; a failure is returned as is."""
        ...

    def __add__[C](self, other: ParsingResult[C]) -> ParsingResult[Any]:
        """Combine two results in sequence, as `Parser.__add__()` combines their parsers."""
        ...


@final
@valueclass
class ParsingSuccess[R]:
    """A successful parsing result: the parsed `value` and the `state` after it."""

    value: R
    state: State

    def __iter__(self) -> Iterator[Any]:
        yield self.value
        yield self.state

    def map[C](self, f: Callable[[R], C]) -> ParsingResult[C]:
        # mypy checks ParsingSuccess against ParsingResult through this very method, and fails.
        return ParsingSuccess(f(self.value), self.state)  # type: ignore[return-value]

    def starmap[C](self, f: Callable[..., C]) -> ParsingResult[C]:
        return ParsingSuccess(f(*cast("tuple[Any, ...]", self.value)), self.state)

    def bind[C](self, f: Callable[[R, State], ParsingResult[C]]) -> ParsingResult[C]:
        return f(self.value, self.state)

    def __add__[C](self, other: ParsingResult[C]) -> ParsingResult[Any]:
        # Assume that the other state is further away from the current one
        # (matches Parser.__add__() behavior)
        if isinstance(other, ParsingError):
            return other
        v1, v2, s2 = self.value, other.value, other.state
        if isinstance(v1, IgnoredValue):
            return other
        if isinstance(v2, IgnoredValue):
            return ParsingSuccess(v1, s2)
        return ParsingSuccess(_Tuple.combine(v1, v2), s2)


@final
@valueclass
class ParsingError:
    """A failed parsing result: where parsing stopped, and why.

    It carries no value, so it is a `ParsingResult` of any value type; its `value` raises
    `NoParseError`. A `committed` failure (see `commit()`) is final: no alternative is tried.
    """

    state: State
    message: str = "got unexpected token"
    committed: bool = False

    def replace_state(self, s: State) -> ParsingError:
        """The same failure at the state `s`."""
        return ParsingError(s, self.message, self.committed)

    @property
    def _error(self) -> NoParseError:
        e = NoParseError(self.message, self.state)
        e._result = self
        return e

    @property
    def value(self) -> NoReturn:
        raise self._error

    def __iter__(self) -> Iterator[Any]:
        raise self._error

    def map(self, f: Callable[[Any], object]) -> ParsingError:
        return self

    def starmap(self, f: Callable[..., object]) -> ParsingError:
        return self

    def bind(self, f: Callable[[Any, State], object]) -> ParsingError:
        return self

    def __add__(self, other: object) -> ParsingError:
        return self


type ParserFn[A, B] = Callable[[Sequence[A], State], ParsingResult[B]]
"""A parsing function: the tokens and the state in, a parsing result out."""

type ParserObjOrFn[A, B] = Parser[A, B] | ParserFn[A, B]
"""A `Parser` or a bare parsing function, as `Parser()` and `define()` accept."""


class Parser[A, B]:
    """A parser object that can parse a sequence of tokens or can be combined with other parsers
    using `+`, `|`, `>>`, `many()`, and other parsing combinators.

    Type: `Parser[A, B]`

    Extensible by design: `forward_decl()` returns a subclass, and the sequence parsers are typed
    through subclasses.

    The generic variables in the type are: `A` — the type of the tokens in the sequence to parse,`B`
    — the type of the parsed value.

    In order to define a parser for your grammar:

    1. You start with primitive parsers by calling `a(value)`, `some(pred)`, `forward_decl()`,
       `finished`
    2. You use parsing combinators `p1 + p2`, `p1 | p2`, `p >> f`, `many(p)`, and others to combine
       parsers into a more complex parser
    3. You can assign complex parsers to variables to define names that correspond to the rules of
       your grammar

    !!! Note

        The constructor `Parser.__init__()` is considered **internal** and may be
        changed in future versions. Use primitive parsers and parsing combinators to
        construct new parsers.
    """

    run: ParserFn[A, B]
    """Run the parser against the tokens with the specified parsing state.

    Type: `(Sequence[A], State) -> ParsingResult[B]`

    The parsing state includes the current position in the sequence being parsed, and the position
    of the rightmost token that has been consumed while parsing for better error messages.

    It returns a `ParsingSuccess` with the parsed value and the state after it, or a `ParsingError`
    if the parser fails; it does not raise `NoParseError`.

    !!! Warning

        This method is **internal** and may be changed in future versions. Use
        `Parser.parse(tokens)` instead and let the parser object take care of
        updating the parsing state.
    """

    name: str | None = None
    _fn: ParserFn[A, B]
    _label: bool = False
    _leaf: bool = False

    def __init__(self, p: ParserObjOrFn[A, B]) -> None:
        """Wrap the parser function `p` into a `Parser` object.

        A function written for 1.0, which returns a `(value, state)` tuple or raises `NoParseError`,
        still works: its tuple is a success, its `NoParseError` a failure `|` backtracks from. The
        first such result emits a `DeprecationWarning`.
        """
        self._define(p)

    def named(self, name: str | None) -> Self:
        # noinspection GrazieInspection
        """Specify the name of the parser for easier debugging.

        Type: `(str) -> Parser[A, B]`

        The name is a label for error messages: when the parser fails without getting past its first
        token, the message says it expected `name` rather than what the parser is made of. It is
        also used in the debug-level parsing log, and is the `Parser.name` attribute. `named(None)`
        removes both the name and the label.

        Examples:
            ```pycon
            >>> expr = (a("x") + a("y")).named("expr")
            >>> expr.name
            'expr'

            ```

            ```pycon
            >>> expr = a("x") + a("y")
            >>> expr.name
            "('x', 'y')"

            ```

            !!! Note

                You can enable the parsing log this way:

                ```python
                import logging

                logging.basicConfig(level=logging.DEBUG)
                import funcparserlib.parser

                funcparserlib.parser.debug = True
                ```

                The way to enable the parsing log may be changed in future versions.
        """
        self.name = name
        self._label = name is not None
        self._update_run()
        return self

    def _rename(self, name: str | None) -> Self:
        """Set the name the library gives a parser it builds, without making it a label."""
        self.name = name
        return self

    def _named_from(self, p: ParserObjOrFn[A, B]) -> Self:
        if not self._label and (name := getattr(p, "name", p.__doc__)) is not None:
            return self._rename(name)
        return self

    def _update_run(self) -> None:
        f = self._fn
        if self._label and not self._leaf:
            f = self._labelled(f)
        self.run = self._wrap_for_debug(f) if debug else f

    def _labelled(self, f: ParserFn[A, B]) -> ParserFn[A, B]:
        def run_labelled(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = f(tokens, s)
            if isinstance(res, ParsingError) and res.state.max == s.pos:
                # Nothing got past this parser's start: it, not its parts, is what was expected.
                st = res.state
                expected = (*s.expected, (self, s.context))
                failed = State(st.pos, st.max, expected, st.context, st.problems)
                return ParsingError(failed, res.message, res.committed)
            return res

        return run_labelled

    def _define(self, p: ParserObjOrFn[A, B], *, adapt: bool = True) -> None:
        """Define the parser created earlier as a forward declaration.

        Type: `(Parser[A, B]) -> None`

        Use `p = forward_decl()` in combination with `p.define(...)` to define recursive parsers.

        See the examples in the docs for `forward_decl()`.
        """
        if not isinstance(p, Parser):
            self._fn = self._adapted(p) if adapt else p
        elif isinstance(p, ForwardDeclParser):
            fwd = p

            # Looked up on each run: `fwd` may be defined after this parser is.
            def run_declared(tokens: Sequence[A], s: State) -> ParsingResult[B]:
                return fwd.run(tokens, s)

            self._fn = run_declared
        else:
            self._fn = p.run
        self._update_run()
        self._named_from(p)

    def _adapted(self, f: ParserFn[A, B]) -> ParserFn[A, B]:
        """`f`, reading a 1.0 result as a `ParsingResult`, with a warning for the first one."""
        warned = False

        def run_adapted(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            nonlocal warned
            try:
                res: object = f(tokens, s)
            except NoParseError as e:
                # A result unpacked inside `f` is in these tokens; a nested `parse()` is not.
                res = e._result or ParsingError(s.expect(self), e.msg)
                legacy = True
            else:
                legacy = isinstance(res, tuple)
                if legacy:
                    value, state = cast("tuple[B, State]", res)
                    res = ParsingSuccess(value, state)
            if legacy and not warned:
                warned = True
                _warn_legacy(f)
            return cast("ParsingResult[B]", res)

        return run_adapted

    def _wrap_for_debug(self, f: ParserFn[A, B]) -> ParserFn[A, B]:
        def run_parser_verbose(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            _log.debug("trying %s", self.name)
            return f(tokens, s)

        return run_parser_verbose

    def parse(
        self, tokens: Sequence[A], *, describe: Callable[[A], TokenDescription] | None = None
    ) -> B:
        """Parse the sequence of tokens and return the parsed value.

        Type: `(Sequence[A]) -> B`

        It takes a sequence of tokens of arbitrary type `A` and returns the parsed value of
        arbitrary type `B`.

        If the parser fails to parse the tokens, it raises `NoParseError`. If `Parser.check()`
        recorded errors, it raises `ParseErrors` with them, and with the `NoParseError` when the
        parse failed as well.

        `describe` turns a token into the `TokenDescription` error messages show. By default it
        reads the `value`, `start` and `end` of a `funcparserlib.lexer.Token`, and shows any other
        object as it is.

        !!! Note

            Although `Parser.parse()` can parse sequences of any objects (including
            `str` which is a sequence of `str` chars), **the recommended way** is
            parsing sequences of `Token` objects.

            You **should** use a regexp-based tokenizer `make_tokenizer()` defined in
            `funcparserlib.lexer` to convert your text into a sequence of `Token`
            objects before parsing it. You will get more readable parsing error messages
            (as `Token` objects contain their position in the source file) and good
            separation of the lexical and syntactic levels of the grammar.
        """
        res = self.run(tokens, State(0, 0))
        show = describe or _describe
        errors = [_problem_error(problem, tokens, show) for problem in res.state.problems]
        if isinstance(res, ParsingError):
            fatal = _parse_error(res, tokens, show)
            if not errors:
                raise fatal
            errors.append(fatal)
        if errors:
            errors.sort(key=lambda e: e.state.pos)
            raise ParseErrors(
                f"{len(errors)} parsing error{'s' if len(errors) > 1 else ''}", errors
            )
        return res.value

    def but_not(self, other: Parser[A, Any]) -> Parser[A, B]:
        """Parse as usual, but only if the other parser fails at the same position.

        Type: `(Parser[A, B], Parser[A, Any]) -> Parser[A, B]`

        Examples:
            ```pycon
            >>> name = some(str.isalpha).named("name")
            >>> expr = name.but_not(a("if"))
            >>> expr.parse(["x"])
            'x'
            >>> expr.parse(["if"])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: 'if', expected: name (but ...)

            ```
        """

        @_parser(f"{self.name} (but not {other.name})", leaf=True)
        def _but_not(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingError):
                return res
            res_not = other.run(tokens, res.state.reset_pos_to(s))
            if isinstance(res_not, ParsingError):
                return res
            return ParsingError(s.expect(_but_not))

        return _but_not

    def when(self, pred: Callable[[B], bool]) -> Parser[A, B]:
        """Wrap the parser to a new one, that parses a token if it satisfies the predicate `pred`.

        Type: `(Parser[A, B], Callable[[B], bool]) -> Parser[A, B]`

        Examples:
            ```pycon
            >>> digit = some(str.isdigit).named("digit")
            >>> expr = digit.when(lambda d: d != "0")
            >>> expr.parse("7")
            '7'
            >>> expr.parse("0")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: '0', expected: digit

            ```
        """

        @_parser(self.name, leaf=True)
        def _when(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingError) or pred(res.value):
                return res
            return ParsingError(s.expect(_when))

        return _when

    def context(self, label: str) -> Parser[A, B]:
        """Name what this parser parses, for the errors inside it: `... in label`.

        Type: `(Parser[A, B], str) -> Parser[A, B]`

        An error inside the parser ends with `in label`; nested contexts are listed innermost first,
        and `NoParseError.context` holds them.

        Examples:
            ```pycon
            >>> args = (a("(") + many(a("x")) + a(")")).context("args")
            >>> call = (a("f") + args).context("call")
            >>> try:
            ...     call.parse("f(xy)")
            ... except NoParseError as e:
            ...     print(e)
            ...     print(e.expected, e.context)
            got unexpected token: 'y', expected: 'x' or ')' in args in call
            ("'x'", "')'") ('args', 'call')

            ```
        """

        @_parser(self.name)
        def _context(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            inner = State(s.pos, s.max, s.expected, (*s.context, label), s.problems)
            res = self.run(tokens, inner)
            st = res.state
            out = State(st.pos, st.max, st.expected, s.context, st.problems)
            if isinstance(res, ParsingError):
                return ParsingError(out, res.message, res.committed)
            return ParsingSuccess(res.value, out)

        return _context

    def check(self, pred: Callable[[B], bool], message: str) -> Parser[A, B]:
        """Record an error, `message`, when `pred` is false for the parsed value, and go on.

        Type: `(Parser[A, B], Callable[[B], bool], str) -> Parser[A, B]`

        Unlike `when()`, a failed check does not fail the parser: parsing goes on with the value,
        and `Parser.parse()` raises `ParseErrors` with every recorded error at the end. An error
        covers the tokens this parser parsed. A check on a branch that `|` backtracks from is
        forgotten with the branch.

        Examples:
            ```pycon
            >>> digit = some(str.isdigit) >> int
            >>> byte = (digit + digit) >> (lambda d: d[0] * 10 + d[1])
            >>> small = byte.check(lambda n: n < 50, "number too big")
            >>> many(small + -a(",")).parse("12,99,77,")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.ParseErrors: 2 parsing errors (2 sub-exceptions)

            ```
        """

        @_parser(self.name)
        def _check(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingError) or pred(res.value):
                return res
            st = res.state
            problem = CheckFailure(message, s.pos, st.pos, s.context)
            checked = State(st.pos, st.max, st.expected, st.context, (*st.problems, problem))
            return ParsingSuccess(res.value, checked)

        return _check

    @overload
    def __add__(self, other: IgnoredEndOfInput) -> Self: ...  # type: ignore[overload-overlap]

    @overload
    def __add__(  # type: ignore[overload-overlap]
        self, other: IgnoredParser[A]
    ) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> Tuple2Parser[A, B, C]: ...

    def __add__(self, other: Parser[A, Any]) -> Parser[A, Any]:
        """Sequential combination of parsers. It runs this parser, then the other parser.

        The return value of the resulting parser is a tuple of each parsed value in the sum of
        parsers. We merge all parsing results of `p1 + p2 + ... + pN` into a single tuple. It means
        that the parsing result may be a 2-tuple, a 3-tuple, a 4-tuple, etc. of parsed values. You
        avoid this by transforming the parsed pair into a new value using the `>>` combinator.

        You can also skip some parsing results in the resulting parsers by using `-p` or `skip(p)`
        for some parsers in your sum of parsers. It means that the parsing result might be a single
        value, not a tuple of parsed values. See the docs for `Parser.__neg__()` for more examples.

        Overloaded types (lots of them to provide stricter checking for the quite dynamic return
        type of this method):

        * `(self: Parser[A, B], IgnoredEndOfInput) -> Self`, for `p + -finished`
        * `(self: Parser[A, B], IgnoredParser[A]) -> Self`
        * `(self: IgnoredParser[A], Parser[A, C]) -> Parser[A, C]`
        * `(self: Parser[A, B], Parser[A, C]) -> Tuple2Parser[A, B, C]`
        * `(self: Tuple2Parser[A, B, C], Parser[A, D]) -> Tuple3Parser[A, B, C, D]`
        * `(self: Tuple3Parser[A, B, C, D], Parser[A, E]) -> Tuple4Parser[A, B, C, D, E]`
        * `(self: Tuple4Parser[A, B, C, D, E], Parser[A, F]) -> Tuple5Parser[A, B, C, D, E, F]`
        * `(self: Tuple5Parser[A, B, C, D, E, F], Parser[A, Any]) -> Parser[A, Any]`

        Examples:
            ```pycon
            >>> expr = a("x") + a("y")
            >>> expr.parse("xy")
            ('x', 'y')

            ```

            ```pycon
            >>> expr = a("x") + a("y") + a("z")
            >>> expr.parse("xyz")
            ('x', 'y', 'z')

            ```

            ```pycon
            >>> expr = a("x") + a("y")
            >>> expr.parse("xz")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: 'z', expected: 'y'

            ```
        """

        @_parser(f"({self.name}, {other.name})")
        def _add(tokens: Sequence[A], s: State) -> ParsingResult[Any]:
            res_l = self.run(tokens, s)
            if isinstance(res_l, ParsingError):
                return res_l
            return res_l + other.run(tokens, res_l.state)

        return _add

    def __or__[C](self, other: Parser[A, C]) -> Parser[A, B | C]:
        """Choice combination of parsers.

        It runs this parser and returns its result. If the parser fails, it runs the other parser.

        Examples:
            ```pycon
            >>> expr = a("x") | a("y")
            >>> expr.parse("x")
            'x'
            >>> expr.parse("y")
            'y'
            >>> expr.parse("z")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: 'z', expected: 'x' or 'y'

            ```
        """

        @_parser(f"{self.name} or {other.name}")
        def _or(tokens: Sequence[A], s: State) -> ParsingResult[B | C]:
            res = self.run(tokens, s)
            if not isinstance(res, ParsingError) or res.committed:
                return res
            return other.run(tokens, res.state.reset_pos_to(s))

        return _or

    def map[C](self, f: Callable[[B], C]) -> Parser[A, C]:
        """Transform the parsing result by applying the specified function; the same as `>>`."""

        @_parser(self.name)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.map(f)

        return _map

    def starmap[C](self, f: Callable[..., C]) -> Parser[A, C]:
        """Transform the parsing results (tuple) by applying the specified function."""

        @_parser(self.name)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.starmap(f)

        return _map

    def __rshift__[C](self, f: Callable[[B], C]) -> Parser[A, C]:
        """Transform the parsing result by applying the specified function.

        Type: `(Callable[[B], C]) -> Parser[A, C]`

        You can use it for transforming the parsed value into another value before including it into
        the parse tree (the AST).

        Examples:
            ```pycon
            >>> def make_canonical_name(s):
            ...     return s.lower()
            >>> expr = (a("D") | a("d")) >> make_canonical_name
            >>> expr.parse("D")
            'd'
            >>> expr.parse("d")
            'd'

            ```
        """
        return self.map(f)

    def bind[C](self, f: Callable[[B], Parser[A, C]]) -> Parser[A, C]:
        """Bind the parser to a monadic function that returns a new parser.

        Type: `(Callable[[B], Parser[A, C]]) -> Parser[A, C]`

        Also known as `>>=` in Haskell.

        !!! Note

            You can parse any context-free grammar without resorting to `bind`. Due
            to its poor performance please use it only when you really need it.
        """

        @_parser(f"({self.name} >>=)")
        def _bind(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.bind(lambda v, s2: f(v).run(tokens, s2))

        return _bind

    def __neg__(self) -> IgnoredParser[A]:
        """Return a parser that parses the same tokens, but its parsing result is ignored by the
        sequential `+` combinator.

        Type: `(Parser[A, B]) -> IgnoredParser[A]`

        You can use it for throwing away elements of concrete syntax (e.g. `","`, `";"`).

        Examples:
            ```pycon
            >>> expr = -a("x") + a("y")
            >>> expr.parse("xy")
            'y'

            ```

            ```pycon
            >>> expr = a("x") + -a("y")
            >>> expr.parse("xy")
            'x'

            ```

            ```pycon
            >>> expr = a("x") + -a("y") + a("z")
            >>> expr.parse("xyz")
            ('x', 'z')

            ```

            ```pycon
            >>> expr = -a("x") + a("y") + -a("z")
            >>> expr.parse("xyz")
            'y'

            ```

            ```pycon
            >>> expr = -a("x") + a("y")
            >>> expr.parse("yz")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: 'y', expected: 'x'

            ```

            ```pycon
            >>> expr = a("x") + -a("y")
            >>> expr.parse("xz")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: 'z', expected: 'y'

            ```

            !!! Note

                You **should not** pass the resulting parser to any combinators other than
                `+`. You **should** have at least one non-skipped value in your
                `p1 + p2 + ... + pN`. The parsed value of `-p` is an `IgnoredValue`
                object, not intended for actual use.
        """

        @_parser(self.name)
        def _ignored(tokens: Sequence[A], s: State) -> ParsingResult[IgnoredValue]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingError):
                return res
            return ParsingSuccess(_IGNORED, res.state)

        return cast("IgnoredParser[A]", _ignored)

    def __invert__(self) -> Parser[A, A]:
        """Return a parser that parses one token, any token this parser does not parse.

        Type: `(Parser[A, B]) -> Parser[A, A]`

        Examples:
            ```pycon
            >>> expr = ~a("x")
            >>> expr.parse("y")
            'y'
            >>> expr.parse("x")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: 'x', expected: anything but 'x'
            >>> expr.parse("")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected end of input, expected: anything ...

            ```
        """

        @_parser(f"anything but {self.name}", leaf=True)
        def _invert(tokens: Sequence[A], s: State) -> ParsingResult[A]:
            if s.pos >= len(tokens):
                s2 = s.expect(_invert)
                return ParsingError(s2, "got unexpected end of input")
            res = self.run(tokens, s)
            if isinstance(res, ParsingError):
                return ParsingSuccess(tokens[s.pos], s.reset_pos(s.pos + 1))
            return ParsingError(s.expect(_invert))

        return _invert


def parser[A, B](name: str | None) -> Callable[[ParserFn[A, B]], Parser[A, B]]:
    """Make a named `Parser` from a parsing function, as a decorator.

    Type: `(str | None) -> Callable[[ParserFn[A, B]], Parser[A, B]]`

    The decorated function takes the tokens and the current `State` and returns a `ParsingSuccess`
    or a `ParsingError`; unlike `Parser(f)`, it does not adapt a 1.0 parsing function. A success
    carries the state moved past what it parsed, `s.reset_pos(pos)`; a failure carries
    `s.expect(p)`, where `p` is the parser that failed, so that the error names it.

    Examples:
        ```pycon
        >>> @parser("digit")
        ... def digit(tokens, s):
        ...     if s.pos < len(tokens) and tokens[s.pos].isdigit():
        ...         return ParsingSuccess(int(tokens[s.pos]), s.reset_pos(s.pos + 1))
        ...     return ParsingError(s.expect(digit))
        >>> many(digit).parse("12")
        [1, 2]
        >>> (digit + digit).parse("1x")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: 'x', expected: digit

        ```
    """

    def decorator(f: ParserFn[A, B]) -> Parser[A, B]:
        return _built(Parser, f).named(name)

    return decorator


def _parser[A, B](
    name: str | None, *, leaf: bool = False
) -> Callable[[ParserFn[A, B]], Parser[A, B]]:
    """`parser()` for the parsers the library builds: a name without a label.

    A `leaf` parser names itself among the expected ones when it fails, so a label given to it later
    changes only its name.
    """

    def decorator(f: ParserFn[A, B]) -> Parser[A, B]:
        p: Parser[A, B] = _built(Parser, f)._rename(name)
        p._leaf = leaf
        return p

    return decorator


def _built[P: Parser[Any, Any]](cls: type[P], f: ParserFn[Any, Any]) -> P:
    """A parser of `cls` running `f` as it is, without the 1.0 adapter."""
    p = cls.__new__(cls)
    p._define(f, adapt=False)
    return p


def _warn_legacy(f: Callable[..., object]) -> None:
    """Warn that `f` returned a `(value, state)` tuple or raised `NoParseError`, as in 1.0."""
    name = getattr(f, "__qualname__", repr(f))
    msg = (
        f"parsing function {name} follows the 1.0 protocol, deprecated in 2.0:"
        " return a ParsingSuccess or a ParsingError"
    )
    code = getattr(f, "__code__", None)
    if code is None:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    else:
        module = getattr(f, "__module__", None)
        warnings.warn_explicit(
            msg, DeprecationWarning, code.co_filename, code.co_firstlineno, module=module
        )


def _parse_error[A](
    res: ParsingError, tokens: Sequence[A], describe: Callable[[A], TokenDescription]
) -> NoParseError:
    """The `NoParseError` of a failed parse: what was found at the furthest position, and why."""
    st = res.state
    names = tuple(dict.fromkeys(p.name for p, _ in st.expected if p.name is not None))
    contexts = [c for _, c in st.expected] or [st.context]
    common = contexts[0][: min(map(len, contexts))]
    while any(c[: len(common)] != common for c in contexts):
        common = common[:-1]
    context = tuple(reversed(common))
    if st.max < len(tokens):
        token: A | None = tokens[st.max]
        shown = describe(tokens[st.max])
        description: TokenDescription | None = shown
        found = f"{_where(shown)}{res.message}: {shown.text}"
    else:
        token = None
        description = _after(tokens, describe)
        found = "got unexpected end of input"
    msg = found + (f", expected: {' or '.join(names)}" if names else "")
    msg += "".join(f" in {c}" for c in context)
    return NoParseError(
        msg, st, token=token, expected=names, context=context, description=description
    )


def _problem_error[A](
    problem: CheckFailure, tokens: Sequence[A], describe: Callable[[A], TokenDescription]
) -> NoParseError:
    """The `NoParseError` of a failed `Parser.check()`, spanning the tokens it checked."""
    if problem.start < len(tokens):
        token: A | None = tokens[problem.start]
        first = describe(tokens[problem.start])
        last = describe(tokens[max(problem.start, problem.end - 1)])
        description: TokenDescription | None = TokenDescription(first.text, first.start, last.end)
    else:
        token, description = None, _after(tokens, describe)
    context = tuple(reversed(problem.context))
    msg = f"{_where(description)}{problem.message}" + "".join(f" in {c}" for c in context)
    state = State(problem.start, problem.start)
    return NoParseError(msg, state, token=token, context=context, description=description)


def _where(d: TokenDescription | None) -> str:
    """The `line,column-line,column: ` prefix of a message, empty for a token without position."""
    if d is None or d.start is None or d.end is None:
        return ""
    return f"{d.start[0]},{d.start[1]}-{d.end[0]},{d.end[1]}: "


def _after[A](
    tokens: Sequence[A], describe: Callable[[A], TokenDescription]
) -> TokenDescription | None:
    """The end of input: the column after the last token, when its position is known."""
    if not tokens:
        return None
    end = describe(tokens[-1]).end
    if end is None:
        return None
    return TokenDescription("end of input", (end[0], end[1] + 1), (end[0], end[1] + 1))


class _Tuple(tuple[Any, ...]):
    """Parsed values from multiple combined parsers."""

    __slots__ = ()

    @classmethod
    def combine(cls, v1: Any, v2: Any) -> Self:
        if isinstance(v1, cls):
            return cls(v1 + (v2,))
        else:
            return cls((v1, v2))


@final
class Tuple2Parser[A, T1, T2](Parser[A, tuple[T1, T2]]):
    """The type of a sequence of 2 parsers, `p1 + ... + p2`; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(self, other: IgnoredEndOfInput) -> Self: ...  # type: ignore[overload-overlap]

    @overload
    def __add__(  # type: ignore[overload-overlap]
        self, other: IgnoredParser[A]
    ) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> Tuple3Parser[A, T1, T2, C]: ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2], C]) -> Parser[A, C]:
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2]], C]) -> Parser[A, C]:
        raise NotImplementedError


@final
class Tuple3Parser[A, T1, T2, T3](Parser[A, tuple[T1, T2, T3]]):
    """The type of a sequence of 3 parsers, `p1 + ... + p3`; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(self, other: IgnoredEndOfInput) -> Self: ...  # type: ignore[overload-overlap]

    @overload
    def __add__(  # type: ignore[overload-overlap]
        self, other: IgnoredParser[A]
    ) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> Tuple4Parser[A, T1, T2, T3, C]: ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3], C]) -> Parser[A, C]:
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3]], C]) -> Parser[A, C]:
        raise NotImplementedError


@final
class Tuple4Parser[A, T1, T2, T3, T4](Parser[A, tuple[T1, T2, T3, T4]]):
    """The type of a sequence of 4 parsers, `p1 + ... + p4`; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(self, other: IgnoredEndOfInput) -> Self: ...  # type: ignore[overload-overlap]

    @overload
    def __add__(  # type: ignore[overload-overlap]
        self, other: IgnoredParser[A]
    ) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> Tuple5Parser[A, T1, T2, T3, T4, C]: ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3, T4], C]) -> Parser[A, C]:
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3, T4]], C]) -> Parser[A, C]:
        raise NotImplementedError


@final
class Tuple5Parser[A, T1, T2, T3, T4, T5](Parser[A, tuple[T1, T2, T3, T4, T5]]):
    """The type of a sequence of 5 parsers, `p1 + ... + p5`; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(self, other: IgnoredEndOfInput) -> Self: ...

    @overload
    def __add__(self, other: IgnoredParser[A]) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> Parser[A, Any]: ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3, T4, T5], C]) -> Parser[A, C]:
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3, T4, T5]], C]) -> Parser[A, C]:
        raise NotImplementedError


@final
@valueclass
class IgnoredValue:
    """Constant to indicate an ignored parsed value."""


_IGNORED = IgnoredValue()


def many[A, B](p: Parser[A, B]) -> Parser[A, list[B]]:
    """Return a parser that applies the parser `p` as many times as it succeeds at parsing the
    tokens.

    Return a parser that infinitely applies the parser `p` to the input sequence of tokens as long
    as it successfully parses them. The parsed value is a list of the sequentially parsed values.

    Examples:
        ```pycon
        >>> expr = many(a("x"))
        >>> expr.parse("x")
        ['x']
        >>> expr.parse("xx")
        ['x', 'x']
        >>> expr.parse("xxxy")  # noqa
        ['x', 'x', 'x']
        >>> expr.parse("y")
        []

        ```
    """

    @_parser(f"{{ {p.name} }}")
    def _many(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        acc = []
        while True:
            res = p.run(tokens, s)
            if not isinstance(res, ParsingError):
                if res.state.pos == s.pos:
                    raise with_hint(
                        ValueError(f"{p.name} matched no tokens inside many()"),
                        "many() repeats a parser until it fails, and one that matches nothing never"
                        " fails: repeat the parser that maybe(), many() or pure() wraps.",
                    )
                acc.append(res.value)
                s = res.state
            elif res.committed:
                return res
            else:
                s = s.replace_rest_from(res.state)
                break

        if debug:
            _log.debug("*matched* %d instances of %s, new state = %s", len(acc), _many.name, s)
        return ParsingSuccess(acc, s)

    return _many


def some[A](pred: Callable[[A], bool]) -> Parser[A, A]:
    """Return a parser that parses a token if it satisfies the predicate `pred`.

    Type: `(Callable[[A], bool]) -> Parser[A, A]`

    Examples:
        ```pycon
        >>> expr = some(lambda s: s.isalpha()).named('alpha')
        >>> expr.parse("x")
        'x'
        >>> expr.parse("y")
        'y'
        >>> expr.parse("1")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: '1', expected: alpha

        ```

        !!! Warning

            The `some()` combinator is quite slow and may be changed or removed in future
            versions. If you need a parser for a token by its type (e.g. any identifier)
            and maybe its value, use `tok(type[, value])` instead. You should use
            `make_tokenizer()` from `funcparserlib.lexer` to tokenize your text first.
    """

    @_parser("some(...)", leaf=True)
    def _some(tokens: Sequence[A], s: State) -> ParsingResult[A]:
        if s.pos >= len(tokens):
            s2 = s.expect(_some)
            return ParsingError(s2, "got unexpected end of input")

        t = tokens[s.pos]
        pos = s.pos + 1
        if pred(t):
            s2 = s.reset_pos(pos)
            if debug:
                _log.debug("*matched* %r, new state = %s", t, s2)
            return ParsingSuccess(t, s2)

        s2 = s.expect(_some)
        if debug:
            _log.debug("failed %r, state = %s, expected: %s", t, s, _some.name)
        return ParsingError(s2)

    return _some


def a[A](value: A) -> Parser[A, A]:
    """Return a parser that parses a token if it's equal to `value`.

    Type: `(A) -> Parser[A, A]`

    Examples:
        ```pycon
        >>> expr = a("x")
        >>> expr.parse("x")
        'x'
        >>> expr.parse("y")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: 'y', expected: 'x'

        ```

        !!! Note

            Although `Parser.parse()` can parse sequences of any objects (including
            `str` which is a sequence of `str` chars), **the recommended way** is
            parsing sequences of `Token` objects.

            You **should** use a regexp-based tokenizer `make_tokenizer()` defined in
            `funcparserlib.lexer` to convert your text into a sequence of `Token` objects
            before parsing it. You will get more readable parsing error messages (as `Token`
            objects contain their position in the source file) and good separation of the
            lexical and syntactic levels of the grammar.
    """
    name = getattr(value, "name", value)  # TODO Explain what is going on here

    def eq_value(t: A) -> bool:
        return t == value

    return some(eq_value).named(repr(name))


# noinspection PyShadowingBuiltins
def tok(type: str, value: str | None = None) -> Parser[Token, str]:  # noqa: A002 — public keyword
    """Return a parser that parses a `Token` and returns the string value of the token.

    Type: `(str, Optional[str]) -> Parser[Token, str]`

    You can match any token of the specified `type` or you can match a specific token by its `type`
    and `value`.

    Examples:
        ```pycon
        >>> expr = tok("expr")
        >>> expr.parse([Token("expr", "foo")])
        'foo'
        >>> expr.parse([Token("expr", "bar")])
        'bar'
        >>> expr.parse([Token("op", "=")])
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: '=', expected: expr

        ```

        ```pycon
        >>> expr = tok("op", "=")
        >>> expr.parse([Token("op", "=")])
        '='
        >>> expr.parse([Token("op", "+")])
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: '+', expected: '='

        ```

        !!! Note

            In order to convert your text to parse into a sequence of `Token` objects,
            use a regexp-based tokenizer `make_tokenizer()` defined in
            `funcparserlib.lexer`. You will get more readable parsing error messages (as
            `Token` objects contain their position in the source file) and good separation
            of the lexical and syntactic levels of the grammar.
    """

    def eq_type(t: Token) -> bool:
        return t.type == type

    p = a(Token(type, value)) if value is not None else some(eq_type).named(type)
    return (p >> (lambda t: t.value))._rename(p.name)


def pure[A](x: A) -> Parser[Any, A]:
    """Wrap any object into a parser.

    Type: `(B) -> Parser[Any, B]`

    A pure parser doesn't touch the tokens sequence, it just returns its pure `x` value.

    Also known as `return` in Haskell.
    """

    @_parser(f"(pure {x!r})")
    def _pure(_: Sequence[Any], s: State) -> ParsingResult[A]:
        return ParsingSuccess(x, s)

    return _pure


def commit[A, B](p: Parser[A, B]) -> Parser[A, B]:
    """Return a parser that parses as `p`, and makes a failure of `p` final once it read a token.

    Type: `(Parser[A, B]) -> Parser[A, B]`

    Without it, `p1 | p2` tries `p2` whenever `p1` fails, and `many(p)` stops at the first failure,
    however far into the tokens `p` got; the error then names where the backtracking ended. A
    committed failure is not backtracked from: no alternative is tried, `many` and `maybe` fail with
    it, and the error names what `p` expected where it stopped. A failure before `p` read any token
    is not committed.

    Examples:
        ```pycon
        >>> call = a("f") + commit(a("(") + a("x") + a(")"))
        >>> word = many(a("f"))
        >>> (call | word).parse("f(y)")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: 'y', expected: 'x'

        ```
    """

    @_parser(p.name)
    def _commit(tokens: Sequence[A], s: State) -> ParsingResult[B]:
        # A fresh furthest position tells whether `p` read a token.
        res = p.run(tokens, State(s.pos, s.pos, (), s.context, s.problems))
        st = res.state
        if st.max > s.max:
            max_, expected = st.max, st.expected
        elif st.max == s.max:
            max_, expected = s.max, (*s.expected, *st.expected)
        else:
            max_, expected = s.max, s.expected
        merged = State(st.pos, max_, expected, st.context, st.problems)
        if isinstance(res, ParsingError):
            return ParsingError(merged, res.message, res.committed or st.max > s.pos)
        return ParsingSuccess(res.value, merged)

    return _commit


def maybe[A, B](p: Parser[A, B]) -> Parser[A, B | None]:
    """Return a parser that returns `None` if the parser `p` fails.

    Examples:
        ```pycon
        >>> expr = maybe(a("x"))
        >>> expr.parse("x")
        'x'
        >>> expr.parse("y") is None
        True

        ```
    """
    return (p | pure(None))._rename(f"[ {p.name} ]")


def skip[A](p: Parser[A, Any]) -> IgnoredParser[A]:
    """An alias for `-p`.

    See also docs for `Parser.__neg__()`.
    """
    return -p


def anything_but[A](p: Parser[A, Any]) -> Parser[A, A]:
    """An alias for `~p`.

    See also docs for `Parser.__invert__()`.
    """
    return ~p


@final
class IgnoredParser[A](Parser[A, IgnoredValue]):
    """The type of `-p` and `skip(p)`, whose value `+` drops; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(self, other: IgnoredEndOfInput) -> Self: ...

    @overload
    def __add__(self, other: IgnoredParser[A]) -> Self: ...

    @overload
    def __add__[C](self, other: ForwardDeclParser[A, C]) -> Parser[A, C]: ...

    @overload
    def __add__(self, other: EndOfInput) -> Parser[A, None]: ...

    @overload
    def __add__[P: Parser[Any, Any]](self, other: P) -> P: ...

    def __add__(self, other):
        raise NotImplementedError


def oneplus[A, B](p: Parser[A, B]) -> Parser[A, list[B]]:
    """Return a parser that applies the parser `p` one or more times.

    A similar parser combinator `many(p)` means apply `p` zero or more times, whereas `oneplus(p)`
    means apply `p` one or more times.

    Examples:
        ```pycon
        >>> expr = oneplus(a("x"))
        >>> expr.parse("x")
        ['x']
        >>> expr.parse("xx")
        ['x', 'x']
        >>> expr.parse("y")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: 'y', expected: 'x'

        ```
    """
    many_p = many(p)

    @_parser(f"({p.name}, {{ {p.name} }})")
    def _oneplus(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        first = p.run(tokens, s)
        if isinstance(first, ParsingError):
            return first
        rest = many_p.run(tokens, first.state)
        if isinstance(rest, ParsingError):
            return rest
        return ParsingSuccess([first.value, *rest.value], rest.state)

    return _oneplus


@final
class IgnoredEndOfInput(Parser[Any, IgnoredValue]):
    """The type of `-finished`; a type for checkers, never built.

    `p + -finished` has the type of `p`.
    """

    def __add__[P: Parser[Any, Any]](self, other: P) -> P:  # type: ignore[override]
        raise NotImplementedError


@final
class EndOfInput(Parser[Any, None]):
    """The type of `finished`: a parser of any tokens that succeeds only at the end of input."""

    # A narrower `-` than Parser's: what makes `p + -finished` typed.
    def __neg__(self) -> IgnoredEndOfInput:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return cast("IgnoredEndOfInput", super().__neg__())

    def named(self, name: str | None) -> EndOfInput:
        """A new end-of-input parser named `name`, as `Parser.named()` names one.

        Unlike other parsers, this one is not renamed: `finished.named("EOF")` leaves `finished` as
        it is in every other grammar.
        """
        p = _end_of_input()
        Parser.named(p, name)
        return p


def _end_of_input() -> EndOfInput:
    """A parser that succeeds only at the end of input, naming itself when it fails."""

    def run(tokens: Sequence[Any], s: State) -> ParsingResult[None]:
        if s.pos >= len(tokens):
            return ParsingSuccess(None, s)
        return ParsingError(s.expect(p))

    p = _built(EndOfInput, run)._rename("end of input")
    p._leaf = True
    return p


finished: EndOfInput = _end_of_input()
"""A parser that throws an exception if there are any unparsed tokens left in the sequence."""


@final
class ForwardDeclParser[A, B](Parser[A, B]):
    """A parser defined after it is used, for recursive grammars; see `forward_decl()`."""

    def __init__(self) -> None:
        def f(_tokens: Any, _s: Any) -> Any:
            raise NotImplementedError("you must define() a forward_decl() somewhere")

        super().__init__(f)
        self._rename("forward_decl()")

    def define(self, p: ParserObjOrFn[A, B]) -> None:
        """Define the parser this forward declaration stands for."""
        self._define(p)

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self, other: IgnoredEndOfInput | IgnoredParser[A]
    ) -> Parser[A, B]: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> Tuple2Parser[A, B, C]: ...

    def __add__(self, other: Parser[A, Any]) -> Parser[A, Any]:  # ty: ignore[invalid-method-override] -- a sum is no forward declaration
        return super().__add__(other)


def forward_decl() -> ForwardDeclParser[Any, Any]:
    """Return an undefined parser that can be used as a forward declaration.

    Type: `Parser[Any, Any]`

    Use `p = forward_decl()` in combination with `p.define(...)` to define recursive parsers.

    Examples:
        ```pycon
        >>> expr = forward_decl()
        >>> expr.define(a("x") + maybe(expr) + a("y"))
        >>> expr.parse("xxyy")  # noqa
        ('x', ('x', None, 'y'), 'y')
        >>> expr.parse("xxy")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected end of input, expected: 'y'

        ```

        !!! Note

            If you care about static types, you should add a type hint for your forward
            declaration, so that your type checker can check types in `p.define(...)` later:

            ```python
            p: ForwardDeclParser[str, int] = forward_decl()
            p.define(a("x"))  # Type checker error
            p.define(a("1") >> int)  # OK
            ```
    """
    return ForwardDeclParser()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
