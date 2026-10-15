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
    "Tuple2Parser",
    "Tuple3Parser",
    "Tuple4Parser",
    "Tuple5Parser",
    "ParserFn",
    "ParserObjOrFn",
    "parser",
    "debug",
]

import dataclasses as dc
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Protocol, Self, cast, final, overload

from ._commons import valueclass, with_hint
from .lexer import Place, Token

_log = logging.getLogger("funcparserlib")

debug = False


class _TokenLike(Protocol):
    value: Any
    start: Place
    end: Place


@final
@valueclass(repr=False)
class State:
    """Parsing state that is maintained basically for error reporting.

    It consists of the current position `pos` in the sequence being parsed, and the position `max`
    of the rightmost token that has been consumed while parsing.
    """

    pos: int
    max: int
    parser: "ParserObjOrFn[Any, Any] | None" = None

    def reset_pos_to(self, s: "State") -> "State":
        """This state with the position of `s`."""
        # dataclasses.replace() hits performance, as it performs additional validation
        return State(s.pos, self.max, self.parser)

    def reset_pos(self, pos: int) -> "State":
        """This state at `pos`, with `max` moved up to `pos` if it is further."""
        return State(pos, max(pos, self.max), self.parser)

    def replace_rest_from(self, s: "State") -> "State":
        """This position, with the rightmost position and the parser of `s`."""
        return State(self.pos, s.max, s.parser)

    def replace_rest(self, max_pos: int, p: "ParserObjOrFn[Any, Any]") -> "State":
        """This position, with the rightmost position `max_pos` reached by the parser `p`."""
        return State(self.pos, max_pos, p)

    def replace_parser_if_not_observed_further(self, p: "ParserObjOrFn[Any, Any]") -> "State":
        """This state with the parser `p`, unless a parser got further than this position."""
        if self.pos == self.max:
            return State(self.pos, self.max, p)
        return self

    def __str__(self) -> str:
        return str((self.pos, self.max))

    def __repr__(self) -> str:
        return f"State({self.pos!r}, {self.max!r})"


class NoParseError(Exception):
    """Raised by `Parser.parse()` when the tokens do not match the grammar.

    `msg` names the unexpected token, its position when the token has one, and the parser that
    expected something else; `state` is where parsing stopped.
    """

    def __init__(self, msg: str, state: State) -> None:
        self.msg: str = msg
        self.state: State = state

    def __str__(self) -> str:
        return self.msg


class ParsingResult[R](Iterable[Any], Protocol):
    """Result monad for parsing combinators.

    Immutable (object's data should not be changed after creation).
    """

    __slots__ = ()

    @property
    def state(self) -> State:
        """The parsing state after this result."""
        ...

    def map[C](self, f: Callable[[R], C]) -> "ParsingResult[C]":
        """Apply `f` to the parsed value of a success; a failure is returned as it is."""
        ...

    def starmap[C](self, f: Callable[..., C]) -> "ParsingResult[C]":
        """Assume that the value is a tuple and apply the function to its elements."""
        ...

    def bind[C](self, f: Callable[[R, State], "ParsingResult[C]"]) -> "ParsingResult[C]":
        """Continue a success with `f`, given its value and state; a failure is returned as is."""
        ...

    def __add__[C](self, other: "ParsingResult[C]") -> "ParsingResult[Any]":
        """Combine two results in sequence, as `Parser.__add__()` combines their parsers."""
        ...


@final
@valueclass
class ParsingSuccess[R](ParsingResult[R]):
    """A successful parsing result: the parsed `value` and the `state` after it."""

    value: R
    state: State = dc.field()

    def __iter__(self) -> Iterator[Any]:
        yield self.value
        yield self.state

    def map[C](self, f: Callable[[R], C]) -> ParsingResult[C]:
        return ParsingSuccess(f(self.value), self.state)

    def starmap[C](self, f: Callable[..., C]) -> ParsingResult[C]:
        values = cast("tuple[Any, ...]", self.value)
        return ParsingSuccess(f(*values), self.state)

    def bind[C](self, f: Callable[[R, State], ParsingResult[C]]) -> ParsingResult[C]:
        return f(self.value, self.state)

    def __add__[C](self, other: ParsingResult[C]) -> ParsingResult[Any]:
        # Assume that the other state is further away from the current one
        # (matches Parser.__add__() behavior)
        if isinstance(other, ParsingSuccess):
            v1, v2, s2 = self.value, other.value, other.state
            if isinstance(v1, IgnoredValue):
                return other
            if isinstance(v2, IgnoredValue):
                return ParsingSuccess(v1, s2)
            return ParsingSuccess(_Tuple.combine(v1, v2), s2)
        return cast("ParsingError[Any]", other)


@final
@valueclass
class ParsingError[R](ParsingResult[R]):
    """A failed parsing result: where parsing stopped, and why.

    Its `value` raises `NoParseError`.
    """

    state: State = dc.field()
    message: str = "got unexpected token"

    def replace_state(self, s: State) -> "ParsingError[R]":
        """The same failure at the state `s`."""
        return ParsingError(s, self.message)

    @property
    def _error(self) -> NoParseError:
        return NoParseError(self.message, self.state)

    @property
    def value(self) -> R:
        raise self._error

    def __iter__(self) -> Iterator[Any]:
        raise self._error

    def map[C](self, f: Callable[[R], C]) -> ParsingResult[C]:
        return cast("ParsingError[C]", self)

    def starmap[C](self, f: Callable[..., C]) -> ParsingResult[C]:
        return cast("ParsingError[C]", self)

    def bind[C](self, f: Callable[[R, State], ParsingResult[C]]) -> ParsingResult[C]:
        return cast("ParsingError[C]", self)

    def __add__[C](self, other: ParsingResult[C]) -> ParsingResult[Any]:
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

    Type: `(Sequence[A], State) -> tuple[B, State]`

    The parsing state includes the current position in the sequence being parsed, and the position
    of the rightmost token that has been consumed while parsing for better error messages.

    If the parser fails to parse the tokens, it raises `NoParseError`.

    !!! Warning

        This method is **internal** and may be changed in future versions. Use
        `Parser.parse(tokens)` instead and let the parser object take care of
        updating the parsing state.
    """

    name: str | None = None

    def __init__(self, p: ParserObjOrFn[A, B]) -> None:
        """Wrap the parser function `p` into a `Parser` object."""
        self._define(p)

    def named(self, name: str | None) -> Self:
        # noinspection GrazieInspection
        """Specify the name of the parser for easier debugging.

        Type: `(str) -> Parser[A, B]`

        This name is used in the debug-level parsing log. You can also get it via the `Parser.name`
        attribute.

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
        return self

    def _named_from(self, p: ParserObjOrFn[A, B]) -> Self:
        if (name := getattr(p, "name", p.__doc__)) is not None:
            return self.named(name)
        return self

    def _define(self, p: ParserObjOrFn[A, B]) -> None:
        """Define the parser created earlier as a forward declaration.

        Type: `(Parser[A, B]) -> None`

        Use `p = forward_decl()` in combination with `p.define(...)` to define recursive parsers.

        See the examples in the docs for `forward_decl()`.
        """
        f = p.run if isinstance(p, Parser) else p
        self.run = self._wrap_for_debug(f) if debug else f
        self._named_from(p)

    def _wrap_for_debug(self, f: ParserFn[A, B]) -> ParserFn[A, B]:
        def run_parser_verbose(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            _log.debug("trying %s", self.name)
            return f(tokens, s)

        return run_parser_verbose

    def parse(self, tokens: Sequence[A]) -> B:
        """Parse the sequence of tokens and return the parsed value.

        Type: `(Sequence[A]) -> B`

        It takes a sequence of tokens of arbitrary type `A` and returns the parsed value of
        arbitrary type `B`.

        If the parser fails to parse the tokens, it raises `NoParseError`.

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
        try:
            (tree, _) = self.run(tokens, State(0, 0))
        except NoParseError as e:
            _format_parsing_error(e, tokens)
            raise
        return cast("B", tree)

    def but_not(self, other: "Parser[A, Any]") -> "Parser[A, B]":
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

        @parser(f"{self.name} (but not {other.name})")
        def _but_not(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingError):
                return res
            res_not = other.run(tokens, res.state.reset_pos_to(s))
            if isinstance(res_not, ParsingError):
                return res
            return ParsingError(s.replace_parser_if_not_observed_further(_but_not))

        return _but_not

    def when(self, pred: Callable[[B], bool]) -> "Parser[A, B]":
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
            funcparserlib.parser.NoParseError: got unexpected token: '0', expected: when(...)

            ```
        """

        @parser("when(...)")
        def _when(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingError) or pred(cast("ParsingSuccess[B]", res).value):
                return res
            return ParsingError(s.replace_parser_if_not_observed_further(_when))

        return _when

    @overload
    def __add__(  # type: ignore[overload-overlap]
        self, other: "IgnoredParser[A]"
    ) -> Self: ...

    @overload
    def __add__[C](self, other: "Parser[A, C]") -> "Tuple2Parser[A, B, C]": ...

    def __add__(self, other: "Parser[A, Any]") -> "Parser[A, Any]":
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

        @parser(f"({self.name}, {other.name})")
        def _add(tokens: Sequence[A], s: State) -> ParsingResult[Any]:
            res_l = self.run(tokens, s)
            if isinstance(res_l, ParsingSuccess):
                res_r = other.run(tokens, res_l.state)
                return res_l + res_r
            return cast("ParsingError[Any]", res_l)

        return _add

    def __or__[C](self, other: "Parser[A, C]") -> "Parser[A, B | C]":
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

        @parser(f"{self.name} or {other.name}")
        def _or(tokens: Sequence[A], s: State) -> ParsingResult[B | C]:
            res: ParsingResult[Any] = self.run(tokens, s)
            if isinstance(res, ParsingSuccess):
                return res

            s = res.state.reset_pos_to(s)
            res = other.run(tokens, s)
            if isinstance(res, ParsingSuccess):
                return res

            err = cast("ParsingError[B | C]", res)
            if s.pos == err.state.max:
                return ParsingError(State(err.state.pos, err.state.max, _or), err.message)
            return err

        return _or

    def map[C](self, f: Callable[[B], C]) -> "Parser[A, C]":
        """Transform the parsing result by applying the specified function; the same as `>>`."""

        @parser(self.name)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.map(f)

        return _map

    def starmap[C](self, f: Callable[..., C]) -> "Parser[A, C]":
        """Transform the parsing results (tuple) by applying the specified function."""

        @parser(self.name)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.starmap(f)

        return _map

    def __rshift__[C](self, f: Callable[[B], C]) -> "Parser[A, C]":
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

    def bind[C](self, f: Callable[[B], "Parser[A, C]"]) -> "Parser[A, C]":
        """Bind the parser to a monadic function that returns a new parser.

        Type: `(Callable[[B], Parser[A, C]]) -> Parser[A, C]`

        Also known as `>>=` in Haskell.

        !!! Note

            You can parse any context-free grammar without resorting to `bind`. Due
            to its poor performance please use it only when you really need it.
        """

        @parser(f"({self.name} >>=)")
        def _bind(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.bind(lambda v, s2: f(v).run(tokens, s2))

        return _bind

    def __neg__(self) -> "IgnoredParser[A]":
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
                `p1 + p2 + ... + pN`. The parsed value of `-p` is an **internal** `_Ignored`
                object, not intended for actual use.
        """

        @parser(self.name)
        def _ignored(tokens: Sequence[A], s: State) -> ParsingResult[IgnoredValue]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingSuccess):
                return ParsingSuccess(_IGNORED, res.state)
            return cast("ParsingError[IgnoredValue]", res)

        return cast("IgnoredParser[A]", _ignored)

    def __invert__(self) -> "Parser[A, A]":
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

        @parser(f"anything but {self.name}")
        def _invert(tokens: Sequence[A], s: State) -> ParsingResult[A]:
            if s.pos >= len(tokens):
                s2 = s.replace_parser_if_not_observed_further(_invert)
                return ParsingError(s2, "got unexpected end of input")
            res = self.run(tokens, s)
            if isinstance(res, ParsingError):
                return ParsingSuccess(tokens[s.pos], s.reset_pos(s.pos + 1))
            return ParsingError(s.replace_parser_if_not_observed_further(_invert))

        return _invert


def parser[A, B](name: str | None) -> Callable[[ParserFn[A, B]], Parser[A, B]]:
    """Make a named `Parser` from a parsing function, as a decorator.

    Type: `(str | None) -> Callable[[ParserFn[A, B]], Parser[A, B]]`

    The decorated function takes the tokens and the current `State` and returns a `ParsingSuccess`
    or a `ParsingError`.
    """

    def decorator(f: ParserFn[A, B]) -> Parser[A, B]:
        return Parser(f).named(name)

    return decorator


def _format_parsing_error(e: NoParseError, tokens: Sequence[Any]) -> None:
    max_pos = e.state.max
    if len(tokens) <= max_pos:
        msg = "got unexpected end of input"
    else:
        t_value = t = tokens[max_pos]
        loc = ""

        try:  # Token-like object
            if t.start is not None and t.end is not None:
                s_line, s_pos = t.start
                e_line, e_pos = t.end
                loc = f"{s_line},{s_pos}-{e_line},{e_pos}: "
            t_value = t.value
        except (AttributeError, TypeError):
            pass

        msg = f"{loc}{e.msg}: "
        if isinstance(t_value, str):
            msg += repr(t_value)
        else:
            msg += str(t_value)

    if isinstance(p := e.state.parser, Parser):
        msg = f"{msg}, expected: {p.name}"

    e.msg = msg


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
    def __add__(  # type: ignore[overload-overlap]
        self, other: "IgnoredParser[A]"
    ) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> "Tuple3Parser[A, T1, T2, C]": ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2], C]) -> "Parser[A, C]":
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2]], C]) -> Parser[A, C]:
        raise NotImplementedError


@final
class Tuple3Parser[A, T1, T2, T3](Parser[A, tuple[T1, T2, T3]]):
    """The type of a sequence of 3 parsers, `p1 + ... + p3`; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self, other: "IgnoredParser[A]"
    ) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> "Tuple4Parser[A, T1, T2, T3, C]": ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3], C]) -> "Parser[A, C]":
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3]], C]) -> Parser[A, C]:
        raise NotImplementedError


@final
class Tuple4Parser[A, T1, T2, T3, T4](Parser[A, tuple[T1, T2, T3, T4]]):
    """The type of a sequence of 4 parsers, `p1 + ... + p4`; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self, other: "IgnoredParser[A]"
    ) -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> "Tuple5Parser[A, T1, T2, T3, T4, C]": ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3, T4], C]) -> "Parser[A, C]":
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3, T4]], C]) -> Parser[A, C]:
        raise NotImplementedError


@final
class Tuple5Parser[A, T1, T2, T3, T4, T5](Parser[A, tuple[T1, T2, T3, T4, T5]]):
    """The type of a sequence of 5 parsers, `p1 + ... + p5`; a type for checkers, never built."""

    @overload  # type: ignore[override]
    def __add__(self, other: "IgnoredParser[A]") -> Self: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> "Parser[A, Any]": ...

    def __add__(self, other):  # ty: ignore[invalid-method-override] -- `+` narrowed per arity
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3, T4, T5], C]) -> "Parser[A, C]":
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


@parser("end of input")
def finished(tokens: Sequence[Any], s: State) -> ParsingResult[None]:
    """A parser that throws an exception if there are any unparsed tokens left in the sequence."""
    if s.pos >= len(tokens):
        return ParsingSuccess(None, s)
    else:
        return ParsingError(s.replace_parser_if_not_observed_further(finished))


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

    @parser(f"{{ {p.name} }}")
    def _many(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        acc = []
        while True:
            res = p.run(tokens, s)
            if isinstance(res, ParsingSuccess):
                if res.state.pos == s.pos:
                    raise with_hint(
                        ValueError(f"{p.name} matched no tokens inside many()"),
                        "many() repeats a parser until it fails, and one that matches nothing never"
                        " fails: repeat the parser that maybe(), many() or pure() wraps.",
                    )
                acc.append(res.value)
                s = res.state
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

    @parser("some(...)")
    def _some(tokens: Sequence[A], s: State) -> ParsingResult[A]:
        if s.pos >= len(tokens):
            s2 = s.replace_parser_if_not_observed_further(_some)
            return ParsingError(s2, "got unexpected end of input")

        t = tokens[s.pos]
        pos = s.pos + 1
        if pred(t):
            s2 = s.reset_pos(pos)
            if debug:
                _log.debug("*matched* %r, new state = %s", t, s2)
            return ParsingSuccess(t, s2)

        s2 = s.replace_parser_if_not_observed_further(_some)
        if debug and isinstance(s2.parser, Parser):
            _log.debug("failed %r, state = %s, expected: %s", t, s, s2.parser.name)
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
    return (p >> (lambda t: t.value)).named(p.name)


def pure[A](x: A) -> Parser[Any, A]:
    """Wrap any object into a parser.

    Type: `(A) -> Parser[A, A]`

    A pure parser doesn't touch the tokens sequence, it just returns its pure `x` value.

    Also known as `return` in Haskell.
    """

    @parser(f"(pure {x!r})")
    def _pure(_: Sequence[Any], s: State) -> ParsingResult[A]:
        return ParsingSuccess(x, s)

    return _pure


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
    return (p | pure(None)).named(f"[ {p.name} ]")


def skip[A](p: Parser[A, Any]) -> "IgnoredParser[A]":
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
    def __add__(self, other: "IgnoredParser[A]") -> Self: ...

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

    @parser(f"({p.name}, {{ {p.name} }})")
    def _oneplus(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        res: ParsingResult[Any]
        res = first = p.run(tokens, s)
        if isinstance(first, ParsingSuccess):
            res = rest = many_p.run(tokens, first.state)
            if isinstance(rest, ParsingSuccess):
                res = ParsingSuccess([first.value] + rest.value, rest.state)
        return cast("ParsingError[Any]", res)

    return _oneplus


@final
class ForwardDeclParser[A, B](Parser[A, B]):
    """A parser defined after it is used, for recursive grammars; see `forward_decl()`."""

    def __init__(self) -> None:
        def f(_tokens: Any, _s: Any) -> Any:
            raise NotImplementedError("you must define() a forward_decl() somewhere")

        super().__init__(f)
        self.named("forward_decl()")

    def define(self, p: ParserObjOrFn[A, B]) -> None:
        """Define the parser this forward declaration stands for."""
        self._define(p)

    # mypy (1.9.0) does not understand polymorphic Self in return, so duplicate it
    def named(self, name: str | None) -> Self:
        self.name = name
        return self


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
