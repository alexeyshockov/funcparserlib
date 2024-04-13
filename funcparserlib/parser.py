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

Parsing combinators define an internal domain-specific language (DSL) for describing
the parsing rules of a grammar. The DSL allows you to start with a few primitive
parsers, then combine your parsers to get more complex ones, and finally cover
the whole grammar you want to parse.

The structure of the language:

* Class `Parser`
    * All the primitives and combinators of the language return `Parser` objects
    * It defines the main `Parser.parse(tokens)` method
* Primitive parsers
    * `tok(type, value)`, `a(value)`, `some(pred)`, `forward_decl()`, `finished`
* Parser combinators
    * `p1 + p2`, `p1 | p2`, `p >> f`, `-p`, `maybe(p)`, `many(p)`, `oneplus(p)`,
      `skip(p)`
* Abstraction
    * Use regular Python variables `p = ...  # Expression of type Parser` to define new
      rules (non-terminals) of your grammar

Every time you apply one of the combinators, you get a new `Parser` object. In other
words, the set of `Parser` objects is closed under the means of combination.

!!! Note

    We took the parsing combinators language from the book [Introduction to Functional
    Programming][1] and translated it from ML into Python.

  [1]: https://www.cl.cam.ac.uk/teaching/Lectures/funprog-jrh-1996/
"""

__all__ = [
    "some",
    "a",
    "tok",
    "many",
    "pure",
    "finished",
    "maybe",
    "skip",
    "anything_but",
    "oneplus",
    "forward_decl",
    "NoParseError",
    "Parser",
]

import dataclasses as dc
import logging
import sys
import warnings
from collections.abc import Sequence
from copy import copy
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
    overload,
    Protocol,
    final,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from funcparserlib.lexer import Token

log = logging.getLogger("funcparserlib")

debug = False

_A = TypeVar("_A")
_B = TypeVar("_B", covariant=True)
_C = TypeVar("_C")

# Additional type variables tuple overloads (up to 5-tuple)
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")

_ParserFn = Callable[[Sequence[_A], "State"], tuple[_B, "State"]]
_ParserOrParserFn = Union["Parser[_A, _B]", _ParserFn]

_Place = tuple[int, int]


_DC_KWARGS: dict[str, bool] = {}
if sys.version_info >= (3, 10):
    _DC_KWARGS["slots"] = True


class TokenLike(Protocol):
    value: Any
    start: _Place
    end: _Place


@final
@dc.dataclass(frozen=True, init=False, **_DC_KWARGS)
class Parser(Generic[_A, _B]):
    """A parser object that can parse a sequence of tokens or can be combined with
    other parsers using `+`, `|`, `>>`, `many()`, and other parsing combinators.

    Type: `Parser[A, B]`

    The generic variables in the type are: `A` — the type of the tokens in the
    sequence to parse,`B` — the type of the parsed value.

    In order to define a parser for your grammar:

    1. You start with primitive parsers by calling `a(value)`, `some(pred)`,
       `forward_decl()`, `finished`
    2. You use parsing combinators `p1 + p2`, `p1 | p2`, `p >> f`, `many(p)`, and
       others to combine parsers into a more complex parser
    3. You can assign complex parsers to variables to define names that correspond to
       the rules of your grammar

    !!! Note

        The constructor `Parser.__init__()` is considered **internal** and may be
        changed in future versions. Use primitive parsers and parsing combinators to
        construct new parsers.
    """

    _logic: _ParserFn[_A, _B]
    name: str = dc.field(default="", compare=False)

    def __init__(self, p: _ParserOrParserFn[_A, _B]) -> None:
        """Wrap the parser function `p` into a `Parser` object."""
        self.define(p)

    def _with_name(self, name: str) -> Self:
        object.__setattr__(self, "name", name)
        return self

    def named(self, name: str) -> Self:
        # noinspection GrazieInspection
        """Specify the name of the parser for easier debugging.

        Type: `(str) -> Parser[A, B]`

        This name is used in the debug-level parsing log. You can also get it via the
        `Parser.name` attribute.

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
        return copy(self)._with_name(name)

    def _with_name_from(self, p: _ParserOrParserFn[_A, _B]) -> Self:
        if (name := getattr(p, "name", p.__doc__)) is not None:
            return self._with_name(name)
        return self

    # TODO Separate from the base parser...
    def define(self, p: _ParserOrParserFn[_A, _B]) -> None:
        """Define the parser created earlier as a forward declaration.

        Type: `(Parser[A, B]) -> None`

        Use `p = forward_decl()` in combination with `p.define(...)` to define
        recursive parsers.

        See the examples in the docs for `forward_decl()`.
        """
        f = _extract_logic(p)
        object.__setattr__(self, "_logic", f)
        # TODO Optimize later
        # if not debug:
        #     object.__setattr__(self, "run", f)

        self._with_name_from(p)

    def run(self, tokens: Sequence[_A], s: "State") -> tuple[_B, "State"]:
        """Run the parser against the tokens with the specified parsing state.

        Type: `(Sequence[A], State) -> tuple[B, State]`

        The parsing state includes the current position in the sequence being parsed,
        and the position of the rightmost token that has been consumed while parsing for
        better error messages.

        If the parser fails to parse the tokens, it raises `NoParseError`.

        !!! Warning

            This is method is **internal** and may be changed in future versions. Use
            `Parser.parse(tokens)` instead and let the parser object take care of
            updating the parsing state.
        """
        if debug:
            log.debug("trying %s" % self.name)
        try:
            return self._logic(tokens, s)
        except NoParseError as e:
            if e.state and e.state.parser == self:
                e.state = dc.replace(e.state, parser=self)
            raise

    def parse(self, tokens: Sequence[_A]) -> _B:
        """Parse the sequence of tokens and return the parsed value.

        Type: `(Sequence[A]) -> B`

        It takes a sequence of tokens of arbitrary type `A` and returns the parsed value
        of arbitrary type `B`.

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
            (tree, _) = self.run(tokens, State(0, 0, None))
            return tree
        except NoParseError as e:
            _format_parsing_error(e, tokens)
            raise

    def but_not(self, other: "Parser[_A, _B]") -> "Parser[_A, _B]":
        """Parse as usual, but only if the other parser fails."""

        @parser(f"{self.name} (but not {other.name})")
        def _but_not(tokens: Sequence[_A], s: State) -> tuple[Union[_B, _C], State]:
            (v, s2) = self.run(tokens, s)
            try:
                (_, s3) = other.run(tokens, dc.replace(s2, pos=s.pos))
                # (_, s3) = other.run(tokens, State(s.pos, s2.max, s2.parser))
                # (_, s3) = other.run(tokens, s)
            except NoParseError as e:
                return v, dc.replace(s2, max=max(s2.max, e.state.max))
                # return v, State(s2.pos, max(s2.max, e.state.max), s2.parser)

            raise NoParseError(
                "got unexpected token", State(s.pos, max(s2.max, s3.max), _but_not)
            )

        return _but_not

    @overload
    def __add__(  # type: ignore[overload-overlap]
        self,
        other: "_IgnoredParser[_A]",
    ) -> Self:
        pass

    @overload
    def __add__(self, other: "Parser[_A, _C]") -> "_Tuple2Parser[_A, _B, _C]":
        pass

    def __add__(self, other: "Parser") -> "Parser":
        """Sequential combination of parsers. It runs this parser, then the other
        parser.

        The return value of the resulting parser is a tuple of each parsed value in
        the sum of parsers. We merge all parsing results of `p1 + p2 + ... + pN` into a
        single tuple. It means that the parsing result may be a 2-tuple, a 3-tuple,
        a 4-tuple, etc. of parsed values. You avoid this by transforming the parsed
        pair into a new value using the `>>` combinator.

        You can also skip some parsing results in the resulting parsers by using `-p`
        or `skip(p)` for some parsers in your sum of parsers. It means that the parsing
        result might be a single value, not a tuple of parsed values. See the docs
        for `Parser.__neg__()` for more examples.

        Overloaded types (lots of them to provide stricter checking for the quite
        dynamic return type of this method):

        * `(self: Parser[A, B], _IgnoredParser[A]) -> Self`
        * `(self: _IgnoredParser[A], Parser[A, C]) -> Parser[A, C]`
        * `(self: Parser[A, B], Parser[A, C]) -> _Tuple2Parser[A, B, C]`
        * `(self: _Tuple2Parser[A, B, C], Parser[A, D]) -> _Tuple3Parser[A, B, C, D]`
        * `(self: _Tuple3Parser[A, B, C, D], Parser[A, E]) ->
           _Tuple4Parser[A, B, C, D, E]`
        * `(self: _Tuple4Parser[A, B, C, D, E], Parser[A, F]) ->
           _Tuple5Parser[A, B, C, D, E, F]`
        * `(self: _Tuple5Parser[A, B, C, D, E, F], Parser[A, Any]) -> Parser[A, Any]`

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
        parser.NoParseError: got unexpected token: 'z', expected: 'y'

        ```
        """
        name = "(%s, %s)" % (self.name, other.name)

        def magic(v1: Any, v2: Any) -> _Tuple:
            if isinstance(v1, _Tuple):
                return _Tuple(v1 + (v2,))
            else:
                return _Tuple((v1, v2))

        @parser(name)
        def _add(tokens: Sequence[_A], s: State) -> tuple[tuple, State]:
            (v1, s2) = self.run(tokens, s)
            (v2, s3) = other.run(tokens, s2)
            return magic(v1, v2), s3

        @parser(name)
        def ignored_right(tokens: Sequence[_A], s: State) -> tuple[_B, State]:
            v, s2 = self.run(tokens, s)
            _, s3 = other.run(tokens, s2)
            return v, s3

        return ignored_right if isinstance(other, _IgnoredParser) else _add

    def __or__(self, other: "Parser[_A, _C]") -> "Parser[_A, Union[_B, _C]]":
        """Choice combination of parsers.

        It runs this parser and returns its result. If the parser fails, it runs the
        other parser.

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
        parser.NoParseError: got unexpected token: 'z', expected: 'x' or 'y'

        ```
        """

        @parser(f"{self.name} or {other.name}")
        def _or(tokens: Sequence[_A], s: State) -> tuple[Union[_B, _C], State]:
            try:
                return self.run(tokens, s)
            except NoParseError as e:
                state = e.state
            try:
                return other.run(tokens, dc.replace(state, pos=s.pos))
            except NoParseError as e:
                if s.pos == e.state.max:
                    e.state = dc.replace(e.state, parser=_or)
                raise

        return _or

    def __rshift__(self, f: Callable[[_B], _C]) -> "Parser[_A, _C]":
        """Transform the parsing result by applying the specified function.

        Type: `(Callable[[B], C]) -> Parser[A, C]`

        You can use it for transforming the parsed value into another value before
        including it into the parse tree (the AST).

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

        @parser(self.name)
        def _shift(tokens: Sequence[_A], s: State) -> tuple[_C, State]:
            (v, s2) = self.run(tokens, s)
            try:
                return f(v), s2
            except NoParseError as e:
                if not e.state:
                    # Complete the error with the current state
                    e.state = State(s.pos, s2.max, _shift)
                raise

        return _shift

    def bind(self, f: Callable[[_B], "Parser[_A, _C]"]) -> "Parser[_A, _C]":
        """Bind the parser to a monadic function that returns a new parser.

        Type: `(Callable[[B], Parser[A, C]]) -> Parser[A, C]`

        Also known as `>>=` in Haskell.

        !!! Note

            You can parse any context-free grammar without resorting to `bind`. Due
            to its poor performance please use it only when you really need it.
        """

        @parser(f"({self.name} >>=)")
        def _bind(tokens: Sequence[_A], s: State) -> tuple[_C, State]:
            (v, s2) = self.run(tokens, s)
            return f(v).run(tokens, s2)

        return _bind

    def __neg__(self) -> "_IgnoredParser[_A]":
        """Return a parser that parses the same tokens, but its parsing result is
        ignored by the sequential `+` combinator.

        Type: `(Parser[A, B]) -> _IgnoredParser[A]`

        You can use it for throwing away elements of concrete syntax (e.g. `","`,
        `";"`).

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
        parser.NoParseError: got unexpected token: 'y', expected: 'x'

        ```

        ```pycon
        >>> expr = a("x") + -a("y")
        >>> expr.parse("xz")
        Traceback (most recent call last):
            ...
        parser.NoParseError: got unexpected token: 'z', expected: 'y'

        ```

        !!! Note

            You **should not** pass the resulting parser to any combinators other than
            `+`. You **should** have at least one non-skipped value in your
            `p1 + p2 + ... + pN`. The parsed value of `-p` is an **internal** `_Ignored`
            object, not intended for actual use.
        """
        return _IgnoredParser(self)

    def __invert__(self) -> "Parser[_A, _A]":
        # TODO Docs

        @parser(f"anything but {self.name}")
        def _invert(tokens: Sequence[_A], s: State) -> tuple[_A, State]:
            if s.pos >= len(tokens):
                s2 = s.replace_parser_if_not_observed_further(_invert)
                raise NoParseError("got unexpected end of input", s2)

            try:
                _, s2 = self.run(tokens, s)
            except NoParseError as e:
                t = tokens[s.pos]
                pos = s.pos + 1
                s2 = State(pos, max(pos, e.state.max), _invert)
                return t, s2

            # TODO Check state
            raise NoParseError("got unexpected token", State(s.pos, s2.max, _invert))

        return _invert


def _extract_logic(p: _ParserOrParserFn) -> _ParserFn:
    return getattr(p, "_logic", p)  # type: ignore


# Decorator to create named parsers directly
def parser(name: str) -> Callable[[_ParserFn[_A, _B]], Parser[_A, _B]]:
    def _parser(f: _ParserFn[_A, _B]) -> Parser[_A, _B]:
        # noinspection PyProtectedMember
        return Parser(f)._with_name(name)

    return _parser


@dc.dataclass(frozen=True, repr=False, **_DC_KWARGS)
class State:
    """Parsing state that is maintained basically for error reporting.

    It consists of the current position `pos` in the sequence being parsed, and the
    position `max` of the rightmost token that has been consumed while parsing.
    """

    pos: int
    max: int
    parser: Optional[_ParserOrParserFn] = None

    def replace_parser_if_not_observed_further(self, p: _ParserOrParserFn) -> Self:
        return dc.replace(self, parser=p) if self.pos == self.max else self

    def __str__(self) -> str:
        return str((self.pos, self.max))

    def __repr__(self) -> str:
        return "State(%r, %r)" % (self.pos, self.max)


class NoParseError(Exception):
    def __init__(self, msg: str, state: State) -> None:
        self.msg = msg
        self.state = state

    def __str__(self) -> str:
        return self.msg


def _format_parsing_error(e: NoParseError, tokens: Sequence) -> None:
    max_pos = e.state.max
    if len(tokens) <= max_pos:
        msg = "got unexpected end of input"
    else:
        t_value = t = tokens[max_pos]
        loc = ""

        try:  # Token-like object
            t_value = t.value
            if t.start is not None and t.end is not None:
                s_line, s_pos = t.start
                e_line, e_pos = t.end
                loc = f"{s_line},{s_pos}-{e_line},{e_pos}: "
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


class _Tuple(tuple):
    pass


@dc.dataclass(frozen=True)
class _Tuple2Parser(  # type: ignore[misc]
    Parser[_A, tuple[_T1, _T2]], Generic[_A, _T1, _T2]
):
    """
    Just for type inference, not intended for actual use.
    """

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self,
        other: "_IgnoredParser[_A]",
    ) -> Self:
        pass

    @overload
    def __add__(self, other: Parser[_A, _C]) -> "_Tuple3Parser[_A, _T1, _T2, _C]":
        pass

    def __add__(self, other):  # type: ignore[no-untyped-def]
        return super().__add__(other)

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open _B explicitly here
    def __rshift__(self, f: Callable[[tuple[_T1, _T2]], _C]) -> Parser[_A, _C]:
        return super().__rshift__(f)


@dc.dataclass(frozen=True)
class _Tuple3Parser(  # type: ignore[misc]
    Parser[_A, tuple[_T1, _T2, _T3]], Generic[_A, _T1, _T2, _T3]
):  # type: ignore[misc]
    """
    Just for type inference, not intended for actual use.
    """

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self,
        other: "_IgnoredParser[_A]",
    ) -> Self:
        pass

    @overload
    def __add__(self, other: Parser[_A, _C]) -> "_Tuple4Parser[_A, _T1, _T2, _T3, _C]":
        pass

    def __add__(self, other):  # type: ignore[no-untyped-def]
        return super().__add__(other)

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open _B explicitly here
    def __rshift__(self, f: Callable[[tuple[_T1, _T2, _T3]], _C]) -> Parser[_A, _C]:
        return super().__rshift__(f)


@dc.dataclass(frozen=True)
class _Tuple4Parser(  # type: ignore[misc]
    Parser[_A, tuple[_T1, _T2, _T3, _T4]], Generic[_A, _T1, _T2, _T3, _T4]
):
    """
    Just for type inference, not intended for actual use.
    """

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self,
        other: "_IgnoredParser[_A]",
    ) -> Self:
        pass

    @overload
    def __add__(
        self, other: Parser[_A, _C]
    ) -> "_Tuple5Parser[_A, _T1, _T2, _T3, _T4, _C]":
        pass

    def __add__(self, other):  # type: ignore[no-untyped-def]
        return super().__add__(other)

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open _B explicitly here
    def __rshift__(
        self, f: Callable[[tuple[_T1, _T2, _T3, _T4]], _C]
    ) -> Parser[_A, _C]:
        return super().__rshift__(f)


@dc.dataclass(frozen=True)
class _Tuple5Parser(  # type: ignore[misc]
    Parser[_A, tuple[_T1, _T2, _T3, _T4, _T5]], Generic[_A, _T1, _T2, _T3, _T4, _T5]
):
    """
    Just for type inference, not intended for actual use.
    """

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self,
        other: "_IgnoredParser[_A]",
    ) -> Self:
        pass

    @overload
    def __add__(self, other: Parser[_A, _C]) -> "Parser[_A, Any]":
        pass

    def __add__(self, other):  # type: ignore[no-untyped-def]
        return super().__add__(other)

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open _B explicitly here
    def __rshift__(
        self, f: Callable[[tuple[_T1, _T2, _T3, _T4, _T5]], Any]
    ) -> Parser[_A, _C]:
        return super().__rshift__(f)


@dc.dataclass(frozen=True, repr=False, **_DC_KWARGS)
class _Ignored:
    value: Any

    def __repr__(self) -> str:
        return "_Ignored(%s)" % repr(self.value)


@parser("end of input")
def finished(tokens: Sequence[Any], s: State) -> tuple[None, State]:
    """A parser that throws an exception if there are any unparsed tokens left in the
    sequence."""
    if s.pos >= len(tokens):
        return None, s
    else:
        s2 = s.replace_parser_if_not_observed_further(finished)
        raise NoParseError("got unexpected token", s2)


def many(p: Parser[_A, _B]) -> Parser[_A, list[_B]]:
    """Return a parser that applies the parser `p` as many times as it succeeds at
    parsing the tokens.

    Return a parser that infinitely applies the parser `p` to the input sequence
    of tokens as long as it successfully parses them. The parsed value is a list of
    the sequentially parsed values.

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

    @parser("{ %s }" % p.name)
    def _many(tokens: Sequence[_A], s: State) -> tuple[list[_B], State]:
        res = []
        try:
            while True:
                (v, s) = p.run(tokens, s)
                res.append(v)
        except NoParseError as e:
            s2 = dc.replace(e.state, pos=s.pos)
            if debug:
                log.debug(
                    "*matched* %d instances of %s, new state = %s"
                    % (len(res), _many.name, s2)
                )
            return res, s2

    return _many


def when(p: Parser[_A, _B], pred: Callable[[_B], bool]) -> Parser[_A, _B]:
    """Wrap the parser to a new one, that parses a token if it satisfies the
    predicate `pred`.

    Type: `(Parser[A, B], Callable[[B], bool]) -> Parser[A, B]`

    Examples: TODO
    """

    @parser("when(...)")
    def _when(tokens: Sequence[_A], s: State) -> tuple[_B, State]:
        (v, s2) = p.run(tokens, s)
        if not pred(v):
            raise NoParseError("got unexpected token", State(s.pos, s2.max, _when))
        return v, s2

    return _when


def some(pred: Callable[[_A], bool]) -> Parser[_A, _A]:
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
    parser.NoParseError: got unexpected token: '1', expected: alpha

    ```

    !!! Warning

        The `some()` combinator is quite slow and may be changed or removed in future
        versions. If you need a parser for a token by its type (e.g. any identifier)
        and maybe its value, use `tok(type[, value])` instead. You should use
        `make_tokenizer()` from `funcparserlib.lexer` to tokenize your text first.
    """

    @parser("some(...)")
    def _some(tokens: Sequence[_A], s: State) -> tuple[_A, State]:
        if s.pos >= len(tokens):
            s2 = s.replace_parser_if_not_observed_further(_some)
            raise NoParseError("got unexpected end of input", s2)
        else:
            t = tokens[s.pos]
            if pred(t):
                pos = s.pos + 1
                s2 = State(pos, max(pos, s.max), s.parser)
                if debug:
                    log.debug("*matched* %r, new state = %s" % (t, s2))
                return t, s2
            else:
                s2 = s.replace_parser_if_not_observed_further(_some)
                if debug and isinstance(s2.parser, Parser):
                    log.debug(
                        "failed %r, state = %s, expected = %s" % (t, s2, s2.parser.name)
                    )
                raise NoParseError("got unexpected token", s2)

    return _some


def a(value: _A) -> Parser[_A, _A]:
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
    parser.NoParseError: got unexpected token: 'y', expected: 'x'

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

    def eq_value(t: _A) -> bool:
        return t == value

    return some(eq_value)._with_name(repr(name))


# noinspection PyShadowingBuiltins
def tok(type: str, value: Optional[str] = None) -> Parser[Token, str]:
    """Return a parser that parses a `Token` and returns the string value of the token.

    Type: `(str, Optional[str]) -> Parser[Token, str]`

    You can match any token of the specified `type` or you can match a specific token by
    its `type` and `value`.

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
    parser.NoParseError: got unexpected token: '=', expected: expr

    ```

    ```pycon
    >>> expr = tok("op", "=")
    >>> expr.parse([Token("op", "=")])
    '='
    >>> expr.parse([Token("op", "+")])
    Traceback (most recent call last):
        ...
    parser.NoParseError: got unexpected token: '+', expected: '='

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

    if value is not None:
        p = a(Token(type, value))
    else:
        p = some(eq_type)._with_name(type)
    return (p >> (lambda t: t.value))._with_name(p.name)


def pure(x: _A) -> Parser[Any, _A]:
    """Wrap any object into a parser.

    Type: `(A) -> Parser[A, A]`

    A pure parser doesn't touch the tokens sequence, it just returns its pure `x`
    value.

    Also known as `return` in Haskell.
    """

    @parser("(pure %r)" % (x,))
    def _pure(_: Sequence[Any], s: State) -> tuple[_A, State]:
        return x, s

    return _pure


def maybe(p: Parser[_A, _B]) -> Parser[_A, Optional[_B]]:
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
    return (p | pure(None))._with_name("[ %s ]" % (p.name,))


def skip(p: Parser[_A, Any]) -> "_IgnoredParser[_A]":
    """An alias for `-p`.

    See also docs for `Parser.__neg__()`.
    """
    return -p


def anything_but(p: Parser[_A, Any]) -> Parser[_A, _A]:
    """An alias for `~p`.

    See also docs for `Parser.__invert__()`.
    """
    return ~p


@dc.dataclass(frozen=True, **_DC_KWARGS)
class _IgnoredParser(Parser[_A, Any]):  # type: ignore[misc]
    def __init__(self, p: _ParserOrParserFn[_A, Any]) -> None:
        super(_IgnoredParser, self).__init__(p)
        f = self._logic

        def ignored(tokens: Sequence[_A], s: State) -> tuple[Any, State]:
            v, s2 = f(tokens, s)
            return v if isinstance(v, _Ignored) else _Ignored(v), s2

        self.define(ignored)
        self._with_name_from(p)

    @overload  # type: ignore[override]
    def __add__(self, other: "_IgnoredParser[_A]") -> Self:
        pass

    @overload
    def __add__(self, other: Parser[_A, _C]) -> Parser[_A, _C]:
        pass

    def __add__(
        self, other: Union["_IgnoredParser[_A]", Parser[_A, _C]]
    ) -> Union["_IgnoredParser[_A]", Parser[_A, _C]]:
        name = f"({self.name}, {other.name})"
        if isinstance(other, _IgnoredParser):

            @_IgnoredParser
            def ip(tokens: Sequence[_A], s: State) -> tuple[Any, State]:
                _, s2 = self.run(tokens, s)
                v, s3 = other.run(tokens, s2)
                return v, s3

            return ip._with_name(name)
        else:

            @parser(name)
            def p(tokens: Sequence[_A], s: State) -> tuple[_C, State]:
                _, s2 = self.run(tokens, s)
                v, s3 = other.run(tokens, s2)
                return v, s3

            return p


def oneplus(p: Parser[_A, _B]) -> Parser[_A, list[_B]]:
    """Return a parser that applies the parser `p` one or more times.

    A similar parser combinator `many(p)` means apply `p` zero or more times, whereas
    `oneplus(p)` means apply `p` one or more times.

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
    parser.NoParseError: got unexpected token: 'y', expected: 'x'

    ```
    """

    @parser("(%s, { %s })" % (p.name, p.name))
    def _oneplus(tokens: Sequence[_A], s: State) -> tuple[list[_B], State]:
        (v1, s2) = p.run(tokens, s)
        (v2, s3) = many(p).run(tokens, s2)
        return [v1] + v2, s3

    return _oneplus


def with_forward_decls(suspension: Callable[[], Parser[_A, _B]]) -> Parser[_A, _B]:
    warnings.warn(
        "Use forward_decl() instead:\n"
        "\n"
        "    p = forward_decl()\n"
        "    ...\n"
        "    p.define(parser_value)\n",
        DeprecationWarning,
    )

    @Parser
    def f(tokens: Sequence[_A], s: State) -> tuple[_B, State]:
        return suspension().run(tokens, s)

    return f


def forward_decl() -> Parser[Any, Any]:
    """Return an undefined parser that can be used as a forward declaration.

    Type: `Parser[Any, Any]`

    Use `p = forward_decl()` in combination with `p.define(...)` to define recursive
    parsers.


    Examples:

    ```pycon
    >>> expr = forward_decl()
    >>> expr.define(a("x") + maybe(expr) + a("y"))
    >>> expr.parse("xxyy")  # noqa
    ('x', ('x', None, 'y'), 'y')
    >>> expr.parse("xxy")
    Traceback (most recent call last):
        ...
    parser.NoParseError: got unexpected end of input, expected: 'y'

    ```

    !!! Note

        If you care about static types, you should add a type hint for your forward
        declaration, so that your type checker can check types in `p.define(...)` later:

        ```python
        p: Parser[str, int] = forward_decl()
        p.define(a("x"))  # Type checker error
        p.define(a("1") >> int)  # OK
        ```
    """

    @parser("forward_decl()")
    def f(_tokens: Any, _s: Any) -> Any:
        raise NotImplementedError("you must define() a forward_decl somewhere")

    return f


if __name__ == "__main__":
    import doctest

    doctest.testmod()
