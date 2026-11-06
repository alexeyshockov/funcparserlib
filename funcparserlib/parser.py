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
    "CompiledParser",
    "longest",
    "prefixed",
    "counted",
    "switch",
    "ForwardDeclParser",
    "State",
    "NoParseError",
    "ParsingResult",
    "ParsingSuccess",
    "ParsingFailure",
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
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import Any, NoReturn, Protocol, Self, TypeGuard, cast, final, overload

from ._commons import valueclass, with_hint
from .lexer import LexerError, Place, Token, Tokenizer

_log = logging.getLogger("funcparserlib")

debug: bool = False


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


_WORD = re.compile(r"\S{1,20}")
_UNREADABLE = "text no token spec matches"


class _Undecided(Exception):
    """Raised by generated code: the interpreted parser has to decide this parse."""


class _Character:
    """What an error shows of the text at an offset, with the place of that offset."""

    __slots__ = ("end", "start", "value")

    def __init__(self, value: str, place: Place) -> None:
        self.value = value
        self.start = place
        self.end = place


class _Byte(int):
    """A byte, shown in hexadecimal with its offset."""

    offset: int

    def __new__(cls, value: int, offset: int) -> Self:
        byte = super().__new__(cls, value)
        byte.offset = offset
        return byte

    def __str__(self) -> str:
        return f"0x{int(self):02x} at offset 0x{self.offset:x}"


class _Offsets(Sequence[Any]):
    """The bytes of a buffer, for errors: each knows its offset."""

    __slots__ = ("_data",)

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: Any) -> Any:
        return _Byte(self._data[i], i)


class _Located(Sequence[Any]):
    """The characters of a text, for errors: each knows its line and column."""

    __slots__ = ("_skip", "_text")

    def __init__(self, text: str, skip: re.Pattern[str] | None = None) -> None:
        self._text = text
        self._skip = skip

    def __len__(self) -> int:
        return len(self._text)

    def __getitem__(self, i: Any) -> Any:
        text = self._text
        offset = i + len(text) if i < 0 else i
        line = text.count("\n", 0, offset) + 1
        column = offset - text.rfind("\n", 0, offset)
        word = _WORD.match(text, offset)
        return _Character(word.group() if word else text[offset], (line, column))

    def _place(self, offset: int) -> Place:
        text = self._text
        return text.count("\n", 0, offset) + 1, offset - text.rfind("\n", 0, offset)

    def end(self) -> TokenDescription:
        """The end of the text, past its last character that is not whitespace."""
        place = self._place(len(self._text.rstrip()))
        return TokenDescription("end of input", place, place)

    def span(self, start: int, end: int) -> TokenDescription:
        """The text from `start` to `end`, exclusive, without the skip and whitespace at its end."""
        if self._skip is not None:
            for stop in range(start + 1, end):
                if self._skip.fullmatch(self._text, stop, end):
                    end = stop
                    break
        text = self._text[start:end].rstrip()
        last = start + max(len(text) - 1, 0)
        return TokenDescription(repr(text), self._place(start), self._place(last))


def _shown(tokens: Sequence[Any], i: int) -> str:
    """The token at `i` in an error that gives its place apart: a word of a text, a byte in hex."""
    if isinstance(tokens, bytes):
        return f"0x{tokens[i]:02x}"
    word = _WORD.match(tokens, i) if isinstance(tokens, str) else None
    return _describe(word[0] if word else tokens[i]).text


def _is_place(x: object) -> TypeGuard[Place]:
    return isinstance(x, tuple) and len(x) == 2 and all(isinstance(n, int) for n in x)


type Expectation = tuple[Parser[Any, Any], tuple[str, ...]]
"""A parser that failed at the furthest position, with the `Parser.context()` labels around it."""


@final
@valueclass
class CheckFailure:
    """A failed `Parser.check()`: its message, the tokens it covers (`start` to `end`, exclusive)
    and the `Parser.context()` labels around it.

    With `unexpected`, the error starts by naming the token at `start` as `Parser.parse()` was told
    to describe it, and `message` is what comes after that.
    """

    message: str
    start: int
    end: int
    context: tuple[str, ...]
    unexpected: bool = False


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

    def _reset_pos_to(self, s: State) -> State:
        """`s`, with the furthest position and what failed there from this state."""
        # dataclasses.replace() hits performance, as it performs additional validation
        return State(s.pos, self.max, self.expected, s.context, s.problems)

    def reset_pos(self, pos: int) -> State:
        """This state at `pos`; reaching past `max` moves it, and forgets what failed there."""
        if pos > self.max:
            return State(pos, pos, (), self.context, self.problems)
        return State(pos, self.max, self.expected, self.context, self.problems)

    def __str__(self) -> str:
        """The position and the furthest position, as a tuple."""
        return str((self.pos, self.max))

    def __repr__(self) -> str:
        """The state as the call that builds it from its two positions."""
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

    _result: ParsingFailure | None = None

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
        """Build an error from its message, the state parsing stopped at and the optional details.

        `token`, `expected`, `context` and `description` are the attributes of the same names.
        """
        self.msg: str = msg
        self.state: State = state
        self.token: object = token
        self.expected: tuple[str, ...] = expected
        self.context: tuple[str, ...] = context
        self.description: TokenDescription | None = description

    def __str__(self) -> str:
        """The message, `msg`."""
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
        """A `ParseErrors` with the same message over `excs`."""
        return ParseErrors(self.message, excs)

    def render(self, source: str | None = None) -> str:
        """Each error's `NoParseError.render()`, in order, separated by blank lines."""
        return "\n\n".join(e.render(source) for e in self.exceptions)


class ParsingResult[R](Iterable[Any], Protocol):
    """Result monad for parsing combinators.

    Immutable (object's data should not be changed after creation). `ParsingSuccess` and
    `ParsingFailure` implement it structurally.
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
        """Iterate over the value, then the state, so that a result unpacks as `value, state`."""
        yield self.value
        yield self.state

    def map[C](self, f: Callable[[R], C]) -> ParsingResult[C]:
        """A success of `f` applied to the value, at the same state."""
        # mypy checks ParsingSuccess against ParsingResult through this very method, and fails.
        return ParsingSuccess(f(self.value), self.state)  # type: ignore[return-value]

    def starmap[C](self, f: Callable[..., C]) -> ParsingResult[C]:
        """A success of `f` applied to the elements of the value, which must be a tuple."""
        return ParsingSuccess(f(*cast("tuple[Any, ...]", self.value)), self.state)

    def bind[C](self, f: Callable[[R, State], ParsingResult[C]]) -> ParsingResult[C]:
        """The result of `f` called with the value and the state."""
        return f(self.value, self.state)

    def __add__[C](self, other: ParsingResult[C]) -> ParsingResult[Any]:
        """Combine with `other`, as `Parser.__add__()` combines the values of two parsers.

        A failure `other` is returned as it is; an ignored value on either side is dropped from the
        tuple.
        """
        # Assume that the other state is further away from the current one
        # (matches Parser.__add__() behavior)
        if isinstance(other, ParsingFailure):
            return other
        v1, v2, s2 = self.value, other.value, other.state
        if isinstance(v1, IgnoredValue):
            return other
        if isinstance(v2, IgnoredValue):
            return ParsingSuccess(v1, s2)
        return ParsingSuccess(_Tuple.combine(v1, v2), s2)


@final
@valueclass
class ParsingFailure:
    """A failed parsing result: where parsing stopped, and why.

    It carries no value, so it is a `ParsingResult` of any value type; its `value` raises
    `NoParseError`. A `committed` failure (see `commit()`) is final: no alternative is tried.
    """

    state: State
    message: str = "got unexpected token"
    committed: bool = False

    def _replace_state(self, s: State) -> ParsingFailure:
        """The same failure at the state `s`."""
        return ParsingFailure(s, self.message, self.committed)

    @property
    def _error(self) -> NoParseError:
        e = NoParseError(self.message, self.state)
        e._result = self
        return e

    @property
    def value(self) -> NoReturn:
        """Never returns: raises the `NoParseError` for this failure."""
        raise self._error

    def __iter__(self) -> Iterator[Any]:
        """Never iterates: raises the `NoParseError` for this failure."""
        raise self._error

    def map(self, f: Callable[[Any], object]) -> ParsingFailure:
        """This failure; `f` is not called."""
        return self

    def starmap(self, f: Callable[..., object]) -> ParsingFailure:
        """This failure; `f` is not called."""
        return self

    def bind(self, f: Callable[[Any, State], object]) -> ParsingFailure:
        """This failure; `f` is not called."""
        return self

    def __add__(self, other: object) -> ParsingFailure:
        """This failure; `other` is not considered."""
        return self


type ParserFn[A, B] = Callable[[Sequence[A], State], ParsingResult[B]]
"""A parsing function: the tokens and the state in, a parsing result out."""

type ParserObjOrFn[A, B] = Parser[A, B] | ParserFn[A, B]
"""A `Parser` or a bare parsing function, as `Parser()` and `define()` accept."""


class Parser[A, B]:
    """A parser object that can parse a sequence of tokens or can be combined with other parsers
    using `+`, `|`, `>>`, `many()`, and other parsing combinators.

    Extensible by design: `Parser(f)` makes a parser of the parsing function `f`, as the `parser()`
    decorator does with a name, and that function is what a grammar adds to the library. A subclass
    may add to a parser, as `forward_decl()` does; overriding `run` is not supported.

    The generic variables in the type are: `A` — the type of the tokens in the sequence to parse,`B`
    — the type of the parsed value.

    In order to define a parser for your grammar:

    1. You start with primitive parsers by calling `a(value)`, `some(pred)`, `forward_decl()`,
       `finished`
    2. You use parsing combinators `p1 + p2`, `p1 | p2`, `p >> f`, `many(p)`, and others to combine
       parsers into a more complex parser
    3. You can assign complex parsers to variables to define names that correspond to the rules of
       your grammar
    """

    run: ParserFn[A, B]
    """Run the parser against the tokens with the specified parsing state.

    The parsing state includes the current position in the sequence being parsed, and the position
    of the rightmost token that has been consumed while parsing for better error messages.

    It returns a `ParsingSuccess` with the parsed value and the state after it, or a
    `ParsingFailure` if the parser fails; it does not raise `NoParseError`.

    A parsing function calls it to run other parsers; see `parser()`. `Parser.named()` and
    `ForwardDeclParser.define()` replace it, so read `p.run` at each call instead of keeping it.
    `Parser(p)` and `define(p)` keep the `run` that `p` has then, so a later `p.named()` does not
    reach them. To parse a whole input, use `Parser.parse(tokens)`, which raises an error for a
    failure.
    """

    name: str | None = None
    _fn: ParserFn[A, B]
    _label: bool = False
    _leaf: bool = False
    _shape: bool = False
    _test: _TokenTest | None = None
    # The keys, `_key(token)`, of every token the parser can start with, when known; a parser with
    # them consumes at least that token.
    _first: frozenset[object] | None = None
    # What the parser is built from, for `compile()`: a tag and its operands.
    _form: tuple[Any, ...] | None = None

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
        self._shape = False
        self._update_run()
        return self

    def _rename(self, name: str | None) -> Self:
        """Set the name the library gives a parser it builds, without making it a label."""
        self.name = name
        return self

    def _named_from(self, p: ParserObjOrFn[A, B]) -> Self:
        if not self._label and (name := getattr(p, "name", p.__doc__)) is not None:
            self._shape = getattr(p, "_shape", False)
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
            if isinstance(res, ParsingFailure) and res.state.max == s.pos:
                # Nothing got past this parser's start: it, not its parts, is what was expected.
                st = res.state
                expected = (*s.expected, (self, s.context))
                failed = State(st.pos, st.max, expected, st.context, st.problems)
                return ParsingFailure(failed, res.message, res.committed)
            return res

        return run_labelled

    def _define(self, p: ParserObjOrFn[A, B], *, adapt: bool = True) -> None:
        """Define the parser created earlier as a forward declaration.

        Use `p = forward_decl()` in combination with `p.define(...)` to define recursive parsers.

        See the examples in the docs for `forward_decl()`.
        """
        self._form = ("ref", p) if isinstance(p, Parser) else None
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
                # A failure unpacked inside `f` is taken as one at these tokens. A `NoParseError`
                # from a nested `parse()` carries none, and fails at this parser.
                res = e._result or ParsingFailure(s.expect(self), e.msg)
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

    def _fused[C](self, conv: Callable[[Any], C]) -> Parser[A, C]:
        """This token test giving `conv` of the token: one parser where `>>` or `-p` wraps it."""
        test = cast("_TokenTest", self._test)
        chain = test.chain if self is test.leaf else (self, *test.chain)
        # After the conversion the test has: `-(p >> f)` still calls `f`.
        prev = test.conv
        then = conv if prev is None else lambda t: conv(prev(t))
        convs = (*test.convs, conv)
        p: Parser[A, C] = _token_test(
            _TokenTest(test.leaf, chain, test.pred, then, test.token, convs)
        )
        p._rename(self.name)
        p._shape = self._shape
        p._first = self._first
        return p

    def _wrap_for_debug(self, f: ParserFn[A, B]) -> ParserFn[A, B]:
        def run_parser_verbose(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            _log.debug("trying %s", self.name)
            return f(tokens, s)

        return run_parser_verbose

    def parse(
        self, tokens: Sequence[A], *, describe: Callable[[A], TokenDescription] | None = None
    ) -> B:
        """Parse the sequence of tokens and return the parsed value.

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
        return self._result(self.run(tokens, State(0, 0)), tokens, describe or _describe)

    def parse_text(self, text: str) -> B:
        """Parse the text, with the line and column of the text in errors.

        It is for a grammar that reads text: the parsers of `funcparserlib.text.Text`, and `a()` and
        `some()` of a character. A grammar of `tok()` reads tokens: give those to `Parser.parse()`,
        or a tokenizer to `Parser.compile()`.

        It raises what `Parser.parse()` raises.
        """
        # Imported here: the module reads the classes of this one.
        from ._compile import leading_skip  # noqa: PLC0415

        skip = leading_skip(self)
        m = skip.match(text) if skip is not None else None
        start = m.end() if m is not None else 0
        res = self.run(cast("Any", text), State(start, start))
        return self._result(res, _Located(text, skip), _describe)

    def parse_bytes(self, data: bytes) -> B:
        """Parse the bytes, with the offset of the bytes in errors, in hexadecimal.

        It is for a grammar of the parsers of `funcparserlib.binary.Binary`, and raises what
        `Parser.parse()` raises.
        """
        res = self.run(cast("Any", data), State(0, 0))
        return self._result(res, _Offsets(data), _describe)

    def _result(
        self, res: ParsingResult[B], tokens: Sequence[Any], show: Callable[[Any], TokenDescription]
    ) -> B:
        """The value of `res`, or the errors of a failed parse and of failed checks."""
        errors = [_problem_error(problem, tokens, show) for problem in res.state.problems]
        if isinstance(res, ParsingFailure):
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

    def compile(
        self, tokenizer: Tokenizer | None = None, *, source_file: str | PathLike[str] | None = None
    ) -> CompiledParser[A, B]:
        """Generate Python functions that parse as this parser does, several times faster.

        Use it on a finished grammar that parses a lot of tokens, or a lot of times. The generated
        functions handle the parses that succeed; for any other, `CompiledParser.parse()` runs this
        parser on the same tokens, so the result and the errors are this parser's.

        That has a cost: a failed parse takes longer, and calls the functions given to `>>` a second
        time. So does a parse that succeeds after a `Parser.check()` failed on a branch `|`
        backtracked from, or after a `commit()` around a parsing function failed. The compiled
        parser is a snapshot of the grammar, and does not follow a `forward_decl()` defined again
        afterwards. A grammar nested too deeply to compile stays interpreted, as does a `many()` of
        a parser that matches no tokens.

        Given the `tokenizer` of the grammar, `CompiledParser.parse_text()` parses text, faster and
        in less memory still. A grammar of `funcparserlib.text.Text` or
        `funcparserlib.binary.Binary` needs no tokenizer.

        Given a `source_file`, the Python source of the generated functions is written there, and
        their frames report that file, so tracebacks, `pdb` and an IDE can show them. Given a
        tokenizer too, the file holds what `CompiledParser.parse_text()` runs, then what
        `CompiledParser.parse()` runs. The file is rewritten on each call and left alone for a
        grammar that stays interpreted, and `OSError` reaches the caller. It is for reading: the
        names in it stand for objects of the running parser, so it cannot be imported.

        Examples:
            ```pycon
            >>> compiled = (many(a("x")) + a("y")).compile()
            >>> compiled.parse("xxy")
            (['x', 'x'], 'y')
            >>> compiled.parse("xz")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.NoParseError: got unexpected token: 'z', expected: 'x' or 'y'

            ```
        """
        # Imported here: the module reads the classes of this one.
        from ._compile import (  # noqa: PLC0415
            Internals,
            compile_grammar,
            compile_text,
            line_after,
            reads_text,
            reads_tokens,
            write_source,
        )

        path = None if source_file is None else str(Path(source_file).resolve())
        internals = Internals(_Tuple, _IGNORED, _ignore, _token_value, _Undecided)
        # What reads text reads bytes too: generated code takes either.
        if reads_text(self):
            fast_text, source = compile_text(self, internals, None, path)
            write_source(path, source)
            return CompiledParser(self, None, None, fast_text, reads_tokens=False)
        fast_text, text_source = None, None
        if tokenizer is not None and tokenizer._pattern is not None:
            fast_text, text_source = compile_text(self, internals, tokenizer._scan, path)
        fast, source = compile_grammar(self, internals, path, line_after(text_source))
        write_source(path, text_source, source)
        return CompiledParser(self, fast, tokenizer, fast_text, reads_tokens=reads_tokens(self))

    def recover(self, until: Parser[A, Any], default: B | None = None) -> Parser[A, B | None]:
        """Record the error of a failure and go on from where `until` matches.

        It gives one parse every error of a document, for a grammar of parts that a line or a
        delimiter separates: `Parser.parse()` raises the recorded errors together as `ParseErrors`,
        as it does for `Parser.check()`. When this parser fails past its first token, the parser
        records what it expected, skips to the first position where `until` matches, without reading
        it, and gives `default`. It fails as usual at its first token, so `many(p.recover(...))`
        ends as `many(p)` does. The text runs out when `until` never matches.

        Examples:
            ```pycon
            >>> line = (a("x") + a("=") + a("1") + a(";")).recover(until=a("x"))
            >>> many(line).parse("x=1;x=1;")
            [('x', '=', '1', ';'), ('x', '=', '1', ';')]
            >>> many(line).parse("x=?;x?1;x=1;")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.ParseErrors: 2 parsing errors (2 sub-exceptions)

            ```
        """

        @_parser(self.name, shape=self._shape, first=self._first)
        def _recover(tokens: Sequence[A], s: State) -> ParsingResult[B | None]:
            # A fresh furthest position tells whether this parser read a token.
            res = self.run(tokens, State(s.pos, s.pos, (), s.context, s.problems))
            st = res.state
            if res.__class__ is not ParsingFailure:
                return ParsingSuccess(res.value, _merged(st, s))
            if st.max == s.pos:
                return res._replace_state(_merged(st, s))
            names = dict.fromkeys(p.name for p, _ in st.expected if p.name is not None)
            expected = f", expected: {' or '.join(names)}" if names else ""
            # Only `Parser.parse()` knows how to describe a token that is no character or byte.
            unexpected = st.max < len(tokens) and not isinstance(tokens, str | bytes)
            found = "got unexpected end of input"
            if st.max < len(tokens):
                found = "" if unexpected else f"got unexpected token: {_shown(tokens, st.max)}"
            problem = CheckFailure(found + expected, st.max, st.max + 1, st.context, unexpected)
            pos = st.max
            while (
                pos < len(tokens) and until.run(tokens, State(pos, pos)).__class__ is ParsingFailure
            ):
                pos += 1
            return ParsingSuccess(default, State(pos, pos, (), s.context, (*st.problems, problem)))

        # What fails past its first token leaves the generated code, as a `commit()` failure does.
        _recover._form = ("commit", self)
        return _recover

    def but_not(self, other: Parser[A, Any]) -> Parser[A, B]:
        """Parse as usual, but only if the other parser fails at the same position.

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

        name = _label_of(self)
        shown = _brief(f"{name} (but not {_label_of(other) or '...'})") if name else "but_not(...)"

        @_parser(shown, leaf=True, first=self._first)
        def _but_not(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingFailure):
                return res
            res_not = other.run(tokens, res.state._reset_pos_to(s))
            if isinstance(res_not, ParsingFailure):
                return res_not if res_not.committed else res
            return ParsingFailure(s.expect(_but_not))

        _but_not._form = ("but_not", self, other)
        return _but_not

    def when(self, pred: Callable[[B], bool]) -> Parser[A, B]:
        """Wrap the parser to a new one, that parses a token if it satisfies the predicate `pred`.

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

        @_parser(_label_of(self) or "when(...)", leaf=True, first=self._first)
        def _when(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingFailure) or pred(res.value):
                return res
            return ParsingFailure(s.expect(_when))

        _when._form = ("when", self, pred)
        return _when

    def context(self, label: str) -> Parser[A, B]:
        """Name what this parser parses, for the errors inside it: `... in label`.

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

        @_parser(self.name, shape=self._shape, first=self._first)
        def _context(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            inner = State(s.pos, s.max, s.expected, (*s.context, label), s.problems)
            res = self.run(tokens, inner)
            st = res.state
            out = State(st.pos, st.max, st.expected, s.context, st.problems)
            if isinstance(res, ParsingFailure):
                return ParsingFailure(out, res.message, res.committed)
            return ParsingSuccess(res.value, out)

        _context._form = ("context", self)
        return _context

    def check(self, pred: Callable[[B], bool], message: str) -> Parser[A, B]:
        """Record an error, `message`, when `pred` is false for the parsed value, and go on.

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

        @_parser(self.name, shape=self._shape, first=self._first)
        def _check(tokens: Sequence[A], s: State) -> ParsingResult[B]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingFailure) or pred(res.value):
                return res
            st = res.state
            problem = CheckFailure(message, s.pos, st.pos, s.context)
            checked = State(st.pos, st.max, st.expected, st.context, (*st.problems, problem))
            return ParsingSuccess(res.value, checked)

        _check._form = ("check", self, pred)
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

        @_parser(f"({self.name}, {other.name})", shape=True, first=self._first)
        def _add(tokens: Sequence[A], s: State) -> ParsingResult[Any]:
            res_l = self.run(tokens, s)
            if res_l.__class__ is ParsingFailure:
                return res_l
            res_r = other.run(tokens, res_l.state)
            if res_r.__class__ is ParsingFailure:
                return res_r
            # ParsingSuccess.__add__(), inlined
            v1: Any = res_l.value
            v2: Any = res_r.value
            if v1.__class__ is IgnoredValue:
                return res_r
            if v2.__class__ is IgnoredValue:
                return ParsingSuccess(v1, res_r.state)
            if v1.__class__ is _Tuple:
                return ParsingSuccess(_Tuple((*v1, v2)), res_r.state)
            return ParsingSuccess(_Tuple((v1, v2)), res_r.state)

        _add._form = ("seq", self, other)
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

        def tried(tokens: Sequence[A], s: State) -> ParsingResult[B | C]:
            res = self.run(tokens, s)
            if not isinstance(res, ParsingFailure) or res.committed:
                return res
            return other.run(tokens, res.state._reset_pos_to(s))

        first = None if self._first is None or other._first is None else self._first | other._first
        if debug or not _narrows(_alternatives(self, other)):
            p = _parser(f"{self.name} or {other.name}", shape=True, first=first)(tried)
        else:
            # Built on the first run: of `p1 | p2 | p3`, only the outermost parser ever runs.
            alts: tuple[Parser[Any, Any], ...] = ()
            table: _Dispatch | None = None

            def dispatched(tokens: Sequence[A], s: State) -> ParsingResult[B | C]:
                nonlocal alts, table
                if table is None:
                    # A parser may be named after this one is built.
                    alts = _alternatives(self, other)
                    table = _dispatch_table(alts)
                pos = s.pos
                if pos < len(tokens):
                    t = tokens[pos]
                    by_key, unknown = table
                    try:
                        candidates, rest = by_key.get(t if t.__class__ is str else _key(t), unknown)
                    except TypeError:  # an unhashable key
                        return tried(tokens, s)
                    st = s
                    failures: list[ParsingResult[Any]] = []
                    for alt in candidates:
                        res = alt.run(tokens, st)
                        if res.__class__ is not ParsingFailure or res.committed:
                            # What the alternatives skipped expected at `pos` is forgotten once
                            # parsing got past it.
                            if res.state.max > pos:
                                return res
                            break
                        failures.append(res)
                        st = res.state._reset_pos_to(s)
                    else:
                        if candidates:
                            # A candidate is never run again: that would double the work at each
                            # level of a nested grammar.
                            if st.max == pos:
                                return _failed_in_order(tokens, s, alts, candidates, failures)
                            last = failures[-1]
                            for alt in rest:
                                last = alt.run(tokens, st)
                                st = last.state._reset_pos_to(s)
                            return last
                return tried(tokens, s)

            p = _parser(f"{self.name} or {other.name}", shape=True, first=first)(dispatched)
        p._form = ("alt", self, other)
        return p

    def map[C](self, f: Callable[[B], C]) -> Parser[A, C]:
        """Transform the parsing result by applying the specified function; the same as `>>`."""
        if self._test is not None and not debug:
            return self._fused(f)

        @_parser(self.name, shape=self._shape, first=self._first)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.map(f)

        _map._form = ("map", self, f)
        return _map

    def starmap[C](self, f: Callable[..., C]) -> Parser[A, C]:
        """Transform the parsing results (tuple) by applying the specified function."""

        @_parser(self.name, shape=self._shape, first=self._first)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.starmap(f)

        _map._form = ("starmap", self, f)
        return _map

    def __rshift__[C](self, f: Callable[[B], C]) -> Parser[A, C]:
        """Transform the parsing result by applying the specified function.

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

        Also known as `>>=` in Haskell.

        !!! Note

            You can parse any context-free grammar without resorting to `bind`. Due
            to its poor performance please use it only when you really need it.
        """

        @_parser(f"({self.name} >>=)", shape=True)
        def _bind(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            return res.bind(lambda v, s2: f(v).run(tokens, s2))

        return _bind

    def __neg__(self) -> IgnoredParser[A]:
        """Return a parser that parses the same tokens, but its parsing result is ignored by the
        sequential `+` combinator.

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

        if self._test is not None and not debug:
            return cast("IgnoredParser[A]", self._fused(_ignore))

        @_parser(self.name, shape=self._shape, first=self._first)
        def _ignored(tokens: Sequence[A], s: State) -> ParsingResult[IgnoredValue]:
            res = self.run(tokens, s)
            if isinstance(res, ParsingFailure):
                return res
            return ParsingSuccess(_IGNORED, res.state)

        _ignored._form = ("ignore", self)
        return cast("IgnoredParser[A]", _ignored)

    def __invert__(self) -> Parser[A, A]:
        """Return a parser that parses one token, any token this parser does not parse.

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

        name = _label_of(self)

        @_parser(f"anything but {name}" if name else "anything else", leaf=True)
        def _invert(tokens: Sequence[A], s: State) -> ParsingResult[A]:
            if s.pos >= len(tokens):
                s2 = s.expect(_invert)
                return ParsingFailure(s2, "got unexpected end of input")
            res = self.run(tokens, s)
            if isinstance(res, ParsingFailure):
                if res.committed:
                    return res
                return ParsingSuccess(tokens[s.pos], s.reset_pos(s.pos + 1))
            return ParsingFailure(s.expect(_invert))

        return _invert


@final
class CompiledParser[A, B]:
    """A grammar compiled to Python functions, see `Parser.compile()`.

    Users get it from `Parser.compile()`; its constructor is internal.
    """

    __slots__ = ("_fast", "_fast_text", "_parser", "_reads_tokens", "_tokenizer")

    def __init__(
        self,
        parser: Parser[A, B],
        fast: Callable[[Sequence[A]], tuple[B, int] | None] | None,
        tokenizer: Tokenizer | None = None,
        fast_text: Callable[[str], tuple[B, int] | None] | None = None,
        *,
        reads_tokens: bool = True,
    ) -> None:
        """`parser`, with its generated functions, `None` for a grammar that stays interpreted."""
        self._parser = parser
        self._fast = fast
        self._tokenizer = tokenizer
        self._fast_text = fast_text
        self._reads_tokens = reads_tokens

    def parse(
        self, tokens: Sequence[A], *, describe: Callable[[A], TokenDescription] | None = None
    ) -> B:
        """Parse the sequence of tokens and return the parsed value, as `Parser.parse()` does.

        It raises what `Parser.parse()` raises. With `funcparserlib.parser.debug` set, the parser
        runs interpreted and writes its parsing log.
        """
        res = None if self._fast is None or debug else self._fast(tokens)
        return self._parser.parse(tokens, describe=describe) if res is None else res[0]

    def parse_text(
        self, text: str, *, describe: Callable[[A], TokenDescription] | None = None
    ) -> B:
        """Parse the text, with the tokenizer given to `Parser.compile()` if the grammar needs one.

        It gives what `Parser.parse()` gives for the tokens of `text`. Where the tokenizer stops at
        text no token spec matches, the error is a mistake among the tokens before it, or what the
        grammar expected in place of that text. A grammar that reads tokens only through `tok()`,
        with a tokenizer whose specs join into one regexp, runs on the text without making a
        `Token`: for any other, this is `CompiledParser.parse()` of the tokens.

        Raises `NoParseError` and `ParseErrors` for text the grammar does not cover, `LexerError`
        for text past where a grammar without `finished` stops, and `ValueError` when a grammar of
        tokens has no tokenizer. `describe` is for tokens, so a grammar of text does not use it.
        """
        res = None
        if self._fast_text is not None and not debug:
            try:
                res = self._fast_text(text)
            except _Undecided:
                res = None
        return res[0] if res is not None else self._interpreted(text, describe)

    def parse_bytes(self, data: bytes) -> B:
        """Parse the bytes, as `Parser.parse_bytes()` does."""
        return self.parse_text(cast("Any", data))

    def _interpreted(self, text: str, describe: Callable[[A], TokenDescription] | None) -> B:
        """The result of `text` without the functions generated for text."""
        if self._tokenizer is None:
            if self._reads_tokens:
                raise with_hint(
                    ValueError("a compiled parser of tokens without a tokenizer cannot parse text"),
                    "Pass the tokenizer of the grammar to `Parser.compile()`:"
                    " `grammar.compile(tokenize)`.",
                )
            if isinstance(text, bytes):
                return self._parser.parse_bytes(text)
            return self._parser.parse_text(text)
        tokens: list[Any] = []
        try:
            tokens.extend(self._tokenizer(text))
        except LexerError as e:
            return self._explained(tokens, e, describe)
        if self._fast_text is None and not debug:
            return self.parse(tokens, describe=describe)
        return self._parser.parse(tokens, describe=describe)

    def _explained(
        self, tokens: list[Any], error: LexerError, describe: Callable[[A], TokenDescription] | None
    ) -> B:
        """Raise the error of `tokens`, then one more for the text the tokenizer stopped at.

        No grammar reads that token, so parsing fails there at the latest, saying what it expected.
        `error` is what is left when the grammar stops before it.
        """
        word = _WORD.match(error.msg, error.place[1] - 1)
        token = Token(_UNREADABLE, word.group() if word else "\r", error.place, error.place)
        try:
            self._parser.parse(cast("Sequence[A]", [*tokens, token]), describe=describe)
        except (NoParseError, ParseErrors) as e:
            raise e from error
        raise error

    def try_parse(self, text: str | bytes) -> B | None:
        """The value of `text`, or `None` where it does not parse.

        It is for text that is often invalid, when the reason does not matter: a parse that fails
        stops there, which can be several times faster than making its error. A failed `check()` or
        `commit()` leaves the generated code, so it takes the time of `parse_text()`. A grammar
        whose value can be `None` cannot tell that from a failure this way. Text nested too deeply
        gives `None` too, as a `RecursionError`. What a function after `>>` raises it raises.
        """
        if self._fast_text is not None and not debug:
            try:
                res = self._fast_text(cast("str", text))
            except _Undecided:
                pass
            else:
                return None if res is None else res[0]
        try:
            return self._interpreted(cast("str", text), None)
        except (NoParseError, ParseErrors, LexerError, RecursionError):
            return None


def _merged(st: State, s: State) -> State:
    """`st`, reached from a fresh furthest position, with what failed furthest of it and `s`."""
    if st.max > s.max:
        max_, expected = st.max, st.expected
    elif st.max == s.max:
        max_, expected = s.max, (*s.expected, *st.expected)
    else:
        max_, expected = s.max, s.expected
    return State(st.pos, max_, expected, st.context, st.problems)


def _moved(s: State, offset: int) -> State:
    """`s` of a region that starts at `offset`, in the input around it."""
    problems = tuple(
        CheckFailure(p.message, p.start + offset, p.end + offset, p.context, p.unexpected)
        for p in s.problems
    )
    return State(s.pos + offset, s.max + offset, s.expected, s.context, problems)


def _read_region(
    p: Parser[Any, Any], body: Parser[Any, Any], region: Sequence[Any], start: int, at: State
) -> ParsingResult[Any]:
    """What `body` gives for `region`, which starts at `start`, in the state of the input around it.

    `at` is the state at `start`. What `body` expected at the end of its region is no part of what
    comes after a success: nothing can follow there.
    """
    inner = body.run(region, State(0, 0, (), at.context, ()))
    st = _moved(inner.state, start)
    if inner.__class__ is not ParsingFailure and inner.state.max >= len(region):
        st = State(st.pos, st.max, (), st.context, st.problems)
    merged = _merged(st, at)
    moved = State(merged.pos, merged.max, merged.expected, at.context, (*at.problems, *st.problems))
    if inner.__class__ is ParsingFailure:
        return inner._replace_state(moved)
    if inner.state.pos != len(region):
        return ParsingFailure(moved, "bytes left in the region")
    return ParsingSuccess(inner.value, moved)


@overload
def prefixed(length: Parser[int, int]) -> Parser[int, bytes]: ...


@overload
def prefixed(length: Parser[str, int]) -> Parser[str, str]: ...


@overload
def prefixed[A](length: Parser[A, int]) -> Parser[A, Sequence[A]]: ...


@overload
def prefixed[A, B](length: Parser[A, int], body: Parser[A, B]) -> Parser[A, B]: ...


def prefixed(length: Parser[Any, int], body: Parser[Any, Any] | None = None) -> Parser[Any, Any]:
    """Return a parser of as many tokens as `length` parses, as they are or through `body`.

    It is the length field of a binary format: the bytes that follow, or with `body`, what `body`
    parses of them. `body` sees those bytes as its whole input, and has to read to their end. A
    length that is negative or runs past the input fails where those bytes start. What `skip` of a
    `funcparserlib.text.Text` matches before `body` is not read for it.

    Examples:
        ```pycon
        >>> from funcparserlib.binary import Binary
        >>> b = Binary()
        >>> prefixed(b.byte).parse_bytes(b"\\x02hi")
        b'hi'
        >>> prefixed(b.byte, many(b.byte)).parse_bytes(b"\\x02hi")
        [104, 105]
        >>> prefixed(b.byte).parse_bytes(b"\\x09hi")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: length 9 runs past the end: 0x68 at offset 0x1

        ```
    """

    @_parser(f"tokens counted by {length.name}", shape=True)
    def _prefixed(tokens: Sequence[Any], s: State) -> ParsingResult[Any]:
        res = length.run(tokens, s)
        if res.__class__ is ParsingFailure:
            return res
        start = res.state.pos
        end = start + res.value
        if res.value < 0 or end > len(tokens):
            return ParsingFailure(res.state, f"length {res.value} runs past the end")
        region = tokens[start:end]
        if body is None:
            return ParsingSuccess(region, res.state.reset_pos(end))
        return _read_region(_prefixed, body, region, start, res.state)

    _prefixed._form = ("prefixed", length, body)
    return _prefixed


def counted[A, B](count: Parser[A, int], item: Parser[A, B]) -> Parser[A, list[B]]:
    """Return a parser of as many of `item` as `count` parses, in a list.

    A negative count fails. Raises `ValueError` when `item` reads no tokens, as `many()` does.

    Examples:
        ```pycon
        >>> from funcparserlib.binary import Binary
        >>> b = Binary()
        >>> counted(b.byte, b.fixed(">H")).parse_bytes(b"\\x02\\x00\\x01\\x00\\x02")
        [1, 2]

        ```
    """

    @_parser(f"items counted by {count.name}", shape=True)
    def _counted(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        res = count.run(tokens, s)
        if res.__class__ is ParsingFailure:
            return cast("ParsingResult[list[B]]", res)
        if res.value < 0:
            return ParsingFailure(res.state, f"count {res.value} is negative")
        items: list[B] = []
        st = res.state
        for _ in range(res.value):
            read = item.run(tokens, st)
            if read.__class__ is ParsingFailure:
                return cast("ParsingResult[list[B]]", read)
            if read.state.pos == st.pos:
                raise with_hint(
                    ValueError(f"{item.name} matched no tokens inside counted()"),
                    "A count bounds how many items are read, and an item that reads nothing"
                    " repeats for nothing: count a parser that reads a token.",
                )
            items.append(read.value)
            st = read.state
        return ParsingSuccess(items, st)

    _counted._form = ("counted", count, item)
    return _counted


def switch[A](key: Parser[A, Any], cases: Mapping[Any, Parser[A, Any]]) -> Parser[A, Any]:
    """Return a parser that reads `key`, then runs the parser `cases` holds for its value.

    A key of `cases` is a value, or a `range` of values. For a value `cases` has no parser for, it
    fails at the start of `key`, as a token that does not match does, so `commit()` does not count
    `key` as read. Unlike `p1 | p2`, no other case is tried.

    Examples:
        ```pycon
        >>> from funcparserlib.binary import Binary
        >>> b = Binary()
        >>> field = switch(b.byte, {1: b.fixed(">H"), range(2, 4): b.byte})
        >>> field.parse_bytes(b"\\x01\\x00\\x07")
        7
        >>> field.parse_bytes(b"\\x03\\x07")
        7

        ```
    """
    table: dict[Any, Parser[A, Any]] = {}
    for case, p in cases.items():
        for value in case if isinstance(case, range) else (case,):
            table[value] = p

    @_parser(f"one of the cases of {key.name}", shape=True)
    def _switch(tokens: Sequence[A], s: State) -> ParsingResult[Any]:
        res = key.run(tokens, s)
        if res.__class__ is ParsingFailure:
            return res
        p = table.get(res.value)
        if p is None:
            at = State(s.pos, s.max, s.expected, s.context, res.state.problems)
            return ParsingFailure(at.expect(_switch))
        return p.run(tokens, res.state)

    _switch._form = ("switch", key, table)
    return _switch


@overload
def longest[A, B, C](p1: Parser[A, B], p2: Parser[A, C], /) -> Parser[A, B | C]: ...


@overload
def longest[A, B, C, D](
    p1: Parser[A, B], p2: Parser[A, C], p3: Parser[A, D], /
) -> Parser[A, B | C | D]: ...


@overload
def longest[A, B, C, D, E](
    p1: Parser[A, B], p2: Parser[A, C], p3: Parser[A, D], p4: Parser[A, E], /
) -> Parser[A, B | C | D | E]: ...


@overload
def longest[A](*parsers: Parser[A, Any]) -> Parser[A, Any]: ...


def longest[A](*parsers: Parser[A, Any]) -> Parser[A, Any]:
    """Return a parser of the longest match of `parsers`, the first of them when several tie.

    `p1 | p2` keeps the first parser that succeeds, so a shorter match can hide a longer one, and
    the order of the alternatives matters. `longest(p1, p2)` runs them all, so it has no order to
    get right, and the functions after `>>` of each run.

    Raises `ValueError` for fewer than two parsers.

    Examples:
        ```pycon
        >>> (a("x") | a("x") + a("y")).parse("xy")
        'x'
        >>> longest(a("x"), a("x") + a("y")).parse("xy")
        ('x', 'y')

        ```
    """
    if len(parsers) < 2:
        raise with_hint(
            ValueError("longest() takes at least two parsers"),
            "One parser is its own longest match: use it as it is.",
        )
    firsts = [p._first for p in parsers]
    first = None if None in firsts else frozenset().union(*cast("list[frozenset[Any]]", firsts))

    @_parser(" or ".join(str(p.name) for p in parsers), shape=True, first=first)
    def _longest(tokens: Sequence[A], s: State) -> ParsingResult[Any]:
        best: ParsingResult[Any] | None = None
        last: ParsingResult[Any] | None = None
        st = s
        for p in parsers:
            res = p.run(tokens, st)
            if res.__class__ is ParsingFailure:
                if res.committed:
                    return res
                last = res
            elif best is None or res.state.pos > best.state.pos:
                best = res
            st = res.state._reset_pos_to(s)
        if best is None:
            return cast("ParsingResult[Any]", last)
        return ParsingSuccess(best.value, st._reset_pos_to(best.state))

    _longest._form = ("longest", parsers)
    return _longest


def parser[A, B](name: str | None) -> Callable[[ParserFn[A, B]], Parser[A, B]]:
    """Make a named `Parser` from a parsing function, as a decorator.

    The decorated function takes the tokens and the current `State` and returns a `ParsingSuccess`
    or a `ParsingFailure`; unlike `Parser(f)`, it does not adapt a 1.0 parsing function. A success
    carries the state moved past what it parsed, `s.reset_pos(pos)`; a failure carries
    `s.expect(p)`, where `p` is the parser that failed, so that the error names it.

    A function that tries another parser after a failure, whose state is `st`, goes on from
    `dataclasses.replace(s, max=st.max, expected=st.expected)`, as `|` does: the error still names
    what failed furthest, and what the failed parser recorded is gone. `st.reset_pos(s.pos)` keeps
    the `Parser.check()` errors of the failed parser, which reject input `|` accepts.

    Examples:
        ```pycon
        >>> @parser("digit")
        ... def digit(tokens, s):
        ...     if s.pos < len(tokens) and tokens[s.pos].isdigit():
        ...         return ParsingSuccess(int(tokens[s.pos]), s.reset_pos(s.pos + 1))
        ...     return ParsingFailure(s.expect(digit))
        >>> many(digit).parse("12")
        [1, 2]
        >>> (digit + digit).parse("1x")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: 'x', expected: digit

        ```

        ```pycon
        >>> from dataclasses import replace
        >>> def either(p, q):
        ...     @parser(None)
        ...     def tried(tokens, s):
        ...         res = p.run(tokens, s)
        ...         if not isinstance(res, ParsingFailure) or res.committed:
        ...             return res
        ...         st = res.state
        ...         return q.run(tokens, replace(s, max=st.max, expected=st.expected))
        ...     return tried
        >>> pair = either(a("x") + a("y"), a("z"))
        >>> pair.parse("z")
        'z'
        >>> pair.parse("xw")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected token: 'w', expected: 'y'

        ```
    """

    def decorator(f: ParserFn[A, B]) -> Parser[A, B]:
        return _built(Parser, f).named(name)

    return decorator


_MAX_NAME = 200
"""The length of a composed name: a parser used twice doubles it at each level."""


def _brief(name: str) -> str:
    return name if len(name) <= _MAX_NAME else f"{name[: _MAX_NAME - 3]}..."


def _parser[A, B](
    name: str | None,
    *,
    leaf: bool = False,
    shape: bool = False,
    first: frozenset[object] | None = None,
) -> Callable[[ParserFn[A, B]], Parser[A, B]]:
    """`parser()` for the parsers the library builds: a name without a label.

    A `leaf` parser names itself among the expected ones when it fails, so a label given to it later
    changes only its name. A `shape` name spells out how the parser is built, `('x', 'y')`, rather
    than what it parses.
    """

    def decorator(f: ParserFn[A, B]) -> Parser[A, B]:
        p: Parser[A, B] = _built(Parser, f)._rename(_brief(name) if shape and name else name)
        p._leaf = leaf
        p._shape = shape
        p._first = first
        return p

    return decorator


def _label_of(p: Parser[Any, Any]) -> str | None:
    """`p.name` when it says what `p` parses, `None` when it spells out how `p` is built."""
    return None if p._shape else p.name


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
        " return a ParsingSuccess or a ParsingFailure"
    )
    code = getattr(f, "__code__", None)
    if code is None:
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    else:
        module = getattr(f, "__module__", None)
        # The function's module registry, as `warnings.warn()` keeps: a warning shown once is not
        # repeated for every grammar built from the function.
        registry = getattr(f, "__globals__", {}).setdefault("__warningregistry__", {})
        warnings.warn_explicit(
            msg,
            DeprecationWarning,
            code.co_filename,
            code.co_firstlineno,
            module=module,
            registry=registry,
        )


def _parse_error[A](
    res: ParsingFailure, tokens: Sequence[A], describe: Callable[[A], TokenDescription]
) -> NoParseError:
    """The `NoParseError` of a failed parse: what was found at the furthest position, and why."""
    st = res.state
    names = tuple(dict.fromkeys(p.name for p, _ in st.expected if p.name is not None))
    contexts = [c for _, c in st.expected] or [st.context]
    common = contexts[0][: min(map(len, contexts))]
    while any(c[: len(common)] != common for c in contexts):
        common = common[:-1]
    context = tuple(reversed(common))
    token: A | None = None
    description: TokenDescription | None
    if st.max >= len(tokens) and isinstance(tokens, _Offsets):
        description = None
        found = f"got unexpected end of input at offset 0x{len(tokens):x}"
    elif st.max >= len(tokens) and isinstance(tokens, _Located):
        description = tokens.end()
        found = f"{_where(description)}got unexpected end of input"
    elif st.max < len(tokens):
        token = tokens[st.max]
        shown = describe(tokens[st.max])
        description = shown
        if isinstance(token, _Character):
            token = cast("A", token.value)
        found = f"{_where(shown)}{res.message}: {shown.text}"
    else:
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
    token: A | None = None
    description: TokenDescription | None
    if isinstance(tokens, _Offsets):
        description = None
        message = f"offset 0x{problem.start:x}: {problem.message}"
        problem = CheckFailure(message, problem.start, problem.end, problem.context)
    elif isinstance(tokens, _Located):
        in_text = problem.start < len(tokens)
        description = tokens.span(problem.start, problem.end) if in_text else tokens.end()
    elif problem.start < len(tokens):
        token = tokens[problem.start]
        first = describe(tokens[problem.start])
        last = describe(tokens[max(problem.start, problem.end - 1)])
        description = TokenDescription(first.text, first.start, last.end)
        if problem.unexpected:
            message = f"got unexpected token: {first.text}{problem.message}"
            problem = CheckFailure(message, problem.start, problem.end, problem.context)
    else:
        description = _after(tokens, describe)
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
        """The sequence of this parser and `other`, as `Parser.__add__()` builds it."""
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2], C]) -> Parser[A, C]:
        """A parser of `f` applied to the elements of the tuple this parser returns."""
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2]], C]) -> Parser[A, C]:
        """A parser of `f` applied to the tuple this parser returns, as `Parser.__rshift__()`."""
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
        """The sequence of this parser and `other`, as `Parser.__add__()` builds it."""
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3], C]) -> Parser[A, C]:
        """A parser of `f` applied to the elements of the tuple this parser returns."""
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3]], C]) -> Parser[A, C]:
        """A parser of `f` applied to the tuple this parser returns, as `Parser.__rshift__()`."""
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
        """The sequence of this parser and `other`, as `Parser.__add__()` builds it."""
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3, T4], C]) -> Parser[A, C]:
        """A parser of `f` applied to the elements of the tuple this parser returns."""
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3, T4]], C]) -> Parser[A, C]:
        """A parser of `f` applied to the tuple this parser returns, as `Parser.__rshift__()`."""
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
        """The sequence of this parser and `other`, as `Parser.__add__()` builds it."""
        raise NotImplementedError

    def starmap[C](self, f: Callable[[T1, T2, T3, T4, T5], C]) -> Parser[A, C]:
        """A parser of `f` applied to the elements of the tuple this parser returns."""
        raise NotImplementedError

    # PyCharm (2024.1) doesn't properly infer from the base method, so
    # open B explicitly here
    def __rshift__[C](self, f: Callable[[tuple[T1, T2, T3, T4, T5]], C]) -> Parser[A, C]:
        """A parser of `f` applied to the tuple this parser returns, as `Parser.__rshift__()`."""
        raise NotImplementedError


@final
@valueclass
class IgnoredValue:
    """Constant to indicate an ignored parsed value."""


_IGNORED = IgnoredValue()


def _ignore(_: object) -> IgnoredValue:
    return _IGNORED


@final
@valueclass
class _TokenTest:
    """A parser that tests one token: `pred` on it, then `conv` of it as the value.

    `leaf` is the `some()` parser it was built from and `chain` the parsers between, outermost
    first; when the test fails, the error names the outermost labelled one, or `leaf`, as the
    parsers would without fusing.
    """

    leaf: Parser[Any, Any]
    chain: tuple[Parser[Any, Any], ...]
    pred: Callable[[Any], bool]
    conv: Callable[[Any], Any] | None
    # The type and value `tok()` tests, and the functions `conv` applies, in order.
    token: tuple[str, str | None] | None = None
    convs: tuple[Callable[[Any], Any], ...] = ()


_token_value = attrgetter("value")


def _key(t: object) -> object:
    """What `|` looks a token up by: its `type`, or the token itself when it has none."""
    return getattr(t, "type", t)


type _Alternatives = tuple[tuple[Parser[Any, Any], ...], tuple[Parser[Any, Any], ...]]
type _Dispatch = tuple[dict[object, _Alternatives], _Alternatives]


def _alternatives(*parsers: Parser[Any, Any]) -> tuple[Parser[Any, Any], ...]:
    """What `|` chooses among in `parsers`, each parser once: a labelled parser is one alternative,
    as its label is what an error names.
    """
    alts: list[Parser[Any, Any]] = []
    seen: set[int] = set()
    stack = list(reversed(parsers))
    while stack:
        q = stack.pop()
        if id(q) in seen:
            continue
        seen.add(id(q))
        if q._form is None or q._form[0] != "alt" or q._label:
            alts.append(q)
        else:
            stack += reversed(q._form[1:])
    return tuple(alts)


def _narrows(alts: tuple[Parser[Any, Any], ...]) -> bool:
    """Whether looking at the next token would narrow the choice among `alts`."""
    return len(alts) >= 3 and sum(alt._first is not None for alt in alts) >= 2


def _dispatch_table(alts: tuple[Parser[Any, Any], ...]) -> _Dispatch:
    """For each token key, the alternatives that can start with it, in order, and the ones after the
    last of them; then the same for a key no alternative names.
    """

    def split(indices: list[int]) -> _Alternatives:
        return tuple(alts[i] for i in indices), alts[indices[-1] + 1 :] if indices else alts

    keys: set[object] = set().union(*(alt._first for alt in alts if alt._first is not None))
    by_key = {
        key: split([i for i, alt in enumerate(alts) if alt._first is None or key in alt._first])
        for key in keys
    }
    return by_key, split([i for i, alt in enumerate(alts) if alt._first is None])


def _failed_in_order(
    tokens: Sequence[Any],
    s: State,
    alts: tuple[Parser[Any, Any], ...],
    candidates: tuple[Parser[Any, Any], ...],
    failures: list[ParsingResult[Any]],
) -> ParsingResult[Any]:
    """The failure of `alts` at `s`, where every one of `candidates` failed reading no token.

    `failures` holds what each candidate gave, run after the one before it, so each added its
    expected parsers to those of the last. The other alternatives run here, between them.
    """
    st, res = s, failures[-1]
    done = len(s.expected)
    candidate = 0
    for alt in alts:
        if candidate < len(candidates) and alt is candidates[candidate]:
            res = failures[candidate]
            added = res.state.expected[done:]
            done += len(added)
            candidate += 1
            st = State(st.pos, st.max, (*st.expected, *added), s.context, s.problems)
        else:
            res = alt.run(tokens, st)
            st = res.state._reset_pos_to(s)
    return ParsingFailure(st, cast("ParsingFailure", res).message)


def _token_test(test: _TokenTest) -> Parser[Any, Any]:
    """One parser running the token test, `>>` and `-p` would run in several."""
    pred, conv, chain, leaf = test.pred, test.conv, test.chain, test.leaf

    def expected() -> Parser[Any, Any]:
        for q in chain:
            if q._label:
                return q
        return leaf

    def run(tokens: Sequence[Any], s: State) -> ParsingResult[Any]:
        pos = s.pos
        if pos >= len(tokens):
            return ParsingFailure(s.expect(expected()), "got unexpected end of input")
        t = tokens[pos]
        if pred(t):
            return ParsingSuccess(t if conv is None else conv(t), s.reset_pos(pos + 1))
        return ParsingFailure(s.expect(expected()))

    p = _built(Parser, run)
    p._test = test
    return p


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

    @_parser(f"{{ {p.name} }}", shape=True)
    def _many(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        acc = []
        while True:
            res = p.run(tokens, s)
            if not isinstance(res, ParsingFailure):
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
                s = res.state._reset_pos_to(s)
                break

        if debug:
            _log.debug("*matched* %d instances of %s, new state = %s", len(acc), _many.name, s)
        return ParsingSuccess(acc, s)

    _many._form = ("many", p)
    return _many


def some[A](pred: Callable[[A], bool]) -> Parser[A, A]:
    """Return a parser that parses a token if it satisfies the predicate `pred`.

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
            return ParsingFailure(s2, "got unexpected end of input")

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
        return ParsingFailure(s2)

    _some._test = _TokenTest(_some, (), pred, None)
    return _some


def a[A](value: A) -> Parser[A, A]:
    """Return a parser that parses a token if it's equal to `value`.

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
    # An object with a `name`, such as a `Token` or an enum member, is shown in errors by it.
    name = getattr(value, "name", value)

    def eq_value(t: A) -> bool:
        return t == value

    p = some(eq_value).named(repr(name))
    # The key every token equal to `value` has, for the types whose equality is known to agree.
    if isinstance(value, str):
        p._first = frozenset({value})
    elif isinstance(value, Token):
        p._first = frozenset({value.type})
    return p


# noinspection PyShadowingBuiltins
def tok(type: str, value: str | None = None) -> Parser[Token, str]:  # noqa: A002 — public keyword
    """Return a parser that parses a `Token` and returns the string value of the token.

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

    def eq_token(t: object) -> bool:
        # What `t == Token(type, value)` compares, without building a token.
        return isinstance(t, Token) and t.type == type and t.value == value

    eq = eq_type if value is None else cast("Callable[[Token], bool]", eq_token)
    p = some(eq).named(type if value is None else repr(value))
    p._first = frozenset({type})
    p._test = _TokenTest(p, (), eq, None, (type, value))
    return (p >> cast("Callable[[Token], str]", _token_value))._rename(p.name)


def pure[A](x: A) -> Parser[Any, A]:
    """Wrap any object into a parser.

    A pure parser doesn't touch the tokens sequence, it just returns its pure `x` value.

    Also known as `return` in Haskell.
    """

    @_parser(f"(pure {x!r})", shape=True)
    def _pure(_: Sequence[Any], s: State) -> ParsingResult[A]:
        return ParsingSuccess(x, s)

    _pure._form = ("pure", x)
    return _pure


def commit[A, B](p: Parser[A, B]) -> Parser[A, B]:
    """Return a parser that parses as `p`, and makes a failure of `p` final once it read a token.

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

    @_parser(p.name, shape=p._shape, first=p._first)
    def _commit(tokens: Sequence[A], s: State) -> ParsingResult[B]:
        # A fresh furthest position tells whether `p` read a token.
        res = p.run(tokens, State(s.pos, s.pos, (), s.context, s.problems))
        st = res.state
        merged = _merged(st, s)
        if isinstance(res, ParsingFailure):
            return ParsingFailure(merged, res.message, res.committed or st.max > s.pos)
        return ParsingSuccess(res.value, merged)

    _commit._form = ("commit", p)
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
    return (p | pure(None))._rename(_brief(f"[ {p.name} ]"))


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
        """The sequence of this parser and `other`, as `Parser.__add__()` builds it."""
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

    @_parser(f"({p.name}, {{ {p.name} }})", shape=True)
    def _oneplus(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        first = p.run(tokens, s)
        if isinstance(first, ParsingFailure):
            return first
        rest = many_p.run(tokens, first.state)
        if isinstance(rest, ParsingFailure):
            return rest
        return ParsingSuccess([first.value, *rest.value], rest.state)

    _oneplus._form = ("oneplus", p)
    return _oneplus


@final
class IgnoredEndOfInput(Parser[Any, IgnoredValue]):
    """The type of `-finished`; a type for checkers, never built.

    `p + -finished` has the type of `p`.
    """

    @overload  # type: ignore[override]
    def __add__[T, C](self, other: ForwardDeclParser[T, C]) -> Parser[T, C]: ...

    @overload
    def __add__[P: Parser[Any, Any]](self, other: P) -> P: ...

    def __add__(self, other):
        """The sequence of this parser and `other`, as `Parser.__add__()` builds it."""
        raise NotImplementedError


@final
class EndOfInput(Parser[Any, None]):
    """The type of `finished`: a parser of any tokens that succeeds only at the end of input."""

    # A narrower `-` than Parser's: what makes `p + -finished` typed.
    def __neg__(self) -> IgnoredEndOfInput:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        """The same parser with its result ignored, as `Parser.__neg__()` does."""
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
        return ParsingFailure(s.expect(p))

    p = _built(EndOfInput, run)._rename("end of input")
    p._leaf = True
    p._form = ("finished",)
    return p


finished: EndOfInput = _end_of_input()
"""A parser that throws an exception if there are any unparsed tokens left in the sequence."""


@final
class ForwardDeclParser[A, B](Parser[A, B]):
    """A parser defined after it is used, for recursive grammars; see `forward_decl()`."""

    def __init__(self) -> None:
        """Declare an undefined parser; `define()` gives it its meaning."""

        def f(_tokens: Any, _s: Any) -> Any:
            raise NotImplementedError("you must define() a forward_decl() somewhere")

        super().__init__(f)
        self._rename("forward_decl()")
        self._shape = True

    def define(self, p: ParserObjOrFn[A, B]) -> None:
        """Define the parser this forward declaration stands for.

        Raises `ValueError` when `p` is this forward declaration itself.
        """
        if p is self:
            raise with_hint(
                ValueError("a forward_decl() is defined as itself"),
                "Define it as the parser it stands for, which may use it inside:"
                " `expr.define(a('x') + maybe(expr))`.",
            )
        self._define(p)

    @overload  # type: ignore[override]
    def __add__(  # type: ignore[overload-overlap]
        self, other: IgnoredEndOfInput | IgnoredParser[A]
    ) -> Parser[A, B]: ...

    @overload
    def __add__[C](self, other: Parser[A, C]) -> Tuple2Parser[A, B, C]: ...

    def __add__(self, other: Parser[A, Any]) -> Parser[A, Any]:  # ty: ignore[invalid-method-override] -- a sum is no forward declaration
        """The sequence of this parser and `other`, as `Parser.__add__()` builds it."""
        return super().__add__(other)


def forward_decl() -> ForwardDeclParser[Any, Any]:
    """Return an undefined parser that can be used as a forward declaration.

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
