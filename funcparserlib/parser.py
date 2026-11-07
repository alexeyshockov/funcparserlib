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
    * `lit(value)`, `regex(pattern)` and `skipping(pattern)` read text, with no tokenizer
    * `tokenized(grammar, tokenizer)` reads text with the tokenizer of a grammar of tokens
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
    "separated",
    "skip",
    "finished",
    "forward_decl",
    "anything_but",
    "tok",
    "lit",
    "regex",
    "skipping",
    "Skipping",
    "Parser",
    "CompiledParser",
    "Located",
    "tokenized",
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
    "Invalid",
    "Expectation",
    "Problem",
    "debug",
]

import logging
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import field, replace
from operator import attrgetter
from os import PathLike
from pathlib import Path
from typing import Any, NoReturn, Protocol, Self, TypeGuard, cast, final, overload

from ._commons import valueclass, with_hint
from ._utils import line_of, place_of
from .lexer import LexerError, Place, Token, Tokenizer
from .util import Lines

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


_STOPPED = "stopped at a rejected value"
_WORD = re.compile(r"\S{1,20}")
_UNREADABLE = "text no token spec matches"


class Invalid(Exception):
    """Raised by a function given to `Parser.validated()`: the value it got is not valid.

    `message` is the error. With `keep`, the value is real and only breaks a rule, so parsing goes
    on with it: what comes after the parser still sees the value. Without it there is no value, and
    no function after `>>` runs on it.
    """

    message: str
    keep: bool

    def __init__(self, message: str, *, keep: bool = False) -> None:
        """Say why the value is not valid, and whether parsing goes on with it."""
        super().__init__(message)
        self.message = message
        self.keep = keep


class _NotAToken(AttributeError):
    """What `tok()` raises for an item without a type, as a character or a byte has none."""


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

    def __init__(self, data: bytes | bytearray) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: Any) -> Any:
        return _Byte(self._data[i], i)


class _TextPlaces(Sequence[Any]):
    """The characters of a text, for errors: each knows its line and column."""

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

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
        return place_of(self._text, offset)

    def end(self) -> TokenDescription:
        """The end of the text, past its last character that is not whitespace."""
        place = self._place(len(self._text.rstrip()))
        return TokenDescription("end of input", place, place)

    def span(self, start: int, end: int) -> TokenDescription:
        """The text from `start` to `end`, exclusive, without the whitespace at its end."""
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
class Problem:
    """An error parsing went on after: its message, the tokens it covers (`start` to `end`,
    exclusive) and the `Parser.context()` labels around it.

    With `unexpected`, the error starts by naming the token at `start` as `Parser.parse()` was told
    to describe it, and `message` is what comes after that. With `rejected`, the part has no value
    until a `Parser.recover()` around it gives one. `cause` is the exception behind the error.
    """

    message: str
    start: int
    end: int
    context: tuple[str, ...]
    unexpected: bool = False
    rejected: bool = False
    cause: Exception | None = None


@final
@valueclass(repr=False)
class State:
    """Parsing state: where parsing is, and what an error message needs.

    `pos` is the position in the sequence being parsed, and `max` the furthest position a parser
    reached. `expected` holds the parsers that failed at `max`, each with the `Parser.context()`
    labels around it; `context` holds the labels around `pos`; `problems` holds the errors parsing
    went on after, from `Parser.validated()` and `Parser.recover()`.
    """

    pos: int
    max: int
    expected: tuple[Expectation, ...] = ()
    context: tuple[str, ...] = ()
    problems: tuple[Problem, ...] = ()
    _skipped: int = field(default=0, compare=False)
    """How much of the text before `pos` the parser that read last skipped after its value."""

    def rejected_since(self, s: State) -> bool:
        """Whether a part read since the state `s` has no value, because it was rejected.

        A parsing function that reads the value of a `Parser.run()` asks this first: over a rejected
        part the value is `None`, and has to be returned rather than used.
        """
        if self.problems is s.problems:
            return False
        return any(p.rejected for p in self.problems[len(s.problems) :])

    @property
    def parser(self) -> Parser[Any, Any] | None:
        """The last parser that failed at `max`, if any."""
        return self.expected[-1][0] if self.expected else None

    def expect(self, p: Parser[Any, Any]) -> State:
        """This state with `p` among the parsers that failed, unless a parser got further."""
        if self.pos == self.max:
            expected = (*self.expected, (p, self.context))
            return State(self.pos, self.max, expected, self.context, self.problems, self._skipped)
        return self

    def _reset_pos_to(self, s: State) -> State:
        """`s`, with the furthest position and what failed there from this state."""
        # dataclasses.replace() hits performance, as it performs additional validation
        return State(s.pos, self.max, self.expected, s.context, s.problems, s._skipped)

    def reset_pos(self, pos: int) -> State:
        """This state at `pos`; reaching past `max` moves it, and forgets what failed there."""
        if pos > self.max:
            return State(pos, pos, (), self.context, self.problems)
        return State(pos, self.max, self.expected, self.context, self.problems)

    def _past_skip(self, end: int, pos: int) -> State:
        """This state at `pos`, for a value that ends at `end` and a skip from there to `pos`."""
        skipped = pos - end
        if pos > self.max:
            return State(pos, pos, (), self.context, self.problems, skipped)
        return State(pos, self.max, self.expected, self.context, self.problems, skipped)

    def _value_end(self, start: int) -> int:
        """Where the value read from `start` to `pos` ends, before the skip after it."""
        return max(start, self.pos - self._skipped)

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
    """Raised by `Parser.parse()` when parsing went on after errors: all of them, by position.

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
class Located[B]:
    """A value and the part of the input it was read from, as `Parser.located()` gives it.

    `start` and `end` are indexes into what was parsed, `end` exclusive: character offsets in a
    text, so that `text[m.start : m.end]` is what the value was read from, and token indexes in
    tokens, so that `tokens[m.start].start` is where it begins. `funcparserlib.util.Lines` gives the
    line and the column of an offset.
    """

    value: B
    start: int
    end: int


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

    @staticmethod
    def _made_leaf(name: str, run: Any, form: tuple[Any, ...]) -> Parser[Any, Any]:
        """A parser of text or of `funcparserlib.binary`: a function and its form."""
        p: Parser[Any, Any] = Parser(run).named(name)
        p._leaf = True
        p._form = form
        return p

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
                failed = State(st.pos, st.max, expected, st.context, st.problems, st._skipped)
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

        If the parser fails to parse the tokens, it raises `NoParseError`. If `Parser.validated()`
        or `Parser.recover()` recorded errors, it raises `ParseErrors` with them, and with the
        `NoParseError` when the parse failed as well.

        `describe` turns a token into the `TokenDescription` error messages show. By default it
        reads the `value`, `start` and `end` of a `funcparserlib.lexer.Token`, and shows any other
        object as it is.

        !!! Note

            Although `Parser.parse()` can parse sequences of any objects (including
            `str` which is a sequence of `str` chars), **the recommended way** is
            parsing sequences of `Token` objects.

            You **should** use a regex-based tokenizer `make_tokenizer()` defined in
            `funcparserlib.lexer` to convert your text into a sequence of `Token`
            objects before parsing it. You will get more readable parsing error messages
            (as `Token` objects contain their position in the source file) and good
            separation of the lexical and syntactic levels of the grammar.
        """
        return self._result(self.run(tokens, State(0, 0)), tokens, describe or _describe)

    def parse_text(self, text: str) -> B:
        """Parse the text, with the line and column of the text in errors.

        It is for a grammar that reads text: `lit()`, `regex()`, the parsers of `skipping()`, and
        `a()` and `some()` of a character, and `tokenized()`. A grammar of `tok()` reads tokens:
        give those to `Parser.parse()`, or bind it to its tokenizer with `tokenized()`.

        It raises what `Parser.parse()` raises, `TypeError` for anything but a `str`, and
        `ValueError` for a grammar of `tok()` that reaches a character. For `tokenized()`, it raises
        the errors of the tokens, and `LexerError` where the grammar succeeds before text no token
        spec matches.
        """
        # Imported here: the module reads the classes of this one.
        from ._compile import leading_skip  # noqa: PLC0415

        _require_text(text)
        skip = leading_skip(self)
        m = skip.match(text) if skip is not None else None
        start = m.end() if m is not None else 0
        res = _sharing_tokens(lambda: self._run_input(text, start, "text"))
        return self._result(res, _TextPlaces(text), _describe)

    def parse_bytes(self, data: bytes | bytearray) -> B:
        """Parse the bytes, with the offset of the bytes in errors, in hexadecimal.

        It is experimental, as `funcparserlib.binary` is, whose grammars it runs. It raises what
        `Parser.parse()` raises, `TypeError` for anything but `bytes` or a `bytearray`, and
        `ValueError` for a grammar of `tok()` that reaches a byte. Parsers read `bytes` methods and
        slices, so to parse a `memoryview`, make the copy with `bytes(view)`.
        """
        _require_bytes(data)
        return self._result(self._run_input(data, 0, "bytes"), _Offsets(data), _describe)

    def scan(
        self,
        tokens: Sequence[A],
        *,
        describe: Callable[[A], TokenDescription] | None = None,
        span: Callable[[Sequence[A], int, int], tuple[int, int]] | None = None,
    ) -> Iterator[Located[B]]:
        """The parts of `tokens` this grammar reads, from left to right, without overlaps.

        The grammar is tried at each index of the tokens, as `Parser.scan_text()` tries it at each
        offset of a text, and goes on past each match. The iterator is lazy: stop at the first
        match, and the tokens after it are not read. A `Located` has the indexes of its first token
        and of the one after its last. Given `span`, a function from the tokens and those two
        indexes to the start and the end of the text they came from, it has those instead.

        A grammar that fails at an index has no match there, unless a `commit()` failed or
        `Parser.validated()` rejected a value: then the iterator raises what `Parser.parse()`
        raises, with `describe` for the tokens. It raises `ValueError` for a match that reads no
        tokens, as it would have no end, and for a `span` that ends before it starts, and
        `TypeError` for a `str` or `bytes`, which `Parser.scan_text()` reads.

        Examples:
            ```pycon
            >>> from funcparserlib.lexer import Token
            >>> pair = tok("name") + tok("int")
            >>> tokens = [Token("int", "1"), Token("name", "a"), Token("int", "2")]
            >>> list(pair.scan(tokens))
            [Located(value=('a', '2'), start=1, end=3)]

            ```
        """
        _require_tokens(tokens)
        return self._scan_tokens(tokens, describe or _describe, span, None)

    def _scan_tokens(
        self,
        tokens: Sequence[A],
        describe: Callable[[A], TokenDescription],
        span: Callable[[Sequence[A], int, int], tuple[int, int]] | None,
        generated: _Generated | None,
    ) -> Iterator[Located[B]]:
        index = 0
        at = None if generated is None else generated(tokens)
        while _has_token(tokens, index):
            found = _generated_at(at, index)
            if found is None:
                index += 1
                continue
            if found is _UNDECIDED:
                res = self.run(tokens, State(index, index))
                if res.state.problems or (isinstance(res, ParsingFailure) and res.committed):
                    self._result(res, tokens, describe)
                if isinstance(res, ParsingFailure):
                    index += 1
                    continue
                value, end = res.value, res.state.pos
            else:
                value, end = cast("tuple[Located[B], int]", found)[0].value, found[1]
            if end == index:
                raise with_hint(
                    ValueError(f"{self.name} matched no tokens at index {index}"),
                    "A match has to read something: repeat the parser that maybe(), many() or"
                    " pure() wraps.",
                )
            start, stop = (index, end) if span is None else span(tokens, index, end)
            if stop < start:
                raise with_hint(
                    ValueError(
                        f"span gave an end of {stop}, before its start of {start},"
                        f" for tokens {index} to {end}"
                    ),
                    "`span` gives the start and then the end of the text the tokens came from.",
                )
            yield Located(value, start, stop)
            index = end

    def scan_text(self, text: str) -> Iterator[Located[B]]:
        """The parts of `text` this grammar reads, from left to right, without overlaps.

        The grammar is tried at each offset of the text, and goes on past each match, so it does not
        need to describe the text around what it reads. A grammar that starts with `tokenized()` is
        tried only where a token of the text starts, as its tokenizer reads the text from its start.
        The iterator is lazy: stop at the first match, and the rest of the text is not read. A
        `Located` ends where its value does, before any skip read after it.

        A grammar that fails at an offset has no match there, unless a `commit()` failed or
        `Parser.validated()` rejected a value: then the iterator raises what `Parser.parse_text()`
        raises. Use `commit()` for what starts as a match and has to be one.

        It raises `TypeError` for anything but a `str`, `ValueError` for a grammar of `tok()`, and
        `ValueError` for a match that reads nothing, as it would have no end.

        Examples:
            ```pycon
            >>> number = regex(r"\\d+") >> int
            >>> [m.value for m in number.scan_text("from 10 to 20")]
            [10, 20]
            >>> next(number.scan_text("from 10 to 20"))
            Located(value=10, start=5, end=7)

            ```
        """
        # Imported here: the module reads the classes of this one.
        from ._compile import reads_tokens  # noqa: PLC0415

        _require_text(text)
        if reads_tokens(self):
            raise _reads_tokens_error("text")
        return self._scan_text(text, None)

    def _scan_text(self, text: str, generated: _Generated | None) -> Iterator[Located[B]]:
        # Imported here: the module reads the classes of this one.
        from ._compile import leading_tokenizer  # noqa: PLC0415

        offset = 0
        shared: dict[tuple[int, int], _Chain] = {}
        tokenizer = leading_tokenizer(self)
        starts = None
        if tokenizer is not None:
            chain = shared[id(tokenizer), id(text)] = _Chain(text, tokenizer)
            starts = _TokenStarts(chain)
        at = None if generated is None else generated(text)
        while offset < len(text):
            token = _chains.set(shared) if _chains.get() is None else None
            try:
                found = self._next_match(text, offset, at, starts)
            finally:
                if token is not None:
                    _chains.reset(token)
            if found is None:
                return
            value, start, end, stop = found
            yield Located(value, start, stop)
            offset = end

    def _next_match(
        self, text: str, offset: int, at: Callable[[int], Any] | None, starts: _TokenStarts | None
    ) -> tuple[B, int, int, int] | None:
        """The value, the start, the end and the end of the value of the first match from `offset`,
        `None` where there is none. Given `starts`, a match starts only where a token does.
        """
        while offset < len(text):
            if starts is not None:
                start = starts.at_or_after(offset)
                if start is None:
                    return None
                offset = start
            found = _generated_at(at, offset)
            if found is None:
                offset += 1
                continue
            if found is not _UNDECIDED:
                located, end = cast("tuple[Located[B], int]", found)
                if end > offset:
                    return located.value, offset, end, located.end
            try:
                res = self.run(cast("Any", text), State(offset, offset))
            except _NotAToken as e:
                raise _reads_tokens_error("text") from e
            if res.state.problems or (isinstance(res, ParsingFailure) and res.committed):
                self._result(res, _TextPlaces(text), _describe)
            if isinstance(res, ParsingFailure):
                offset += 1
                continue
            end = res.state.pos
            if end == offset:
                raise with_hint(
                    ValueError(f"{self.name} matched no text at offset {offset}"),
                    "A match has to read something: repeat the parser that maybe(), many() or"
                    " pure() wraps.",
                )
            return res.value, offset, end, res.state._value_end(offset)
        return None

    def replace_text(self, text: str, f: Callable[[Located[B]], str]) -> str:
        """The `text` with each match `Parser.scan_text()` gives replaced by what `f` returns for
        it.

        Everything outside a match stays as it is, including the skip after one. It raises what
        `Parser.scan_text()` raises, so it gives no text at all when a part of it is wrong.

        Examples:
            ```pycon
            >>> number = regex(r"\\d+") >> int
            >>> number.replace_text("from 10 to 20", lambda m: str(m.value * 2))
            'from 20 to 40'

            ```
        """
        return _replaced(text, self.scan_text(text), f)

    def _run_input(self, data: str | bytes | bytearray, start: int, kind: str) -> ParsingResult[B]:
        """The result of `data` from `start`, which a `tok()` cannot read."""
        try:
            return self.run(cast("Any", data), State(start, start))
        except _NotAToken as e:
            raise _reads_tokens_error(kind) from e

    def _result(
        self, res: ParsingResult[B], tokens: Sequence[Any], show: Callable[[Any], TokenDescription]
    ) -> B:
        """The value of `res`, or the errors of a failed parse and those parsing went on after."""
        errors = [_problem_error(problem, tokens, show) for problem in res.state.problems]
        if isinstance(res, ParsingFailure) and res.message is not _STOPPED:
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

    def compile(self, *, source_file: str | PathLike[str] | None = None) -> CompiledParser[A, B]:
        """Generate Python functions that parse as this parser does, several times faster.

        Use it on a finished grammar that parses a lot of tokens, or a lot of times. The generated
        functions handle the parses that succeed; for any other, `CompiledParser.parse()` runs this
        parser on the same tokens, so the result and the errors are this parser's.

        That has a cost: a failed parse takes longer, and calls the functions given to `>>` a second
        time. So does a parse that succeeds after a `Parser.validated()` rejected a value on a
        branch `|` backtracked from, or after a `commit()` around a parsing function failed. The
        compiled parser is a snapshot of the grammar, and does not follow a `forward_decl()` defined
        again afterwards. A grammar nested too deeply to compile stays interpreted, as does a
        `many()` of a parser that matches no tokens.

        A grammar of `tokenized()` reads text through generated functions too, faster and in less
        memory still, and so does a grammar that reads no `tok()`, such as one of `lit()` and
        `regex()` or of `funcparserlib.binary`.

        Given a `source_file`, the Python source of the generated functions is written there, and
        their frames report that file, so tracebacks, `pdb` and an IDE can show them. For a grammar
        of `tokenized()`, the file holds what `CompiledParser.parse_text()` runs, then what it runs
        over the tokens. The file is rewritten on each call and left alone for a grammar that stays
        interpreted, and `OSError` reaches the caller. It is for reading: the names in it stand for
        objects of the running parser, so it cannot be imported.

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
        internals = Internals(_Tuple, _IGNORED, _ignore, _token_value, _Undecided, _located_at)
        # What reads text reads bytes too: generated code takes either.
        if reads_text(self):
            fast_text, source = compile_text(self, internals, None, path)
            write_source(path, source)
            return CompiledParser(self, None, fast_text, reads_tokens=False)
        fast_text, text_source = None, None
        tokens_read = reads_tokens(self)
        if not tokens_read:
            fast_text, text_source = compile_text(self, internals, None, path)
        fast, source = compile_grammar(self, internals, path, line_after(text_source))
        write_source(path, text_source, source)
        return CompiledParser(self, fast, fast_text, reads_tokens=tokens_read)

    @overload
    def recover(self, until: Parser[A, Any] | None = None) -> Parser[A, B | None]: ...

    @overload
    def recover[C](
        self, until: Parser[A, Any] | None = None, *, default: C
    ) -> Parser[A, B | C]: ...

    def recover(
        self, until: Parser[A, Any] | None = None, *, default: Any = None
    ) -> Parser[A, Any]:
        """Give `default` for a broken part, and go on from where it ends.

        The errors stay recorded: `Parser.parse()` raises them together as `ParseErrors`, and
        `default` only lets the grammar around the part go on. What comes after this parser, a
        function after `>>` included, runs on `default`, so a grammar that makes one has to say what
        it means: a value of a type of its own says so best.

        A part is broken when a function given to `Parser.validated()` rejected something in it. Its
        input was read, so parsing goes on after it. With `until`, a part is also broken when this
        parser fails past its first token: the failure is recorded, and parsing goes on from the
        first position where `until` matches, without reading it. The text runs out when `until`
        never matches. A parser fails as usual at its first token, so `many(p.recover(...))` ends as
        `many(p)` does.

        A parser that needs a rejected value to go on, such as `counted()` of a rejected count,
        stops there, as the rest of its input cannot be read without it. Only `until` recovers from
        that.

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
        def _recover(tokens: Sequence[A], s: State) -> ParsingResult[Any]:
            # A fresh furthest position tells whether this parser read a token.
            res = self.run(tokens, State(s.pos, s.pos, (), s.context, s.problems, s._skipped))
            st = res.state
            if res.__class__ is not ParsingFailure:
                if st.rejected_since(s):
                    return ParsingSuccess(default, _merged(_closed(st, s), s))
                return ParsingSuccess(res.value, _merged(st, s))
            if until is None or st.max == s.pos:
                return res._replace_state(_merged(st, s))
            problems = _closed(st, s).problems
            if res.message is not _STOPPED:
                names = dict.fromkeys(p.name for p, _ in st.expected if p.name is not None)
                expected = f", expected: {' or '.join(names)}" if names else ""
                problems = (*problems, _unexpected(tokens, st.max, expected, st.context))
            pos = st.max
            while (
                pos < len(tokens) and until.run(tokens, State(pos, pos)).__class__ is ParsingFailure
            ):
                pos += 1
            return ParsingSuccess(default, State(pos, pos, (), s.context, problems))

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
            if isinstance(res, ParsingFailure) or res.state.rejected_since(s):
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
            if isinstance(res, ParsingFailure) or res.state.rejected_since(s) or pred(res.value):
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
            inner = State(s.pos, s.max, s.expected, (*s.context, label), s.problems, s._skipped)
            res = self.run(tokens, inner)
            st = res.state
            out = State(st.pos, st.max, st.expected, s.context, st.problems, st._skipped)
            if isinstance(res, ParsingFailure):
                return ParsingFailure(out, res.message, res.committed)
            return ParsingSuccess(res.value, out)

        _context._form = ("context", self)
        return _context

    def validated[C](self, f: Callable[[B], C], expected: str | None = None) -> Parser[A, C]:
        """Make the value of the parser with `f`, which can say that the value is not valid.

        `f` returns the new value, or raises `Invalid`. Either way the input was read, so a rejected
        value does not fail the parser: `|` tries nothing else and parsing goes on, which lets one
        run find every error. `Parser.parse()` raises `ParseErrors` with them at the end. No
        function after `>>` runs over a rejected value, and no value is made up for it: only
        `Parser.recover()` gives the part a value. `Invalid(message, keep=True)` goes on with the
        value instead, for a value that is real and only breaks a rule.

        A `NoParseError` or `ParseErrors` from `f` rejects the value too, so `f` can parse the value
        with another grammar. Its error is then the `__cause__` of the one reported, whose message
        is that error's, or names the token and `expected` when it is given.

        To decide among alternatives, use `Parser.when()`, which fails the parser.

        Examples:
            ```pycon
            >>> def small(digit: str) -> int:
            ...     if digit > "5":
            ...         raise Invalid(f"{digit} is too big")
            ...     return int(digit)
            >>> digits = many(some(str.isdigit).validated(small)) + -finished
            >>> digits.parse("15")
            [1, 5]
            >>> digits.parse("179")
            Traceback (most recent call last):
                ...
            funcparserlib.parser.ParseErrors: 2 parsing errors (2 sub-exceptions)

            ```
        """

        @_parser(self.name, shape=self._shape, first=self._first)
        def _validated(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            st = res.state
            if isinstance(res, ParsingFailure) or st.rejected_since(s):
                return cast("ParsingResult[C]", res)
            try:
                return ParsingSuccess(f(res.value), st)
            except Invalid as e:
                end = st._value_end(s.pos)
                problem = Problem(e.message, s.pos, end, s.context, False, not e.keep, e)
                found, value = (problem,), res.value if e.keep else None
            except (NoParseError, ParseErrors) as e:
                end = st._value_end(s.pos)
                found, value = _nested(tokens, s.pos, end, s.context, e, expected), None
            problems = (*st.problems, *found)
            checked = State(st.pos, st.max, st.expected, st.context, problems, st._skipped)
            return ParsingSuccess(cast("C", value), checked)

        # In generated code it is a conversion: what `f` raises leaves that code, as any exception.
        _validated._form = ("map", self, f)
        return _validated

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
                if _has_token(tokens, pos):
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

    def located(self) -> Parser[A, Located[B]]:
        """A parser of the same input that gives its value with where it was read from.

        `Located.start` and `Located.end` are indexes into what is parsed: character offsets in a
        text and token indexes in tokens. In a grammar of `skipping()`, a span ends where the last
        parser that read text stopped, before the skip read after it. Inside a `prefixed()` region,
        they are relative to the region.

        A grammar with `located()` compiles to generated code as any other grammar does; a
        `prefixed()` with a body in a grammar of text with a `located()` runs as the parser does.

        Examples:
            ```pycon
            >>> spaced = skipping(r"\\s+")
            >>> word = spaced.regex(r"[a-z]+").located()
            >>> (word + word).parse_text("foo  bar")
            (Located(value='foo', start=0, end=3), Located(value='bar', start=5, end=8))

            ```
        """

        @_parser(self.name, shape=self._shape, first=self._first)
        def _located(tokens: Sequence[A], s: State) -> ParsingResult[Located[B]]:
            res = self.run(tokens, s)
            if res.state.problems is not s.problems and res.state.rejected_since(s):
                return cast("ParsingResult[Located[B]]", res)
            end = res.state._value_end(s.pos)
            return res.map(lambda value: _located_at(tokens, value, s.pos, end))

        _located._form = ("located", self)
        return _located

    def map[C](self, f: Callable[[B], C]) -> Parser[A, C]:
        """Transform the parsing result by applying the specified function; the same as `>>`."""
        if self._test is not None and not debug:
            return self._fused(f)

        @_parser(self.name, shape=self._shape, first=self._first)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            if res.state.problems is not s.problems and res.state.rejected_since(s):
                return cast("ParsingResult[C]", res)
            return res.map(f)

        _map._form = ("map", self, f)
        return _map

    def starmap[C](self, f: Callable[..., C]) -> Parser[A, C]:
        """Transform the parsing results (tuple) by applying the specified function."""

        @_parser(self.name, shape=self._shape, first=self._first)
        def _map(tokens: Sequence[A], s: State) -> ParsingResult[C]:
            res = self.run(tokens, s)
            if res.state.problems is not s.problems and res.state.rejected_since(s):
                return cast("ParsingResult[C]", res)
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
            if res.__class__ is ParsingFailure:
                return cast("ParsingResult[C]", res)
            if res.state.rejected_since(s):
                return _stopped(res.state)
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


@final
class CompiledParser[A, B]:
    """A grammar compiled to Python functions, see `Parser.compile()`.

    Users get it from `Parser.compile()`; its constructor is internal.
    """

    __slots__ = ("_bound", "_fast", "_fast_text", "_parser", "_reads_tokens", "_scanners")

    def __init__(
        self,
        parser: Parser[A, B],
        fast: Callable[[Sequence[A]], tuple[B, int] | None] | None,
        fast_text: Callable[[str], tuple[B, int] | None] | None = None,
        *,
        bound: tuple[Parser[Token, Any], Tokenizer, Callable[[Sequence[Any]], Any] | None]
        | None = None,
        reads_tokens: bool = True,
    ) -> None:
        """`parser`, with its generated functions, `None` for a grammar that stays interpreted.

        `bound` is the grammar of tokens, the tokenizer and the functions over the tokens of a
        parser of `tokenized()`.
        """
        self._parser = parser
        self._fast = fast
        self._fast_text = fast_text
        self._bound = bound
        self._reads_tokens = reads_tokens
        self._scanners: dict[bool, _Generated | None] = {}

    def _scanner(self, text: bool) -> _Generated | None:
        """Generated code of `located()` of the grammar, for a scan of text or of tokens."""
        if text not in self._scanners:
            # Imported here: the module reads the classes of this one.
            from ._compile import Internals, compile_scanner  # noqa: PLC0415

            internals = Internals(_Tuple, _IGNORED, _ignore, _token_value, _Undecided, _located_at)
            self._scanners[text] = compile_scanner(self._parser.located(), internals, text=text)
        return self._scanners[text]

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
        self, text: str, *, describe: Callable[[Token], TokenDescription] | None = None
    ) -> B:
        """Parse the text, as `Parser.parse_text()` does.

        A grammar of `tokenized()` that reads tokens only through `tok()`, with a tokenizer whose
        specs join into one regex, runs on the text without making a `Token`; any other tokenizes
        and runs the functions generated for the tokens. `describe` is for the tokens of
        `tokenized()`.

        Raises what `Parser.parse_text()` raises.
        """
        _require_text(text)
        return self._read(text, describe)

    def parse_bytes(self, data: bytes | bytearray) -> B:
        """Parse the bytes, as `Parser.parse_bytes()` does."""
        _require_bytes(data)
        return self._read(cast("Any", data), None)

    def _read(self, text: str, describe: Callable[[Token], TokenDescription] | None) -> B:
        """The result of `text` or bytes, through the functions generated for text if there are."""
        return _sharing_tokens(lambda: self._read_shared(text, describe))

    def _read_shared(self, text: str, describe: Callable[[Token], TokenDescription] | None) -> B:
        res = None
        if self._fast_text is not None and not debug:
            try:
                res = self._fast_text(text)
            except _Undecided:
                res = None
        return res[0] if res is not None else self._interpreted(text, describe)

    def _interpreted(self, text: str, describe: Callable[[Token], TokenDescription] | None) -> B:
        """The result of `text` without the functions generated for text."""
        if self._bound is not None:
            grammar, tokenizer, fast = self._bound
            _require_text(text)
            # After the functions for text, the parser runs, not the functions for tokens.
            fast = fast if self._fast_text is None else None
            return cast("B", _parse_tokens_of_text(grammar, tokenizer, text, fast, describe))
        if self._reads_tokens:
            data: object = text
            raise _reads_tokens_error("bytes" if isinstance(data, bytes | bytearray) else "text")
        if isinstance(text, bytes | bytearray):
            return self._parser.parse_bytes(text)
        return self._parser.parse_text(text)

    def scan(
        self,
        tokens: Sequence[A],
        *,
        describe: Callable[[A], TokenDescription] | None = None,
        span: Callable[[Sequence[A], int, int], tuple[int, int]] | None = None,
    ) -> Iterator[Located[B]]:
        """The parts of `tokens` the grammar reads, as `Parser.scan()` gives them.

        Generated code tries the grammar at each index, and the parser decides where it cannot. It
        reads the type of every token first, as `CompiledParser.parse()` does.
        """
        _require_tokens(tokens)
        generated = self._scanner(text=False)
        return self._parser._scan_tokens(tokens, describe or _describe, span, generated)

    def scan_text(self, text: str) -> Iterator[Located[B]]:
        """The parts of `text` the grammar reads, as `Parser.scan_text()` gives them.

        Generated code tries the grammar where `Parser.scan_text()` does, and the parser decides
        where it cannot.
        """
        # Imported here: the module reads the classes of this one.
        from ._compile import reads_tokens  # noqa: PLC0415

        _require_text(text)
        if reads_tokens(self._parser):
            raise _reads_tokens_error("text")
        return self._parser._scan_text(text, self._scanner(text=True))

    def replace_text(self, text: str, f: Callable[[Located[B]], str]) -> str:
        """The `text` with its matches replaced, as `Parser.replace_text()` gives it."""
        return _replaced(text, self.scan_text(text), f)

    def try_parse(self, text: str | bytes | bytearray) -> B | None:
        """The value of `text`, or `None` where it does not parse.

        It is for text that is often invalid, when the reason does not matter: a parse that fails
        stops there, which can be several times faster than making its error. A rejected value or a
        failed `commit()` leaves the generated code, so it takes the time of `parse_text()`. A
        grammar whose value can be `None` cannot tell that from a failure this way. Text nested too
        deeply gives `None` too, as a `RecursionError`. What a function after `>>` raises it raises,
        and it raises `TypeError` for anything but text or bytes.
        """
        if isinstance(text, bytes | bytearray | memoryview):
            _require_bytes(text)
        else:
            _require_text(text)
        if self._fast_text is not None and not debug:
            try:
                res = _sharing_tokens(lambda: cast("Any", self._fast_text)(cast("str", text)))
            except _Undecided:
                pass
            else:
                return None if res is None else res[0]
        try:
            return _sharing_tokens(lambda: self._interpreted(cast("str", text), None))
        except (NoParseError, ParseErrors, LexerError, RecursionError):
            return None


class _Chain:
    """The tokens of a text from any offset, each read once.

    A tokenizer reads a token from where it is by the text alone, so tokens read from two offsets
    are the same from the first offset they share: what is read at an offset is kept for each start.
    """

    __slots__ = ("_counts", "_lines", "_read", "skip", "text", "tokenizer")

    def __init__(self, text: str, tokenizer: Tokenizer) -> None:
        self.text = text
        self.tokenizer = tokenizer
        self.skip = tokenizer._skip
        self._lines: Lines | None = None
        self._read: dict[int, tuple[str, int] | None] = {}
        self._counts: dict[int, int] = {}

    def read(self, pos: int) -> tuple[str, int] | None:
        """The type and the end of the token at `pos`, `None` at the end or text no spec matches."""
        try:
            return self._read[pos]
        except KeyError:
            found = self.tokenizer._read(self.text, pos) if pos < len(self.text) else None
            self._read[pos] = found
            return found

    def count(self, pos: int) -> int:
        """How many tokens a parser reads from `pos`: those not skipped, and one for unreadable
        text.
        """
        counts = self._counts
        path: list[int] = []
        at = pos
        while at not in counts:
            found = self.read(at)
            if found is None:
                counts[at] = 1 if at < len(self.text) else 0
                break
            path.append(at)
            at = found[1]
        n = counts[at]
        for start in reversed(path):
            token_type = cast("tuple[str, int]", self._read[start])[0]
            if token_type not in self.skip:
                n += 1
            counts[start] = n
        return counts[pos]

    def place(self, offset: int) -> Place:
        """The line and the column of `offset`."""
        if self._lines is None:
            self._lines = Lines(self.text)
        return self._lines.place(offset)


class _TextTokens(Sequence[Token]):
    """The tokens of a text from `start`, read as a parser asks for them, with their offsets.

    Where no token spec matches, the last token is one no grammar reads.
    """

    __slots__ = ("_chain", "_complete", "_ends", "_next", "_start", "_starts", "_tokens")

    def __init__(self, chain: _Chain, start: int) -> None:
        self._chain = chain
        self._start = start
        self._next = start
        self._tokens: list[Token] = []
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._complete = False

    @classmethod
    def of_text(cls, text: str, tokenizer: Tokenizer) -> _TextTokens:
        """All the tokens of `text`, read once from its start, with no table of what was read."""
        tokens = cls(_Chain(text, tokenizer), 0)
        items, starts, ends = tokens._tokens, tokens._starts, tokens._ends
        lines = Lines(text)
        pos = 0
        try:
            for token in tokenizer(text):
                # A token from `Tokenizer.__call__()` has its place; its offset is where that is.
                start = lines.offset(cast("Place", token.start))
                items.append(token)
                starts.append(start)
                ends.append(start + len(token.value))
            pos = len(text)
        except LexerError as e:
            pos = lines.offset(e.place)
            word = _WORD.match(text, pos)
            items.append(Token(_UNREADABLE, word.group() if word else "\r", e.place, e.place))
            starts.append(pos)
            ends.append(pos)
        tokens._next = pos
        tokens._complete = True
        return tokens

    def __len__(self) -> int:
        if self._complete:
            return len(self._tokens)
        return self._chain.count(self._start)

    def _fill(self, i: int) -> bool:
        """Whether there is a token `i`, reading the tokens up to it."""
        if self._complete:
            return i < len(self._tokens)
        chain, tokens = self._chain, self._tokens
        while len(tokens) <= i:
            pos = self._next
            found = chain.read(pos)
            if found is None:
                if pos >= len(chain.text) or (self._starts and self._starts[-1] == pos):
                    return False
                word = _WORD.match(chain.text, pos)
                place = chain.place(pos)
                tokens.append(Token(_UNREADABLE, word.group() if word else "\r", place, place))
                self._starts.append(pos)
                self._ends.append(pos)
                continue
            token_type, end = found
            self._next = end
            if token_type in chain.skip:
                continue
            line, column = chain.place(end)
            value = chain.text[pos:end]
            tokens.append(Token(token_type, value, chain.place(pos), (line, column - 1)))
            self._starts.append(pos)
            self._ends.append(end)
        return True

    @overload
    def __getitem__(self, i: int) -> Token: ...

    @overload
    def __getitem__(self, i: slice) -> list[Token]: ...

    def __getitem__(self, i: int | slice) -> Token | list[Token]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        if i < 0 or not self._fill(i):
            raise IndexError(i)
        return self._tokens[i]

    def offset(self, i: int) -> int:
        """Where token `i` starts in the text; for `i` past the last token, where the tokens end."""
        if self._fill(i):
            return self._starts[i]
        # A failed `_fill()` has read past the skipped tokens after the last one.
        if self._starts and self._starts[-1] == self._ends[-1]:
            return self._starts[-1]
        return self._next

    def end(self, i: int) -> int:
        """Where token `i` ends in the text."""
        self._fill(i)
        return self._ends[i]

    def unreadable(self) -> int | None:
        """The offset of the text no token spec matches, `None` where every token reads."""
        n = len(self)
        if n and self._fill(n - 1) and self._tokens[n - 1].type == _UNREADABLE:
            return self._starts[n - 1]
        return None


class _Skipped:
    """Where what a tokenizer skips ends, as a match of a pattern gives it."""

    __slots__ = ("_end",)

    def __init__(self, end: int) -> None:
        self._end = end

    def end(self) -> int:
        """The offset after the skipped tokens."""
        return self._end


class _TokenStarts:
    """The offsets where the tokens of a text start, read from its start as a scan asks for them.

    Text no token spec matches starts no token, and the tokens go on where a spec matches again.
    """

    __slots__ = ("_chain", "_last", "_pos")

    def __init__(self, chain: _Chain) -> None:
        self._chain = chain
        self._pos = 0
        self._last: int | None = None

    def at_or_after(self, offset: int) -> int | None:
        """The first offset from `offset` where a token starts, `None` past the last."""
        while self._last is None or self._last < offset:
            if not self._advance():
                return None
        return self._last

    def _advance(self) -> bool:
        """Whether there is another token, reading on to where it starts."""
        chain = self._chain
        while self._pos < len(chain.text):
            found = chain.read(self._pos)
            if found is None:
                self._pos += 1
                continue
            token_type, end = found
            start, self._pos = self._pos, end
            if token_type not in chain.skip:
                self._last = start
                return True
        return False


class _SkippedTokens:
    """The tokens a tokenizer skips at a position, read as `re.Pattern.match()` reads a skip."""

    __slots__ = ("_tokenizer",)

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    def match(self, text: str, pos: int = 0, /) -> _Skipped | None:
        """Where the skipped tokens from `pos` end, `None` where none starts there."""
        tokenizer, end = self._tokenizer, pos
        while (found := tokenizer._read(text, end)) is not None and found[0] in tokenizer._skip:
            end = found[1]
        return _Skipped(end) if end > pos else None


class _Tokenized[B](Parser[str, B]):
    """The type of `tokenized()`: a grammar of tokens that reads text through its tokenizer."""

    _grammar: Parser[Token, B]
    _tokenizer: Tokenizer

    def parse_text(self, text: str) -> B:
        """Parse the text, as `Parser.parse()` parses its tokens, with the errors of the tokens."""
        _require_text(text)
        return _parse_tokens_of_text(self._grammar, self._tokenizer, text, None, None)

    def compile(self, *, source_file: str | PathLike[str] | None = None) -> CompiledParser[str, B]:
        """Generate the functions of the grammar for its text, and for its tokens."""
        # Imported here: the module reads the classes of this one.
        from ._compile import Internals, compile_grammar, compile_text, line_after, write_source  # noqa: PLC0415

        path = None if source_file is None else str(Path(source_file).resolve())
        internals = Internals(_Tuple, _IGNORED, _ignore, _token_value, _Undecided, _located_at)
        grammar, tokenizer = self._grammar, self._tokenizer
        fast_text, text_source = None, None
        if tokenizer._pattern is not None:
            fast_text, text_source = compile_text(grammar, internals, tokenizer._scan, path)
        fast, source = compile_grammar(grammar, internals, path, line_after(text_source))
        write_source(path, text_source, source)
        bound = (grammar, tokenizer, fast)
        return CompiledParser(self, None, fast_text, bound=bound, reads_tokens=False)


def tokenized[B](grammar: Parser[Token, B], tokenizer: Tokenizer) -> Parser[str, B]:
    """A parser of text that reads it through `tokenizer`, as `grammar` reads the tokens.

    It is a parser of text as `lit()` and `regex()` are, so it combines with them and has
    `Parser.parse_text()`, `Parser.scan_text()`, `Parser.replace_text()` and `Parser.compile()`.
    Where it is reached, it tokenizes from there as `grammar` reads, and it ends where `grammar`
    stops. The tokens in the tokenizer's `skip` after its last token are skipped, and those where it
    is reached are not read, as what comes before skips them; `Parser.parse_text()` of a grammar
    that starts with it reads them at the start of the text. `Parser.located()` inside `grammar`
    gives offsets into the text.

    From the start of a text it reads the tokens `tokenizer` gives for the text. Where no token spec
    matches, its tokens end with one no grammar reads, so `grammar` fails there or stops before it.
    `Parser.parse_text()` of it raises the error of the tokens, and `LexerError` when `grammar`
    succeeds before the text no spec matches.

    Examples:
        ```pycon
        >>> from funcparserlib.lexer import TokenSpec, make_tokenizer
        >>> tokenize = make_tokenizer(
        ...     [TokenSpec("space", r"\\s+"), TokenSpec("int", r"\\d+"), TokenSpec("op", r"[+]")],
        ...     skip={"space"},
        ... )
        >>> number = tok("int") >> int
        >>> total = tokenized(number + many(-tok("op", "+") + number), tokenize)
        >>> total.parse_text("1 + 2 + 3")
        (1, [2, 3])
        >>> [m.value for m in total.scan_text("from 1 + 2 to 3")]
        [(1, [2]), (3, [])]

        ```
    """

    def run(text: Any, s: State) -> ParsingResult[B]:
        _require_text(text)
        chain = _chain_of(text, tokenizer)
        if (first := chain.read(s.pos)) is not None and first[0] in chain.skip:
            # What a grammar of text skips, the parser before skips, as after `skipping()`.
            return ParsingFailure(s.expect(p))
        tokens = _TextTokens(chain, s.pos)
        res = grammar.run(tokens, State(0, 0, (), s.context, ()))
        return _from_tokens(res, tokens, s)

    p = _built(_Tokenized, run)._rename(grammar.name)
    p._grammar = grammar
    p._tokenizer = tokenizer
    # What walks a grammar does not look inside: the bound grammar reads tokens, and this part text.
    p._form = ("tokenized", grammar, _SkippedTokens(tokenizer), tokenizer)
    return cast("Parser[str, B]", p)


_chains: ContextVar[dict[tuple[int, int], _Chain] | None] = ContextVar(
    "funcparserlib_chains", default=None
)
"""The tokens `tokenized()` read in the parse running, by tokenizer and text."""


def _sharing_tokens[T](f: Callable[[], T]) -> T:
    """`f()`, where the parsers of `tokenized()` share the tokens they read of a text."""
    if _chains.get() is not None:
        return f()
    token = _chains.set({})
    try:
        return f()
    finally:
        _chains.reset(token)


def _chain_of(text: str, tokenizer: Tokenizer) -> _Chain:
    """The tokens of `text` the running parse read through `tokenizer`; new ones outside a parse."""
    shared = _chains.get()
    if shared is None:
        return _Chain(text, tokenizer)
    key = (id(tokenizer), id(text))
    chain = shared.get(key)
    if chain is None or chain.text is not text or chain.tokenizer is not tokenizer:
        chain = shared[key] = _Chain(text, tokenizer)
    return chain


def _from_tokens[B](res: ParsingResult[B], tokens: _TextTokens, s: State) -> ParsingResult[B]:
    """`res` of `tokens`, which start at the state `s` of a text, in that text."""
    st = res.state
    if st.pos == 0:
        pos, skipped = s.pos, s._skipped
    else:
        pos = tokens.offset(st.pos)
        skipped = pos - tokens.end(st.pos - 1)
    furthest = max(tokens.offset(st.max), pos)
    if furthest > s.max:
        max_, expected = furthest, st.expected
    elif furthest == s.max:
        max_, expected = s.max, (*s.expected, *st.expected)
    else:
        max_, expected = s.max, s.expected
    problems = s.problems
    if st.problems:
        found = tuple(_moved_to_text(p, tokens) for p in st.problems)
        problems = (*problems, *found)
    out = State(pos, max_, expected, s.context, problems, skipped)
    if isinstance(res, ParsingFailure):
        return ParsingFailure(out, res.message, res.committed)
    return ParsingSuccess(res.value, out)


def _moved_to_text(problem: Problem, tokens: _TextTokens) -> Problem:
    """`problem` of `tokens`, at the offsets in the text of its tokens."""
    n = len(tokens)
    first = tokens.offset(min(problem.start, n))
    if problem.start >= n:
        return replace(problem, start=first, end=first + 1)
    last = tokens.end(min(problem.end, n) - 1) if problem.end > problem.start else first
    return replace(problem, start=first, end=last)


def _parse_tokens_of_text[B](
    grammar: Parser[Token, B],
    tokenizer: Tokenizer,
    text: str,
    fast: Callable[[Sequence[Any]], Any] | None,
    describe: Callable[[Any], TokenDescription] | None,
) -> B:
    """The value of the tokens of `text`, through `fast` first, with the errors of the tokens."""
    tokens = _TextTokens.of_text(text, tokenizer)
    stop = tokens.unreadable()
    if fast is not None and not debug and stop is None:
        res = fast(tokens)
        if res is not None:
            return cast("B", res[0])
    try:
        value = grammar.parse(tokens, describe=describe)
    except (NoParseError, ParseErrors) as e:
        if stop is not None:
            raise e from _lexer_error(text, stop)
        raise
    if stop is not None:
        raise _lexer_error(text, stop)
    return value


def _lexer_error(text: str, offset: int) -> LexerError:
    """The error of a tokenizer that stopped at `offset` of `text`."""
    place = place_of(text, offset)
    return LexerError(place, line_of(text, place[0]))


type _Generated = Callable[[Any], Callable[[int], tuple[Any, int] | None] | None]
"""Generated code of `located()` of a grammar, for an input: what a compiled scan runs first."""

_UNDECIDED = object()
"""What generated code gives at a position where the parser decides."""


def _generated_at(at: Callable[[int], Any] | None, pos: int) -> Any:
    """What generated code gives at `pos`: a value and an end, `None`, or `_UNDECIDED`."""
    if at is None or debug:
        return _UNDECIDED
    try:
        return at(pos)
    except _Undecided:
        return _UNDECIDED


def _replaced[B](text: str, matches: Iterable[Located[B]], f: Callable[[Located[B]], str]) -> str:
    """`text` with each of `matches` replaced by what `f` returns for it."""
    parts: list[str] = []
    last = 0
    for m in matches:
        parts += (text[last : m.start], f(m))
        last = m.end
    parts.append(text[last:])
    return "".join(parts)


def _located_at[B](tokens: Sequence[Any], value: B, start: int, end: int) -> Located[B]:
    """`value` read from `start` to `end` of `tokens`; for the tokens of a text, in the text."""
    if isinstance(tokens, _TextTokens):
        first = tokens.offset(start)
        return Located(value, first, tokens.end(end - 1) if end > start else first)
    return Located(value, start, end)


def _has_token(tokens: Sequence[Any], pos: int) -> bool:
    """Whether there is a token at `pos`, read without the length of `tokens`.

    The length of the tokens of `tokenized()` is read by tokenizing the rest of the text.
    """
    try:
        tokens[pos]
    except IndexError:
        return False
    return True


def _unexpected(
    tokens: Sequence[Any], pos: int, expected: str, context: tuple[str, ...]
) -> Problem:
    """The error of the token at `pos`, which `Parser.parse()` describes unless it is text."""
    if pos >= len(tokens):
        return Problem(f"got unexpected end of input{expected}", pos, pos + 1, context)
    if isinstance(tokens, str | bytes):
        message = f"got unexpected token: {_shown(tokens, pos)}{expected}"
        return Problem(message, pos, pos + 1, context)
    return Problem(expected, pos, pos + 1, context, unexpected=True)


def _nested(
    tokens: Sequence[Any],
    start: int,
    end: int,
    context: tuple[str, ...],
    error: NoParseError | ParseErrors,
    expected: str | None,
) -> tuple[Problem, ...]:
    """The errors of a value that another grammar could not parse."""
    if expected is not None:
        found = _unexpected(tokens, start, f", expected: {expected}", context)
        return (replace(found, end=end, rejected=True, cause=error),)
    errors = error.exceptions if isinstance(error, ParseErrors) else (error,)
    return tuple(Problem(e.msg, start, end, context, False, True, e) for e in errors)


def _reads_tokens_error(kind: str) -> ValueError:
    """The error of a grammar of `tok()` given `kind` of input, text or bytes."""
    return with_hint(
        ValueError(f"a grammar of tokens cannot parse {kind}"),
        "Pass tokens to `Parser.parse()`, or the grammar and its tokenizer to `tokenized()`.",
    )


def _require_tokens(tokens: object) -> None:
    """Raise `TypeError` for a `str` or `bytes`, which is text to scan, not tokens."""
    if isinstance(tokens, str | bytes | bytearray):
        raise with_hint(
            TypeError(f"{type(tokens).__name__} is no sequence of tokens to scan"),
            "Scan text with `Parser.scan_text()`, which reads it with its line and column.",
        )


def _require_text(text: object) -> None:
    """Raise `TypeError` for anything but a `str`."""
    if not isinstance(text, str):
        raise with_hint(
            TypeError(f"{type(text).__name__} cannot be parsed as text"),
            "Pass tokens to `Parser.parse()`, and bytes to `Parser.parse_bytes()`.",
        )


def _require_bytes(data: object) -> None:
    """Raise `TypeError` for anything but bytes; only the owner of a `memoryview` can copy it."""
    if isinstance(data, memoryview):
        raise with_hint(
            TypeError("a memoryview cannot be parsed"),
            "Parsers read `bytes` methods and slices: pass `bytes(view)` to make the copy.",
        )
    if not isinstance(data, bytes | bytearray):
        raise with_hint(
            TypeError(f"{type(data).__name__} cannot be parsed as bytes"),
            "Pass tokens to `Parser.parse()`, and text to `Parser.parse_text()`.",
        )


def _stopped(st: State) -> ParsingFailure:
    """The failure of a parser that needs a value a rejection left it without.

    It is final, and adds no error of its own: the rejection is the error.
    """
    return ParsingFailure(st, _STOPPED, committed=True)


def _closed(st: State, s: State) -> State:
    """`st` with the rejections since `s` closed, so what comes after has values again."""
    done = len(s.problems)
    problems = tuple(replace(p, rejected=False) for p in st.problems[done:])
    return State(
        st.pos, st.max, st.expected, st.context, (*st.problems[:done], *problems), st._skipped
    )


def _merged(st: State, s: State) -> State:
    """`st`, reached from a fresh furthest position, with what failed furthest of it and `s`."""
    if st.max > s.max:
        max_, expected = st.max, st.expected
    elif st.max == s.max:
        max_, expected = s.max, (*s.expected, *st.expected)
    else:
        max_, expected = s.max, s.expected
    return State(st.pos, max_, expected, st.context, st.problems, st._skipped)


def _moved(s: State, offset: int) -> State:
    """`s` of a region that starts at `offset`, in the input around it."""
    problems = tuple(replace(p, start=p.start + offset, end=p.end + offset) for p in s.problems)
    return State(s.pos + offset, s.max + offset, s.expected, s.context, problems, s._skipped)


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
        st = State(st.pos, st.max, (), st.context, st.problems, st._skipped)
    merged = _merged(st, at)
    problems = (*at.problems, *st.problems)
    moved = State(merged.pos, merged.max, merged.expected, at.context, problems, merged._skipped)
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
    length that is negative or runs past the input fails where those bytes start. What a
    `skipping()` skips before `body` is not read for it.

    Examples:
        ```pycon
        >>> from funcparserlib import binary as b
        >>> prefixed(b.byte()).parse_bytes(b"\\x02hi")
        b'hi'
        >>> prefixed(b.byte(), many(b.byte())).parse_bytes(b"\\x02hi")
        [104, 105]
        >>> prefixed(b.byte()).parse_bytes(b"\\x09hi")
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
        if res.state.rejected_since(s):
            return _stopped(res.state)
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
        >>> from funcparserlib import binary as b
        >>> counted(b.byte(), b.fixed(">H")).parse_bytes(b"\\x02\\x00\\x01\\x00\\x02")
        [1, 2]

        ```
    """

    @_parser(f"items counted by {count.name}", shape=True)
    def _counted(tokens: Sequence[A], s: State) -> ParsingResult[list[B]]:
        res = count.run(tokens, s)
        if res.__class__ is ParsingFailure:
            return cast("ParsingResult[list[B]]", res)
        if res.state.rejected_since(s):
            return _stopped(res.state)
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
        >>> from funcparserlib import binary as b
        >>> field = switch(b.byte(), {1: b.fixed(">H"), range(2, 4): b.byte()})
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
        if res.state.rejected_since(s):
            return _stopped(res.state)
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
    the errors the failed parser recorded, which reject input `|` accepts.

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
    elif st.max >= len(tokens) and isinstance(tokens, _TextPlaces):
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
    problem: Problem, tokens: Sequence[A], describe: Callable[[A], TokenDescription]
) -> NoParseError:
    """The `NoParseError` of a recorded error, spanning the tokens it covers."""
    token: A | None = None
    description: TokenDescription | None
    if isinstance(tokens, _Offsets):
        description = None
        problem = replace(problem, message=f"offset 0x{problem.start:x}: {problem.message}")
    elif isinstance(tokens, _TextPlaces):
        in_text = problem.start < len(tokens)
        description = tokens.span(problem.start, problem.end) if in_text else tokens.end()
    elif problem.start < len(tokens):
        token = tokens[problem.start]
        first = describe(tokens[problem.start])
        last = describe(tokens[max(problem.start, problem.end - 1)])
        description = TokenDescription(first.text, first.start, last.end)
        if problem.unexpected:
            problem = replace(
                problem, message=f"got unexpected token: {first.text}{problem.message}"
            )
    else:
        description = _after(tokens, describe)
    context = tuple(reversed(problem.context))
    msg = f"{_where(description)}{problem.message}" + "".join(f" in {c}" for c in context)
    state = State(problem.start, problem.start)
    error = NoParseError(msg, state, token=token, context=context, description=description)
    error.__cause__ = problem.cause
    return error


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
    failed = cast("ParsingFailure", res)
    return ParsingFailure(st, failed.message, failed.committed)


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
        try:
            t = tokens[pos]
        except IndexError:
            return ParsingFailure(s.expect(expected()), "got unexpected end of input")
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
        try:
            t = tokens[s.pos]
        except IndexError:
            return ParsingFailure(s.expect(_some), "got unexpected end of input")
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

            You **should** use a regex-based tokenizer `make_tokenizer()` defined in
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
            use a regex-based tokenizer `make_tokenizer()` defined in
            `funcparserlib.lexer`. You will get more readable parsing error messages (as
            `Token` objects contain their position in the source file) and good separation
            of the lexical and syntactic levels of the grammar.
    """

    def eq_type(t: Token) -> bool:
        try:
            return t.type == type
        except AttributeError as e:
            raise _NotAToken(*e.args) from e

    def eq_token(t: object) -> bool:
        # What `t == Token(type, value)` compares, without building a token.
        if isinstance(t, Token):
            return t.type == type and t.value == value
        if not hasattr(t, "type"):
            raise _NotAToken(f"{t.__class__.__name__!r} object has no attribute 'type'")
        return False

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
        res = p.run(tokens, State(s.pos, s.pos, (), s.context, s.problems, s._skipped))
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
    """Return a parser that parses one token, any token `p` does not parse.

    Examples:
        ```pycon
        >>> expr = anything_but(a("x"))
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

    name = _label_of(p)

    @_parser(f"anything but {name}" if name else "anything else", leaf=True)
    def _anything_but(tokens: Sequence[A], s: State) -> ParsingResult[A]:
        if not _has_token(tokens, s.pos):
            return ParsingFailure(s.expect(_anything_but), "got unexpected end of input")
        res = p.run(tokens, s)
        if isinstance(res, ParsingFailure):
            if res.committed:
                return res
            return ParsingSuccess(tokens[s.pos], s.reset_pos(s.pos + 1))
        return ParsingFailure(s.expect(_anything_but))

    return _anything_but


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


def separated[A, B](
    p: Parser[A, B], sep: Parser[A, Any], *, trailing: bool = False
) -> Parser[A, list[B]]:
    """Return a parser of any number of `p` with a `sep` between two of them, as a list of their
    values, without the values of `sep`.

    A value of `p` stays one item of the list, also when it is a tuple. A `sep` that no `p` follows
    is left unread, so it fails what comes next, unless `trailing` is true, which reads one after
    the last item.

    Examples:
        ```pycon
        >>> item = a("x") + a("y")
        >>> expr = separated(item, a(","))
        >>> expr.parse("xy,xy")
        [('x', 'y'), ('x', 'y')]
        >>> expr.parse("")
        []
        >>> (expr + -finished).parse("xy,")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: got unexpected end of input, expected: 'x'
        >>> (separated(item, a(","), trailing=True) + -finished).parse("xy,")
        [('x', 'y')]

        ```
    """
    first: Parser[A, list[B]] = p >> (lambda value: [value])
    items = first + many(-sep + p)
    if trailing:
        items = items + -maybe(sep)
    return maybe(items) >> (lambda r: [] if r is None else [*r[0], *r[1]])


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
        if not _has_token(tokens, s.pos):
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


@final
class Skipping:
    """The type of `skipping()`: it makes parsers of text that skip what its pattern matches."""

    __slots__ = ("_skip",)

    def __init__(self, pattern: str, flags: int = 0) -> None:
        """Make parsers that skip what the regex `pattern`, with its `flags`, matches."""
        if not isinstance(pattern, str):
            raise with_hint(
                TypeError(f"skip pattern must be a str, not {type(pattern).__name__}"),
                "To skip nothing, use `regex()` and `lit()`.",
            )
        compiled = re.compile(pattern, flags)
        if compiled.match(""):
            raise with_hint(
                ValueError(f"skip pattern {pattern!r} matches the empty string"),
                "A skip pattern consumes at least one character: repeat with `+`.",
            )
        end = "\n" if compiled.flags & re.VERBOSE else ""
        try:
            self._skip: re.Pattern[str] | None = re.compile(f"(?:{pattern}{end})+", flags)
        except re.error as e:
            raise with_hint(
                ValueError(f"skip pattern {pattern!r} cannot repeat: {e}"),
                "Pass a global flag such as `(?i)` as `flags` instead.",
            ) from e

    @classmethod
    def _nothing(cls) -> Skipping:
        """The parsers of `regex()` and `lit()`, which skip nothing."""
        plain = cls.__new__(cls)
        plain._skip = None
        return plain

    def _after_skip(self, text: str, pos: int) -> int:
        if self._skip is not None and (m := self._skip.match(text, pos)):
            return m.end()
        return pos

    def _reading(
        self, name: str, read: Any, regex: re.Pattern[str] | None, value: str | None
    ) -> Parser[str, str]:
        def run(text: Any, s: State) -> ParsingResult[str]:
            end = read(text, s.pos)
            if end is None:
                return ParsingFailure(s.expect(p))
            return ParsingSuccess(text[s.pos : end], s._past_skip(end, self._after_skip(text, end)))

        form = ("text", regex.match if regex else None, value, self._skip)
        p: Parser[Any, str] = Parser._made_leaf(name, run, form)
        return p

    def regex(self, pattern: str, flags: int = 0) -> Parser[str, str]:
        """A parser of the text the regex `pattern`, with its `flags`, matches.

        Give it a name with `Parser.named()` for its errors. Raises `ValueError` for a pattern that
        matches the empty string.
        """
        compiled = re.compile(pattern, flags)
        if compiled.match(""):
            raise with_hint(
                ValueError(f"pattern {pattern!r} matches the empty string"),
                "A match consumes at least one character: repeat with `+` rather than `*`.",
            )

        def read(text: str, start: int) -> int | None:
            m = compiled.match(text, start)
            return m.end() if m is not None and m.end() > start else None

        return self._reading(f"/{pattern}/", read, compiled, None)

    def lit(self, value: str) -> Parser[str, str]:
        """A parser of the text `value`, which it reads where a longer word starts with it too.

        To read a keyword, end the regex with a word boundary: `regex(r"if\\b")`. Raises
        `ValueError` for an empty `value`, and `TypeError` for anything but a `str`.
        """
        if not isinstance(value, str):
            raise with_hint(
                TypeError(f"literal value must be a str, not {type(value).__name__}"),
                "Pass the text to read, as `lit('if')`.",
            )
        if not value:
            raise with_hint(
                ValueError("literal value is empty"), "A literal consumes at least one character."
            )

        def read(text: str, start: int) -> int | None:
            return start + len(value) if text.startswith(value, start) else None

        return self._reading(repr(value), read, None, value)

    @property
    def end(self) -> Parser[str, None]:
        """A parser that succeeds at the end of the text, past what it skips."""

        def run(text: Any, s: State) -> ParsingResult[None]:
            start = self._after_skip(text, s.pos)
            if start >= len(text):
                return ParsingSuccess(None, s._past_skip(s._value_end(0), start))
            return ParsingFailure(s.expect(p))

        p: Parser[Any, None] = Parser._made_leaf("end of input", run, ("text_end", self._skip))
        return p


def skipping(pattern: str, flags: int = 0) -> Skipping:
    """Parsers of text that each skip what the regex `pattern` matches after what they read.

    `pattern`, with its `flags`, is for what may sit between tokens, such as whitespace and
    comments, and it matches as many times as it can. The text may start with it, for any first
    parser of the grammar: the first skip pattern found in the grammar is read before parsing
    starts. Inside a token, such as a string written as a grammar, use `regex()` and `lit()`, which
    skip nothing.

    Raises `ValueError` for a `pattern` that matches the empty string or cannot repeat, and
    `TypeError` for anything but a `str`.

    Examples:
        ```pycon
        >>> from funcparserlib.parser import many, skipping
        >>> spaced = skipping(r"\\s+")
        >>> number = (spaced.regex(r"\\d+") >> int).named("number")
        >>> total = number + many(-spaced.lit("+") + number) + -spaced.end
        >>> total.parse_text(" 1 + 2 ")
        (1, [2])
        >>> total.parse_text("1 +\\n  x")
        Traceback (most recent call last):
            ...
        funcparserlib.parser.NoParseError: 2,3-2,3: got unexpected token: 'x', expected: number

        ```
    """
    return Skipping(pattern, flags)


_plain = Skipping._nothing()


def regex(pattern: str, flags: int = 0) -> Parser[str, str]:
    """A parser of the text the regex `pattern`, with its `flags`, matches, skipping nothing.

    It gives the whole match. Give it a name with `Parser.named()` for its errors. Raises
    `ValueError` for a pattern that matches the empty string.

    Examples:
        ```pycon
        >>> from funcparserlib.parser import finished, lit, regex
        >>> number = regex(r"\\d+").named("number") >> int
        >>> (number + -lit(".") + number + -finished).parse_text("1.25")
        (1, 25)

        ```
    """
    return _plain.regex(pattern, flags)


def lit(value: str) -> Parser[str, str]:
    """A parser of the text `value`, skipping nothing, which reads where a longer word starts with
    it.

    To read a keyword, end the regex with a word boundary: `regex(r"if\\b")`. Raises `ValueError`
    for an empty `value`, and `TypeError` for anything but a `str`.
    """
    return _plain.lit(value)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
