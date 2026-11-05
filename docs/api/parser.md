# `funcparserlib.parser` — Functional parsing combinators

<!-- markdownlint-disable MD050 -- the `:::` identifiers name dunder methods, not bold text -->

::: funcparserlib.parser
    options:
        show_root_heading: false

::: funcparserlib.parser.Parser

::: funcparserlib.parser.Parser.parse

::: funcparserlib.parser.Parser.parse_text

::: funcparserlib.parser.Parser.parse_bytes

::: funcparserlib.parser.Parser.compile

::: funcparserlib.parser.CompiledParser

::: funcparserlib.parser.CompiledParser.parse

::: funcparserlib.parser.CompiledParser.parse_text

::: funcparserlib.parser.CompiledParser.parse_bytes

::: funcparserlib.parser.CompiledParser.try_parse

::: funcparserlib.parser.ForwardDeclParser.define

::: funcparserlib.parser.Parser.named

## Primitive Parsers

::: funcparserlib.parser.tok

::: funcparserlib.parser.a

::: funcparserlib.parser.some

::: funcparserlib.parser.forward_decl

::: funcparserlib.parser.parser

### `finished`

A parser that throws an exception if there are any unparsed tokens left in the sequence.

Type: `EndOfInput`, a `Parser[Any, None]`

**Examples:**

```pycon
>>> from funcparserlib.parser import a, finished
>>> expr = a("x") + finished
>>> expr.parse("x")
('x', None)

```

```pycon
>>> expr = a("x") + finished
>>> expr.parse("xy")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: got unexpected token: 'y', expected: end of input

```

## Parser Combinators

::: funcparserlib.parser.Parser.__add__

::: funcparserlib.parser.Parser.__neg__

::: funcparserlib.parser.Parser.__or__

::: funcparserlib.parser.Parser.__rshift__

::: funcparserlib.parser.maybe

::: funcparserlib.parser.many

::: funcparserlib.parser.oneplus

::: funcparserlib.parser.longest

::: funcparserlib.parser.skip

::: funcparserlib.parser.Parser.map

::: funcparserlib.parser.Parser.starmap

::: funcparserlib.parser.Parser.when

::: funcparserlib.parser.Parser.but_not

::: funcparserlib.parser.Parser.__invert__

::: funcparserlib.parser.anything_but

## Reading by a Number Just Read

The parsers of a [`Binary`](binary.md) read what depends on a number just read: a length, a count, a type.

::: funcparserlib.parser.prefixed

::: funcparserlib.parser.counted

::: funcparserlib.parser.switch

## Error Reporting

When parsing fails, `NoParseError` says where parsing stopped, what it found there and what every
parser that failed at that position expected. A name given with `Parser.named()` is a label: the
error names it instead of what the parser is made of. `commit()` makes a failure final once a
parser has read a token, `Parser.context()` names the part of the grammar an error is in, and
`Parser.check()` records an error without stopping the parse.

::: funcparserlib.parser.NoParseError

::: funcparserlib.parser.NoParseError.render

::: funcparserlib.parser.ParseErrors

::: funcparserlib.parser.TokenDescription

::: funcparserlib.parser.commit

::: funcparserlib.parser.Parser.context

::: funcparserlib.parser.Parser.check

::: funcparserlib.parser.Parser.recover

## Extra: Parser Monad

As a functional programmer, you might be pleased to know, that parsers in funcparserlib
form _a monad_ with `Parser.bind()` as `>>=` and `pure()` as `return`.

We could have expressed other parsing combinators in terms of `bind()`, but would be
inefficient in Python:

```python
# noinspection PyUnresolvedReferences
class Parser:
    def __add__(self, other):
        return self.bind(lambda x: other.bind(lambda y: pure((x, y))))

    def __rshift__(self, other):
        return self.bind(lambda x: pure(x))
```

::: funcparserlib.parser.Parser.bind

::: funcparserlib.parser.pure
