# `funcparserlib.parser` — Functional parsing combinators

<!-- markdownlint-disable MD050 -- the `:::` identifiers name dunder methods, not bold text -->

::: funcparserlib.parser
    options:
        show_root_heading: false

::: funcparserlib.parser.Parser

::: funcparserlib.parser.Parser.parse

::: funcparserlib.parser.ForwardDeclParser.define

::: funcparserlib.parser.Parser.named

## Primitive Parsers

::: funcparserlib.parser.tok

::: funcparserlib.parser.a

::: funcparserlib.parser.some

::: funcparserlib.parser.forward_decl

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

::: funcparserlib.parser.skip

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
