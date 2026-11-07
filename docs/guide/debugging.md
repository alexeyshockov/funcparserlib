# Debugging a Grammar

When a grammar does not do what you expect, the first thing to find out is which part is wrong: the tokenizer, one rule,
or a function after `>>`. Each step here looks at one of them.

Let's use a tokenizer for numeric expressions again:

```pycon
>>> from funcparserlib.lexer import Token, TokenSpec, make_tokenizer
>>> from funcparserlib.parser import Parser, finished, many, maybe, tok

>>> tokenizer = make_tokenizer(
...     [
...         TokenSpec("whitespace", r"\s+"),
...         TokenSpec("int", r"\d+"),
...         TokenSpec("op", r"[*][*]|[*]"),
...     ],
...     skip={"whitespace"},
... )


>>> def tokenize(s: str) -> list[Token]:
...     return list(tokenizer(s))


>>> def op(name: str) -> Parser[Token, str]:
...     return tok("op", name)


>>> number = (tok("int") >> int).named("number")

```

## Look at the Tokens First

A grammar can only be as right as the tokens it reads, so print them. Here the order of the alternatives in a spec is
wrong, and `**` comes out as two tokens:

```pycon
>>> wrong = make_tokenizer(
...     [
...         TokenSpec("whitespace", r"\s+"),
...         TokenSpec("int", r"\d+"),
...         TokenSpec("op", r"[*]|[*][*]"),
...     ],
...     skip={"whitespace"},
... )

>>> list(wrong("2 ** 3"))
[Token('int', '2'), Token('op', '*'), Token('op', '*'), Token('int', '3')]

>>> tokenize("2 ** 3")
[Token('int', '2'), Token('op', '**'), Token('int', '3')]

```

## Run One Rule on Its Own

Every rule of a grammar is a parser, so you can call `parse()` on any of them with a small input. Start from the rules at
the bottom and go up until one gives a value you did not expect:

```pycon
>>> power = number + many(-op("**") + number)
>>> term = power + many(-op("*") + power)

>>> number.parse(tokenize("2"))
2

>>> power.parse(tokenize("2 ** 3"))
(2, [3])

>>> term.parse(tokenize("2 ** 3 * 4"))
(2, [3], [(4, [])])

```

## Make Sure the Whole Input Is Read

A parser stops where it can't go on, and without `finished` that is not an error. If a grammar returns less than you gave
it, this is the first thing to check:

```pycon
>>> term.parse(tokenize("2 * 3 *"))
(2, [], [(3, [])])

>>> (term + -finished).parse(tokenize("2 * 3 *"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: got unexpected end of input, expected: number

```

## Print What a Parser Gives

To see a value in the middle of a grammar, put a function after `>>` that prints it and returns it as it is:

```pycon
>>> def show[T](value: T) -> T:
...     print(value)
...     return value


>>> (number >> show).parse(tokenize("2"))
2
2

```

The first line comes from `show()`, and the second one is the result of `parse()`. You can also stop in `show()` with
`breakpoint()`. Such a function runs each time its parser succeeds, also on alternatives that fail later, and
[a compiled grammar runs it again](performance.md#know-what-compiling-costs) when a parse fails.

## Log What Is Tried

For the order in which parsers run, turn on the parsing log. The flag `funcparserlib.parser.debug` is read when a parser
is built, so set it before your grammar, and give the parsers you want to see a name with `named()`:

```pycon
>>> import logging
>>> import sys

>>> import funcparserlib.parser

>>> logger = logging.getLogger("funcparserlib")
>>> handler = logging.StreamHandler(sys.stdout)
>>> logger.addHandler(handler)
>>> logger.setLevel(logging.DEBUG)
>>> funcparserlib.parser.debug = True

>>> number = (tok("int") >> int).named("number")
>>> pair = (number + number).named("pair")
>>> (pair | number).parse(tokenize("1"))
trying pair or number
trying pair
trying number
trying int
trying int
*matched* Token('int', '1'), new state = (1, 1)
trying number
trying int
trying int
trying number
trying int
trying int
*matched* Token('int', '1'), new state = (1, 1)
1

>>> funcparserlib.parser.debug = False
>>> logger.removeHandler(handler)

```

`*matched*` is a token that was read, and the second `number` after it is the one that fails at the end of the input.
The log is the same with a compiled grammar, as with the flag set it runs the parsers. The way to enable the log may
change in future versions.

## Read the Generated Code

A compiled grammar gives what its parser gives, so when the value is wrong, debug the grammar without `compile()`. To see
what was generated, give `compile()` a file, as [Performance](performance.md#keep-the-generated-code-in-charge) shows.

## Find a Rule That Reads Nothing

A rule that reaches itself before reading anything raises `RecursionError`, which
[Tips and Tricks](../getting-started/tips-and-tricks.md#watch-out-for-left-recursion) explains. A `many()` of a parser that
succeeds without reading anything would go on forever, so it raises `ValueError` when that happens:

```pycon
>>> many(maybe(op("*"))).parse(tokenize("1"))
Traceback (most recent call last):
    ...
ValueError: [ '*' ] matched no tokens inside many()
many() repeats a parser until it fails, and one that matches nothing never fails: repeat the parser that maybe(), many() or pure() wraps.

```

The error says what to do: `many()` already accepts no items at all, so take the `maybe()` out.

## Next

[Error Handling](errors.md) shows how to give the people who use your grammar errors they can act on.
