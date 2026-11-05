# Tips and Tricks

Let's use the tokenizer we have defined previously for our examples in this chapter:

```pycon
>>> from typing import List
>>> from funcparserlib.lexer import make_tokenizer, TokenSpec, Token
>>> from funcparserlib.parser import tok, Parser, many, forward_decl, finished


>>> def tokenize(s: str) -> List[Token]:
...     specs = [
...         TokenSpec("whitespace", r"\s+"),
...         TokenSpec("float", r"[+\-]?\d+\.\d*([Ee][+\-]?\d+)*"),
...         TokenSpec("int", r"[+\-]?\d+"),
...         TokenSpec("op", r"(\*\*)|[+\-*/()]"),
...     ]
...     tokenizer = make_tokenizer(specs)
...     return [t for t in tokenizer(s) if t.type != "whitespace"]


>>> def op(name: str) -> Parser[Token, str]:
...     return tok("op", name)

```

## Name Alternative Parsers for Better Error Messages

Consider the following grammar:

```pycon
>>> number = (tok("int") >> int) | (tok("float") >> float)
>>> paren = -op("(") + number + -op(")")
>>> mul = number + op("*") + number
>>> expr = paren | mul

```

When a parser fails to parse its input, it usually reports the token it expects:

```pycon
>>> paren.parse(tokenize("(1"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: got unexpected end of input, expected: ')'

```

If there were several parsing alternatives, the parser will report an error after the longest successfully parsed
sequence:

```pycon

>>> expr.parse(tokenize("1 + 2"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,3-1,3: got unexpected token: '+', expected: '*'

```

If there were several parsing alternatives and all of them failed at the current token, the error lists what each of
them expected there:

```pycon
>>> number.parse(tokenize("*"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,1-1,1: got unexpected token: '*', expected: int or float

>>> expr.parse(tokenize("+"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,1-1,1: got unexpected token: '+', expected: '(' or int or float

```

The error names the parsers that failed, and a list of every alternative of a big grammar is hard to read. A name given
with [`Parser.named(name)`](../api/parser.md#funcparserlib.parser.Parser.named) is a label: when the parser fails at its
first token, the error names it instead of its parts. The naming style is up to you. For example:

```pycon
>>> number = ((tok("int") >> int) | (tok("float") >> float)).named("number")
>>> paren = -op("(") + number + -op(")")
>>> mul = number + op("*") + number
>>> expr = (paren | mul).named("number or '('")

```

Test it:

```pycon
>>> number.parse(tokenize("*"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,1-1,1: got unexpected token: '*', expected: number

>>> expr.parse(tokenize("+"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,1-1,1: got unexpected token: '+', expected: number or '('

```

## How to Handle Conflicting Alternatives

If one of the parsing alternatives is a subpart of another one, then you should put the longest alternative first.
Otherwise parsing the shorter one will make another one unreachable:

```pycon
>>> p = (number + number) | (number + number + number)

>>> p.parse(tokenize("1 2 3"))
(1, 2)

```

Parse the longest alternative first:

```pycon
>>> p = (number + number + number) | (number + number)

>>> p.parse(tokenize("1 2 3"))
(1, 2, 3)

>>> p.parse(tokenize("1 2"))
(1, 2)

```

## Parse a Shared Prefix Once

When an alternative fails, `|` goes back to where it started and tries the next one. Alternatives that start the same
way parse their shared start once for each alternative tried:

```pycon
>>> from funcparserlib.parser import maybe

>>> p = (number + -op("*") + number) | number

>>> p.parse(tokenize("1"))
1

```

Here `number` runs twice on the `1`. It is cheap, but a shared start can be a large part of a grammar, and if it is
recursive the repeated work multiplies with each level of nesting.

Parse the shared start once and make the rest optional:

```pycon
>>> p = number + maybe(-op("*") + number)

>>> p.parse(tokenize("2 * 3"))
(2, 3)

>>> p.parse(tokenize("1"))
(1, None)

```

The result changes shape, so a function after `>>` has to tell the two cases apart.
[Parsing KDL](../recipes/kdl.md#the-grammar) does this for an entry of a node.

## Compile a Grammar That Parses a Lot

[`Parser.compile()`](../api/parser.md#funcparserlib.parser.Parser.compile) generates Python functions from a finished
grammar. They run several times faster than the parsers do, and `parse()` works the same way:

```pycon
>>> expr = (number + many(-op("*") + number) + -finished).compile()

>>> expr.parse(tokenize("2 * 3 * 4"))
(2, [3, 4])

```

The functions keep no record of what was expected, so a parse that fails runs again in the parser it was compiled from.
Errors are the same as without compiling:

```pycon
>>> expr.parse(tokenize("2 * *"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected token: '*', expected: number

```

This makes a failed parse slower, and calls the functions you gave to `>>` twice, so compile a grammar when most of what
it reads is valid and its `>>` functions only compute a value.

Given a tokenizer, a compiled grammar parses text itself. As long as the grammar reads tokens through `tok()`, this
makes no `Token` objects, which takes less time and much less memory. An error says what was expected, also at text the
tokenizer cannot read:

```pycon
>>> tokenizer = make_tokenizer(
...     [
...         TokenSpec("whitespace", r"\s+"),
...         TokenSpec("int", r"\d+"),
...         TokenSpec("op", r"[*]"),
...     ],
...     skip={"whitespace"},
... )
>>> digit = tok("int") >> int
>>> product = (digit + many(-op("*") + digit) + -finished).compile(tokenizer)

>>> product.parse_text("2 * 3 * 4")
(2, [3, 4])

>>> product.parse_text("2 * x")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected text: 'x', expected: int

```

## Watch Out for Left Recursion

There are certain kinds grammar rules you cannot use with `funcparserlib`. These are the rules that contain recursion in
their leftmost parts. These rules lead to infinite recursion during parsing, that results in a `RecursionError`
exception.

For example, we want to define an expression `expr` either a multiplication operator `mul` or a `number`. We also want
an expression to be a sequence of an expression `expr`, followed by an operator `"**"`, followed by another expression
`expr`:

```pycon
>>> expr = forward_decl()
>>> mul = expr + op("*") + expr
>>> expr.define(mul | number)

```

This looks reasonable at the first glance, but it contains left recursion. In order to parse the first token for `expr`,
we need to parse the first token for `mul`, for that we need to parse the first token for `expr`, and so on. This left
recursion in your grammar results in a stack overflow exception:

```pycon
>>> expr.parse(tokenize("1 * 2"))
Traceback (most recent call last):
    ...
RecursionError: maximum recursion depth exceeded

```

You should think how to re-write your grammar to avoid left-recursive definitions. In our case of several multiplication
operators we really want a number, followed by zero or more pairs of `*` and number:

```pycon
>>> expr = forward_decl()
>>> mul = number + many(op("**") + number)
>>> expr.define(mul)

```

Test it:

```pycon
>>> expr.parse(tokenize("1 ** 2"))
(1, [('**', 2)])


>>> expr.parse(tokenize("3"))
(3, [])

```

Remember that your parsers have to consume at least one token from the input before going into recursive definitions.
