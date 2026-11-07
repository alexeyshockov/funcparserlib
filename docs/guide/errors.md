# Error Handling

Most input a little language sees is written by a person, and people make mistakes. This chapter shows what
funcparserlib tells you about a mistake, and how to make that better for your users.

Let's use a tokenizer for numeric expressions again, this time with lines:

```pycon
>>> from funcparserlib.lexer import LexerError, Token, TokenSpec, make_tokenizer
>>> from funcparserlib.parser import NoParseError, ParseErrors, Parser, finished, forward_decl, many, maybe, tok

>>> tokenizer = make_tokenizer(
...     [
...         TokenSpec("whitespace", r"[ \t]+"),
...         TokenSpec("newline", r"\n"),
...         TokenSpec("int", r"\d+"),
...         TokenSpec("op", r"[+*/()]"),
...     ],
...     skip={"whitespace"},
... )


>>> def tokenize(s: str) -> list[Token]:
...     return list(tokenizer(s))


>>> def op(name: str) -> Parser[Token, str]:
...     return tok("op", name)


>>> number = (tok("int") >> int).named("number")
>>> expr = forward_decl()
>>> term = number | (-op("(") + expr + -op(")"))
>>> expr.define(term + many(-op("+") + term))
>>> document = expr + -finished

```

## Read an Error

When tokens don't match the grammar, `Parser.parse()` raises `NoParseError`:

```pycon
>>> document.parse(tokenize("1 + * 2"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected token: '*', expected: number or '('

```

The error starts with where the token is, as `line,column-line,column`. Then it says what was found, and what would have
been accepted instead.

Parsing goes back and tries other alternatives many times along the way. The error reports the furthest place any of
them reached, and everything that failed there, so it points at the longest part of the input that made sense.

`NoParseError.render()` shows the place in the source, which is usually what you want to show to a user:

```pycon
>>> source = "1 + * 2"
>>> try:
...     document.parse(tokenize(source))
... except NoParseError as e:
...     print(e.render(source))
...     print(e.expected)
1,5-1,5: got unexpected token: '*', expected: number or '('
1 | 1 + * 2
        ^
('number', "'('")

```

The parts of the message are also attributes of the error: `token`, `expected`, `context` and `description`.

## Handle Every Failure in One Place

Text can fail before it gets to the parser: the tokenizer raises `LexerError` for text that no token spec matches. All
the errors have `render()`, so one handler can show them all:

```pycon
>>> def report(source: str) -> None:
...     try:
...         print(document.parse(tokenize(source)))
...     except (LexerError, NoParseError, ParseErrors) as e:
...         print(e.render(source))


>>> report("1 + 2")
(1, [2])

>>> report("1 + x")
1,5: cannot tokenize data
1 | 1 + x
        ^

>>> report("1 + (2")
got unexpected end of input, expected: '+' or ')'
1 | 1 + (2
          ^

```

We will meet `ParseErrors` [in a moment](#report-errors-of-meaning).

## Say What Is Being Parsed

You have already seen in [Tips and Tricks](../getting-started/tips-and-tricks.md#name-alternative-parsers-for-better-error-messages)
how `Parser.named()` replaces the parts of a parser with one name in an error. `Parser.context()` does something else:
it says which part of the grammar an error is in, so the error ends with `in` and the name:

```pycon
>>> group = (-op("(") + number + -op(")")).context("group")

>>> group.parse(tokenize("(1 + 2)"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,4-1,4: got unexpected token: '+', expected: ')' in group

```

## Stop Going Back

Going back is usually what you want, but it can hide a mistake. `many()` stops at the first failure, so this parser
reads the `* 3` and quietly leaves the rest of the input alone:

```pycon
>>> product = number + many(-op("*") + number)

>>> product.parse(tokenize("2 * 3 * )"))
(2, [3])

```

A grammar that ends with `finished` would report the `)`. Inside a larger grammar, the rest of the rules may accept what
is left, or the error may name more alternatives than make sense here. Once we have seen a `*`, nothing but a number can
follow it, and `commit()` says so:

```pycon
>>> from funcparserlib.parser import commit

>>> product = number + many(commit(-op("*") + number))

>>> product.parse(tokenize("2 * 3 * )"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,9-1,9: got unexpected token: ')', expected: number

```

Once a committed parser has read a token, its failure is final: `p1 | p2` doesn't try the next alternative, and `many()`
and `maybe()` don't go back. A failure at its first token works as usual, so `many()` above still ends where it should.

## Report Errors of Meaning

Some input matches the grammar and is still wrong. `Parser.validated()` takes a function that makes the value, or raises
`Invalid` when it can't. The input was read, so parsing goes on and one run finds every error. At the end of the parse
they come together in `ParseErrors`, an `ExceptionGroup` of `NoParseError`:

```pycon
>>> from funcparserlib.parser import Invalid, some

>>> def divisor_of(n: int) -> int:
...     if n == 0:
...         raise Invalid("division by zero")
...     return n

>>> divisor = number.validated(divisor_of)
>>> ratio = number + many(-op("/") + divisor) + -finished

>>> source = "1 / 0 / 2 / 0"
>>> try:
...     ratio.parse(tokenize(source))
... except ParseErrors as group:
...     print(group.render(source))
1,5-1,5: division by zero
1 | 1 / 0 / 2 / 0
        ^
<BLANKLINE>
1,13-1,13: division by zero
1 | 1 / 0 / 2 / 0
                ^

```

A rejected value has no value: no function after `>>` runs over it, so your code never sees it. To go on with a value that
is real and only breaks a rule, raise `Invalid(message, keep=True)`. To give a broken part a value of your own, use
`Parser.recover()`, as `p.recover(default=...)`.

The function decides what counts as invalid: catch the exception you mean and raise `Invalid`, so that a bug in it stays
a bug. To parse a value with another grammar, call that grammar in the function. Its `NoParseError` rejects the value,
and `expected` names what should have been there:

```pycon
>>> from funcparserlib.parser import lit, regex

>>> version = regex(r"\d+") + -lit(".") + regex(r"\d+") + -finished
>>> parsed = some(lambda t: True).validated(version.parse_text, "a version")
>>> parsed.parse(["1.2"])
('1', '2')

```

To decide among alternatives, use `Parser.when()`: a rejected value does not make `|` try the next one.

## Report Every Error in One Run

A parse stops at its first syntax error. For a file that a person fixes and runs again, it is better to report all of
them. `Parser.recover()` with `until` records a failure and goes on from a point you choose, here the next line:

```pycon
>>> newline = tok("newline")
>>> line = (expr + -newline).recover(until=newline)
>>> lines = many(line + -maybe(newline)) + -finished

>>> try:
...     lines.parse(tokenize("1 + 2\n3 4\n5 + (6 7\n8 + 9\n"))
... except ParseErrors as group:
...     for e in group.exceptions:
...         print(e)
2,3-2,3: got unexpected token: '4', expected: '+' or newline
3,8-3,8: got unexpected token: '7', expected: '+' or ')'

```

A parser recovers from a failure past its first token, so `many()` still ends where it would.

## Describe Your Own Tokens

An error knows the text and the position of a `Token`. If your tokens come from another library, tell it how to show them
with the `describe` argument of `Parser.parse()`:

```pycon
>>> from funcparserlib.parser import TokenDescription, some

>>> words = [{"text": "one", "column": 1}, {"text": "2", "column": 5}]


>>> def describe(word: dict) -> TokenDescription:
...     column = word["column"]
...     return TokenDescription(word["text"], (1, column), (1, column + len(word["text"]) - 1))


>>> word = some(lambda t: t["text"].isalpha()).named("word")

>>> (word + word).parse(words, describe=describe)
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected token: 2, expected: word

```

## Watch Out for Deep Nesting

Parsers recurse with the nesting of the input, so text nested deeply enough raises `RecursionError`, which is not a
`NoParseError`. If your input comes from outside your program, catch it as well. A
[compiled grammar](performance.md) uses less of the stack, so it reads deeper input.

## Next

Errors stay the same when you [compile a grammar](performance.md), and they are what makes a failed parse slower there.
