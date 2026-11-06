# Parsing Without Tokens

In [Getting Started](../getting-started/index.md) we split the input into tokens and then parsed the tokens. This works
well when a token is the same thing wherever it appears. Sometimes it isn't:

* A `<` opens a tag in one rule and compares two numbers in another
* A comment holds other comments, and a regex cannot count how deep they go
* You want an error inside a token, where a regex can only fail at its start

For these cases you can leave the tokenizer out. The parsers read the text themselves, at the position the parsing has
reached. We call this a **text grammar**, and a grammar that reads tokens a **token grammar**. Both are `Parser` objects
and combine with the same operators.

## A First Text Grammar

Let's make the parsers with [`Text`](../api/parser.md#reading-text): `regex()` for a regex and `lit()` for a fixed
string. `skip` is a regex for what may sit between two tokens, such as whitespace. Every parser skips it after what it
reads, and the text may also start with it:

```pycon
>>> from funcparserlib.parser import Text, many
>>> spaced = Text(skip=r"\s+")
>>> number = (spaced.regex(r"\d+") >> int).named("number")
>>> total = number + many(-spaced.lit("+") + number) + -spaced.end
>>> total.parse_text(" 1 + 2 + 3 ")
(1, [2, 3])

```

`Parser.parse_text()` takes the text itself. `spaced.end` succeeds at the end of the text, past what `skip` matches, and
`named()` gives a parser the name its errors use.

The functions `regex()` and `lit()` of `funcparserlib.parser` skip nothing, which is what a token that has whitespace
inside needs. Such a grammar ends with `finished`, as a token grammar does. The first `skip` found in the grammar is still
read before parsing starts, so that a document can start with whitespace, even before a first parser that skips nothing:
whitespace at the very start of the text never reaches a parser. Use the `Text` for the grammar, and the functions for the
inside of a token. Open the string with a function, so that the whitespace after the quote belongs to the string, and
close it with the `Text`, so that the whitespace after the string is skipped:

```pycon
>>> from funcparserlib.parser import lit, regex
>>> chars = many(regex(r'[^"\\\n]+') | regex(r"\\."))
>>> string = -lit('"') + chars + -spaced.lit('"') >> "".join
>>> (string + string + -spaced.end).parse_text('"a  b"   "c"  ')
('a  b', 'c')

```

## Alternatives Without an Order

`p1 | p2` keeps the first parser that succeeds. With tokens, the tokenizer chose between `<` and `<=` before the grammar
began, by the order of its specs. Without them the order is in the grammar, and a shorter match hides a longer one:

```pycon
>>> less = spaced.lit("<") | spaced.lit("<=")
>>> (less + number).parse_text("<= 1")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,2-1,2: got unexpected token: '=', expected: number

```

`longest()` runs all the alternatives and keeps the longest match, the first of them when two tie. The order of the
alternatives stops mattering:

```pycon
>>> from funcparserlib.parser import longest
>>> less = longest(spaced.lit("<"), spaced.lit("<="))
>>> (less + number).parse_text("<= 1")
('<=', 1)

```

The functions after `>>` of every alternative run, so an alternative that has side effects runs even when another wins.

## Errors Inside a Token

A text grammar says what it expected, with the line and column of the text. This is the failure of the grammar above:

```pycon
>>> from funcparserlib.parser import NoParseError
>>> source = "1 +\n  x"
>>> try:
...     total.parse_text(source)
... except NoParseError as e:
...     print(e.render(source))
2,3-2,3: got unexpected token: 'x', expected: number
2 |   x
      ^

```

A token written as a grammar fails inside itself. Here is the string from above, left open at the end of the line:

```pycon
>>> try:
...     (string + string + -spaced.end).parse_text('"a" "b\n')
... except NoParseError as e:
...     print(e)
1,7-1,7: got unexpected token: '\n', expected: /[^"\\\n]+/ or /\\./ or '"'

```

[Error Handling](errors.md) has the rest of what an error can do.

## Tokens That Depend on Where They Are

What a token means can depend on the rules around it. A comment that nests is a recursive rule, where a regex could
only count to a fixed depth:

```pycon
>>> from funcparserlib.parser import finished, forward_decl
>>> comment = forward_decl()
>>> comment.define(
...     lit("/*")
...     + many(regex(r"[^/*]+|/(?!\*)|\*(?!/)") | comment)
...     + lit("*/")
... )
>>> gap = many(regex(r"\s+") | comment)
>>> words = many(-gap + regex(r"\w+")) + -gap + -finished
>>> words.parse_text("a /* b /* c */ d */ e")
['a', 'e']

```

A parser that depends on what was read before it is made with `Parser.bind()`. The lines of an outline, where each
level is indented by two spaces more, are the lines of a deeper `item()` after a line:

```pycon
>>> def item(depth):
...     line = regex(rf" {{{2 * depth}}}\w+(?:\n|\Z)")
...     return line.bind(lambda l: many(item(depth + 1)) >> (lambda kids: (l.strip(), kids)))
>>> (many(item(0)) + -finished).parse_text("a\n  b\n    c\n  d\ne\n")
[('a', [('b', [('c', [])]), ('d', [])]), ('e', [])]

```

## Compiling a Text Grammar

An interpreted text grammar is slower than a token grammar. Compiled, the two are close. A compiled grammar gives the
same value and the same errors:

```pycon
>>> compiled = total.compile()
>>> compiled.parse_text("1 + 2")
(1, [2])

```

See [Performance](performance.md) for what compiling costs, and for `try_parse()`. For a grammar of a real language, see
[KDL without a tokenizer](../recipes/kdl.md#without-a-tokenizer).

## Reading Bytes

Bytes are read the same way. The experimental [`funcparserlib.binary`](../api/binary.md) has the parsers of a byte grammar,
`Parser.parse_bytes()` runs it, and `prefixed()`, `counted()` and `switch()` read what depends on a number just read: a
length, a count, a type. A grammar of them compiles as a text grammar does. For a whole format, see
[Parsing MessagePack](../recipes/msgpack.md).
