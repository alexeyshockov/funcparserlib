# Parsing Without Tokens

In [Getting Started](../getting-started/index.md) we split the input into tokens and then parsed the tokens. This works
well when a token is the same thing wherever it appears. Sometimes it isn't:

* A `<` opens a tag in one rule and compares two numbers in another
* A comment holds other comments, and a regexp cannot count how deep they go
* You want an error inside a token, where a regexp can only fail at its start

For these cases you can leave the tokenizer out. The parsers read the text themselves, at the position the parsing has
reached. We call this a **text grammar**, and a grammar that reads tokens a **token grammar**. Both are `Parser` objects
and combine with the same operators.

## A First Text Grammar

Let's make the parsers with [`Text`](../api/text.md): `token()` for a regexp and `literal()` for a fixed string. `skip`
is a regexp for what may sit between two tokens, such as whitespace. Every parser skips it after what it reads, and the
text may also start with it:

```pycon
>>> from funcparserlib.parser import many
>>> from funcparserlib.text import Text
>>> text = Text(skip=r"\s+")
>>> number = (text.token(r"\d+") >> int).named("number")
>>> total = number + many(-text.literal("+") + number) + -text.end
>>> total.parse_text(" 1 + 2 + 3 ")
(1, [2, 3])

```

`Parser.parse_text()` takes the text itself. `text.end` succeeds at the end of the text, as `finished` does for tokens,
and `named()` gives a parser the name its errors use.

A `Text` without `skip` skips nothing, which is what a token that has whitespace inside needs. Keep two of them: the one
with `skip` for the grammar, and one without for the inside of a token. Open the string with the second, so that the
whitespace after the quote belongs to the string, and close it with the first, so that the whitespace after the string
is skipped:

```pycon
>>> inside = Text()
>>> chars = many(inside.token(r'[^"\\\n]+') | inside.token(r"\\."))
>>> string = -inside.literal('"') + chars + -text.literal('"') >> "".join
>>> (string + string + -text.end).parse_text('"a  b"   "c"  ')
('a  b', 'c')

```

## Alternatives Without an Order

`p1 | p2` keeps the first parser that succeeds. With tokens, the tokenizer chose between `<` and `<=` before the grammar
began, by the order of its specs. Without them the order is in the grammar, and a shorter match hides a longer one:

```pycon
>>> less = text.literal("<") | text.literal("<=")
>>> (less + number).parse_text("<= 1")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,2-1,2: got unexpected token: '=', expected: number

```

`longest()` runs all the alternatives and keeps the longest match, the first of them when two tie. The order of the
alternatives stops mattering:

```pycon
>>> from funcparserlib.parser import longest
>>> less = longest(text.literal("<"), text.literal("<="))
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
...     (string + string + -text.end).parse_text('"a" "b\n')
... except NoParseError as e:
...     print(e)
1,7-1,7: got unexpected token: '\n', expected: /[^"\\\n]+/ or /\\./ or '"'

```

[Error Handling](errors.md) has the rest of what an error can do.

## Tokens That Depend on Where They Are

What a token means can depend on the rules around it. A comment that nests is a recursive rule, where a regexp could
only count to a fixed depth:

```pycon
>>> from funcparserlib.parser import forward_decl
>>> blank = Text()
>>> comment = forward_decl()
>>> comment.define(
...     blank.literal("/*")
...     + many(blank.token(r"[^/*]+|/(?!\*)|\*(?!/)") | comment)
...     + blank.literal("*/")
... )
>>> gap = many(blank.token(r"\s+") | comment)
>>> words = many(-gap + blank.token(r"\w+")) + -gap + -blank.end
>>> words.parse_text("a /* b /* c */ d */ e")
['a', 'e']

```

A parser that depends on what was read before it is made with `Parser.bind()`. The lines of an outline, where each
level is indented by two spaces more, are the lines of a deeper `item()` after a line:

```pycon
>>> outline = Text()
>>> def item(depth):
...     line = outline.token(rf" {{{2 * depth}}}\w+(?:\n|\Z)")
...     return line.bind(lambda l: many(item(depth + 1)) >> (lambda kids: (l.strip(), kids)))
>>> (many(item(0)) + -outline.end).parse_text("a\n  b\n    c\n  d\ne\n")
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

Bytes are read the same way. A [`Binary`](../api/binary.md) makes the parsers of a byte grammar, `Parser.parse_bytes()`
runs it, and `prefixed()`, `counted()` and `switch()` read what depends on a number just read: a length, a count, a
type. A grammar of them compiles as a text grammar does. For a whole format, see
[Parsing MessagePack](../recipes/msgpack.md).
