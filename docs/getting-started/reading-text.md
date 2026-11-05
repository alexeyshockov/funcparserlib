# Reading Text Directly

A grammar of `tok()` reads the list of tokens that a [tokenizer](tokenizing.md) made before the parsing began. This is
the right tool when the tokens are the same wherever they are. A grammar of [`Text`](../api/text.md) parsers has no
tokenizer: each parser reads the text at the position the parsing has reached. Use it when:

* a token means something different in different places, such as a `<` that opens a tag in one rule and compares two
  numbers in another;
* a token has a structure of its own that a regexp cannot count, such as a comment inside a comment;
* the errors matter: a token written as a grammar fails inside itself, where a regexp only fails at its start.

A **token grammar** is defined by its tokenizer and a **text grammar** by the text its parsers read, and there is no
list of token specs to keep in order. Both are `Parser` objects and combine with the same operators.

## The first grammar

`Text` makes the parsers, `token()` for a regexp and `literal()` for a fixed string. `skip` is a regexp for what may sit
between two tokens, such as whitespace, and every parser skips it after what it reads. The text may also start with it:

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

## Alternatives without an order

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

## Errors

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

An error at the end of the text has a line and column too. Positions are worked out only when an error is made, so a
parse that succeeds pays nothing for them.

A token written as a grammar fails inside itself. The string above, left open at the end of the line:

```pycon
>>> try:
...     (string + string + -text.end).parse_text('"a" "b\n')
... except NoParseError as e:
...     print(e)
1,7-1,7: got unexpected token: '\n', expected: /[^"\\\n]+/ or /\\./ or '"'

```

## Context

What a token means can depend on where it is, because the rules around it do. A comment that nests is a recursive rule,
where a regexp could only count to a fixed depth:

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

## Running it: `compile()`, `parse_text()` and `try_parse()`

`Parser.parse_text()` is the interpreter. `Parser.compile()` makes a `CompiledParser` that runs the same grammar faster
and gives the same value and the same errors:

```pycon
>>> compiled = total.compile()
>>> compiled.parse_text("1 + 2")
(1, [2])

```

A program picks the method of the compiled parser by what it wants from a failure:

* `parse_text()` gives the full error. A parse that fails runs again to make it, which costs more than the failed pass.
* `try_parse()` gives `None` and does not make an error. Use it for input that is often invalid when the reason does not
  matter, such as a request that can only be logged and rejected: a parse that fails stops where it fails.

```pycon
>>> compiled.try_parse("1 + x") is None
True

```

Two things to keep in mind with `try_parse()`:

* A grammar whose value can be `None` cannot tell that value from a failure. If the grammar has such a value, use
  `parse_text()`. Text nested too deeply gives `None` as well, as it would raise `RecursionError`.
* A function after `>>` runs again when `parse_text()` fails on a compiled parser, since the error is made by parsing
the text a second time. `try_parse()` runs it once, up to the failure, but for a failed `check()` or `commit()`:

```pycon
>>> calls = []
>>> def count(digits):
...     calls.append(digits)
...     return int(digits)
>>> digit = text.token(r"\d+") >> count
>>> pair = (digit + digit + -text.end).compile()
>>> pair.try_parse("1 2 x"), calls
(None, ['1', '2'])
>>> calls.clear()
>>> pair.parse_text("1 2 x")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected token: 'x', expected: end of input
>>> calls
['1', '2', '1', '2']

```

A function that must not run twice belongs after the parse, on the value it returned.

For a grammar of a real language, see [KDL without a tokenizer](../recipes/kdl.md#without-a-tokenizer).

## Every Error in One Run

A parse stops at its first error. For a configuration read at startup, it is better to report all of them, so that one
restart fixes them. [`Parser.recover()`](../api/parser.md#funcparserlib.parser.Parser.recover) records a failure and
goes on from a point you choose, here the next line. The errors come together in a `ParseErrors`:

```pycon
>>> from funcparserlib.parser import ParseErrors, many, maybe
>>> from funcparserlib.text import Text

>>> text = Text(skip=r"[ \t]+")
>>> newline = text.token(r"\n+").named("newline")
>>> key = text.token(r"[a-z_]+").named("key")
>>> value = text.token(r"\d+").named("number")
>>> setting = (key + -text.literal("=") + value).recover(until=newline)
>>> config = many(setting + -maybe(newline)) + -text.end

>>> source = "port = 8080\nhost = ?\nname 1\nretries = 3\n"
>>> try:
...     config.parse_text(source)
... except ParseErrors as group:
...     for e in group.exceptions:
...         print(e)
2,8-2,8: got unexpected token: '?', expected: number
3,6-3,6: got unexpected token: '1', expected: '='

```

A parser recovers from a failure past its first token, so `many()` still ends where it would.

## Reading Bytes

Bytes are read the same way. A [`Binary`](../api/binary.md) makes the parsers of a byte grammar, `Parser.parse_bytes()`
runs it, and `prefixed()`, `counted()` and `switch()` read what depends on a number just read: a length, a count, a
type. A grammar of them compiles as a text grammar does. For a whole format, see
[Parsing MessagePack](../recipes/msgpack.md).
