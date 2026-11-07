# Positions and Matches

A parse gives you values. Sometimes you also need to know where they came from, to report on a value after parsing or to
go back to the text it was read from. And sometimes a grammar describes only some parts of a text, and you want to find
those parts or to rewrite them.

## Keep Where a Value Came From

[`Parser.located()`](../api/parser.md#funcparserlib.parser.Parser.located) gives the value of a parser together with where
it was read from. It works on any parser, whatever its value is:

```pycon
>>> from funcparserlib.parser import Located, commit, lit, many, regex, skipping

>>> spaced = skipping(r"\s+")
>>> name = spaced.regex(r"[a-z]+").named("name")
>>> number = (spaced.regex(r"\d+") >> int).named("number")
>>> setting = (name + -spaced.lit("=") + number).located()

>>> text = "width = 80  height = 24"
>>> many(setting).parse_text(text)
[Located(value=('width', 80), start=0, end=10), Located(value=('height', 24), start=12, end=23)]

```

In a text, `start` and `end` are character offsets, so slicing gives what the value was read from. A span ends where its
value ends, before the whitespace `spaced` skipped after it:

```pycon
>>> text[0:10]
'width = 80'

```

Your own class can keep the span with the value:

```pycon
>>> from dataclasses import dataclass


>>> @dataclass(frozen=True)
... class Setting:
...     name: str
...     value: int
...     start: int
...     end: int


>>> def to_setting(m: Located[tuple[str, int]]) -> Setting:
...     return Setting(m.value[0], m.value[1], m.start, m.end)


>>> (setting >> to_setting).parse_text("width = 80")
Setting(name='width', value=80, start=0, end=10)

```

In a token grammar, `start` and `end` are indexes of tokens, `end` being past the last one. The tokens have the line and
the column:

```pycon
>>> from funcparserlib.lexer import TokenSpec, make_tokenizer
>>> from funcparserlib.parser import tok

>>> tokenizer = make_tokenizer(
...     [TokenSpec("space", r"\s+"), TokenSpec("name", r"[a-z]+"), TokenSpec("number", r"\d+")],
...     skip={"space"},
... )
>>> tokens = list(tokenizer("width\n  80"))
>>> m = (tok("name") + tok("number")).located().parse(tokens)
>>> m
Located(value=('width', '80'), start=0, end=2)
>>> tokens[m.start].start, tokens[m.end - 1].end
((1, 1), (2, 4))

```

Bound to its tokenizer with [`tokenized()`](../api/parser.md#funcparserlib.parser.tokenized), a token grammar reads the
text, and its spans are character offsets, as in a grammar without a tokenizer:

```pycon
>>> from funcparserlib.parser import tokenized

>>> tokenized((tok("name") + tok("number")).located(), tokenizer).parse_text("width\n  80")
Located(value=('width', '80'), start=0, end=10)

```

## Turn an Offset Into a Line and a Column

[`Lines`](../api/util.md#funcparserlib.util.Lines) is built once from a text, and then finds each position quickly. Use
one for all the positions of a text, as finding them in the text each time takes much longer on a large one:

```pycon
>>> from funcparserlib.util import Lines

>>> lines = Lines("width = 80\nheight = 24")
>>> lines.place(11)
(2, 1)
>>> lines.offset((2, 1))
11

```

`offset()` also gives token grammars what their tokens do not have: `Token.start` is a place, and this is its offset.

## Find Matches in a Text

[`Parser.scan()`](../api/parser.md#funcparserlib.parser.Parser.scan) tries a grammar at each position of a text, and
gives what it finds from left to right, without overlaps. The grammar does not have to describe the rest of the text:

```pycon
>>> version = regex(r"\d+") + -lit(".") + regex(r"\d+")

>>> for m in version.scan("From 1.2 to 3.14, see above."):
...     print(m.value, m.start, m.end)
('1', '2') 5 8
('3', '14') 12 16

```

It reads the text as far as you take matches, so the first of them costs only what it takes to reach it:

```pycon
>>> next(version.scan("From 1.2 to 3.14, see above.")).value
('1', '2')

```

A token grammar finds matches bound to its tokenizer. Text no token spec matches is text where the grammar fails, so
the scan goes past it:

```pycon
>>> pair = tokenized(tok("name") + tok("number"), tokenizer)
>>> [(m.value, m.start, m.end) for m in pair.scan("Set width 80, then height 24!")]
[(('width', '80'), 4, 12), (('height', '24'), 19, 28)]

```

It starts its tokens at each position it tries, so a match can start inside what the tokens of the whole text make
one word:

```pycon
>>> [(m.start, m.end) for m in tokenized(tok("name", "if"), tokenizer).scan("elif if")]
[(2, 4), (5, 7)]

```

## Rewrite the Matches

[`Parser.replace()`](../api/parser.md#funcparserlib.parser.Parser.replace) gives the text with each match replaced by what
your function returns for it. Everything else stays as it was, to the character:

```pycon
>>> def bump(m: Located[tuple[str, str]]) -> str:
...     major, minor = m.value
...     return f"{major}.{int(minor) + 1}"


>>> version.replace("From 1.2 to   3.14.", bump)
'From 1.3 to   3.15.'

```

[Migrating a File Format](../recipes/migration.md) is a whole program built on it.

## Stop at What Is Broken

A grammar that fails at a position has no match there, so a construct that is nearly right is simply passed over. When
that is a mistake, use `commit()`: what starts as a match has to be one, and the parse stops with the usual error:

```pycon
>>> link = commit(-lit("[[") + regex(r"[a-z]+") + -lit("]]"))

>>> [m.value for m in link.scan("see [[intro]]")]
['intro']

>>> list(link.scan("see [[intro"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,12-1,12: got unexpected end of input, expected: ']]'

```

A value that [`Parser.validated()`](errors.md#report-errors-of-meaning) rejects raises in the same way. Matches found before
the error have already been given, while `replace()` gives no text at all.

## Know What They Cost

A grammar that doesn't use these pays nothing for them. One that does pays for each located value:

* A compiled grammar with a `located()` at every value takes about a third more time than one without
* `scan()` and `replace()` of a compiled grammar run the parser it was compiled from
* Inside a `prefixed()` region, positions are relative to the region
