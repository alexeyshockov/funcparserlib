# Parsing KDL

[KDL](https://kdl.dev/) is a document language of nodes, each with arguments, properties and children. This recipe
parses all of KDL 2.0 with funcparserlib alone: `make_tokenizer()` reads the text, and a grammar that follows the
[spec](https://kdl.dev/spec/) rule by rule builds a tree. The whole program is `examples/kdl.py`; every block below is
included from it.

```pycon
>>> from examples.kdl import parse
>>> [server] = parse("""
... (app)server "web" port=8080 {
...     /- tls #true    // a slashdash comments out a whole node
...     route "/api" timeout=1.5
... }
... """)
>>> server.name, server.type, server.args[0].value, server.props["port"].value
('server', 'app', 'web', 8080)
>>> [route.name for route in server.children]
['route']

```

## The tree

```python
--8<-- "examples/kdl.py:tree"
```

A number with a fraction or an exponent is a `Decimal`, so `1.23E+1000` keeps its value.

## The tokenizer

Most recipes drop whitespace tokens before parsing. KDL needs them: a newline ends a node, and `a "b"c` is an error
because two values have no whitespace between them. So the token stream keeps two kinds of it, `ws` for what may sit
inside a node and `nl` for what ends one.

The spec's character classes come first. It defines whitespace and newlines by code point, wider than the regexp `\s`:

```python
--8<-- "examples/kdl.py:characters"
```

A `/* */` comment nests, which a regexp cannot count. `block_comment()` writes the nesting out to a fixed depth, so a
comment holding comments more than 8 levels deep fails with `LexerError`. This is the one place the parser falls short
of the spec:

```python
--8<-- "examples/kdl.py:comments"
```

A comment and a line continuation are both whitespace, so they are part of the `ws` token, and the grammar does not
see them. A line comment ends its line, so it is part of `nl`.

The order of the specs matters, as the first to match wins. A raw string refers to its opening `#`s with `\1`, to close
with as many:

```python
--8<-- "examples/kdl.py:tokenizer"
```

`a "b"` is:

```pycon
>>> from examples.kdl import tokenize
>>> list(tokenize('a "b"'))
[Token('ident', 'a'), Token('ws', ' '), Token('string', '"b"')]

```

## Strings

The tokenizer takes a string whole, and functions give it its value. A bad escape or indent does not stop the parse:
`Parser.check()` records it and parsing goes on, so these functions have to get through a string that a check has
already reported. A multi-line string is read in the order the spec gives: whitespace escapes first, then the indent,
then the other escapes.

```python
--8<-- "examples/kdl.py:strings"
```

```pycon
>>> [script] = parse('''
... script """
...     echo one
...       echo two
...     """
... ''')
>>> script.args[0].value
'echo one\n  echo two'

```

## Numbers and keywords

```python
--8<-- "examples/kdl.py:numbers"
```

## The grammar

Each rule is a rule of the spec. A slashdash is a rule like any other: `dropped()` parses what follows it and throws it
away, whether that is a node, an entry or a block of children, so a slashdashed node is checked as thoroughly as any.
`-p` drops a value too, and `dropped()` leaves a `None` where a rule has to give one.

```python
--8<-- "examples/kdl.py:grammar"
```

`separated()` parses a node once. The spec says a node ends in a terminator, and the last one in a block may omit it,
so writing that as two rules would parse the last node twice, at each level of nesting.

The version marker `/- kdl-version 2` needs no rule, as it is a slashdashed node.

```pycon
>>> [node.name for node in parse("/- kdl-version 2\na; /- b; c\n")]
['a', 'c']

```

## Parsing a document

```python
--8<-- "examples/kdl.py:parse"
```

## Errors

Tokens out of place fail with `NoParseError`, and `render()` shows where:

```pycon
>>> from funcparserlib.parser import NoParseError, ParseErrors
>>> text = 'server port=8080{}x\n'
>>> try:
...     parse(text)
... except NoParseError as e:
...     print(e.render(text))
1,19-1,19: got unexpected token: 'x', expected: whitespace or '/-' or newline or ';' or end of input
1 | server port=8080{}x
                      ^

```

The errors of `Parser.check()` come together at the end of the parse, in a `ParseErrors`:

```pycon
>>> text = r'a "\q" "\w"'
>>> try:
...     parse(text)
... except ParseErrors as group:
...     for e in group.exceptions:
...         print(e.render(text))
1,3-1,6: unknown escape in a string
1 | a "\q" "\w"
      ^^^^
1,8-1,11: unknown escape in a string
1 | a "\q" "\w"
           ^^^^

```

What no token spec matches fails earlier, in the tokenizer, with a `LexerError` that gives the place and the line:

```pycon
>>> parse("server\n    tls #yes\n")
Traceback (most recent call last):
    ...
funcparserlib.lexer.LexerError: cannot tokenize data: 2,9: "    tls #yes"

```

The parsers recurse with the nesting of the document, so a deep enough one raises `RecursionError`. With Python's default
limit that takes several dozen levels of `{`. A parser of untrusted input should catch it and report it as a parsing
error.
