# Parsing Markdown with markdown-it

funcparserlib parses a sequence of any objects, so the tokens can come from another library's lexer. This recipe uses
[markdown-it-py](https://markdown-it-py.readthedocs.io/) to tokenize Markdown and funcparserlib to build a typed
syntax tree from its tokens. The whole program is `examples/markdown.py`; every block below is included from it.

## The token stream

markdown-it-py's [token stream](https://markdown-it-py.readthedocs.io/en/latest/using.html#the-token-stream) is flat.
A block is an `*_open` token, its contents and a matching `*_close` token, so `# Title *here*` is:

```text
heading_open  h1
inline        "Title *here*"
heading_close h1
```

The text of a block is an `inline` token whose `children` are a second stream: `text "Title "`, `em_open`,
`text "here"`, `em_close`. The parser is therefore two grammars, one over each stream.

## The tree

The result is a tree of frozen dataclasses, one for each construct the grammars cover:

```python
--8<-- "examples/markdown.py:tree"
```

## Matching a token by type

`tok()` matches funcparserlib's own `Token`. A markdown-it token is matched by its `type` with `some()`, named after
the type, so an error says which token the grammar expected:

```python
--8<-- "examples/markdown.py:token"
```

## The inline grammar

Each rule reads the stream as markdown-it-py writes it. `-p` drops the value of the open and close tokens, and `>>`
builds the node from what is left. Emphasis and links contain inline content, so `inline` is a forward declaration:

```python
--8<-- "examples/markdown.py:inline"
```

## The block grammar

A block rule reads an `inline` token by parsing its children with the inline grammar, inside `>>`. List items and
block quotes contain blocks, so `block` is a forward declaration too:

```python
--8<-- "examples/markdown.py:block"
```

## Parsing a document

```python
--8<-- "examples/markdown.py:parse"
```

```pycon
>>> from examples.markdown import parse
>>> parse("# Title *here*\n\n- one\n- two")
[Heading(level=1, children=[Text(text='Title '), Emphasis(children=[Text(text='here')])]), BulletList(items=[[Paragraph(children=[Text(text='one')])], [Paragraph(children=[Text(text='two')])]])]

```

## Errors

A construct the grammars do not cover, such as an image or an HTML block, fails with `NoParseError`. By default an
error shows a token as the object it is, which for a markdown-it token is its whole `repr()`. `parse()` takes a
`describe` function that turns a token into the `TokenDescription` the message shows; this one gives the token's type
and line:

```python
--8<-- "examples/markdown.py:describe"
```

The two forward declarations are named, `forward_decl().named("block")` and `forward_decl().named("inline content")`,
so an error says what kind of content was expected rather than listing every rule:

```pycon
>>> parse("- one\n- two ![image](i.png)")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: got unexpected token: image on line 2, expected: inline content or end of input

```

## Migrating Notes to Alerts

The tree above is built from a whole document. A migration needs the other way round: find each construct wherever it
is, rewrite it, and keep every other character of the file. `examples/notes.py` turns notes into GitHub alerts. A note is
written as a paragraph that starts with `Note:` or `**Note:**`, as a block quote whose first paragraph is one, or as a
`Note` heading followed by paragraphs, and a heading that says `Note` has to be followed by one:

```python
--8<-- "examples/notes.py:grammar"
```

[`Parser.scan()`](../api/parser.md#funcparserlib.parser.Parser.scan) tries the grammar at each index of markdown-it's
tokens, so a note inside a list item or a block quote is found as one anywhere else. markdown-it gives the lines of a
block in `map`, and a close token has none, so `span` takes the lines of the tokens that have them:

```python
--8<-- "examples/notes.py:span"
```

markdown-it gives the text of a block without the prefix each of its lines has in a container, `-` in a list item or
`>` in a block quote. The new text gets it back: the first line takes the prefix of the block's first line, and the
lines after it are indented as far. A note that was a block quote is the alert itself, so its own `>` is no part of the
prefix. Lines end as markdown-it ends them, so a file with `\r\n` keeps them:

```python
--8<-- "examples/notes.py:prefix"
```

```python
--8<-- "examples/notes.py:migrate"
```

```pycon
>>> from examples.notes import migrate
>>> print(migrate("- one\n- Note: keep it\n  short\n- two\n"), end="")
- one
- > [!NOTE]
  > keep it
  > short
- two

```

A `Note` heading with something else after it stops the migration at its line, before anything is written:

```pycon
>>> migrate("#### Note\n\n- a list\n")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 3,1-3,1: got unexpected token: bullet_list_open, expected: paragraph_open

```
