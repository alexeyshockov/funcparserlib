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

A construct the grammars do not cover, such as an image or an HTML block, fails with `NoParseError`.
