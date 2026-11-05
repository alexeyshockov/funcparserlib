# Parsing YAML with ruamel.yaml

funcparserlib parses a sequence of any objects, so the hard part of a format can stay in another library. Most of what
makes YAML difficult is scanning: indentation, the many scalar styles, and characters whose meaning depends on where
they are. This recipe takes the tokens of [ruamel.yaml](https://yaml.dev/doc/ruamel.yaml/)'s scanner, parses them into
a tree with funcparserlib, and gives the tree its meaning in a second pass. The whole program is `examples/yaml.py`;
every block below is included from it.

## The subset

The parser reads one document of block and flow collections, scalars of every style, anchors, aliases and merge keys.
Plain scalars follow the YAML 1.2 core schema, so `no` is a string and `0o17` is a number. It leaves out tags,
directives, multiple documents and keys that are not scalars. A duplicate key is an error.

## The token stream

The scanner has already resolved indentation, so `a: [1]` is:

```text
StreamStartToken
BlockMappingStartToken
KeyToken
ScalarToken             a
ValueToken
FlowSequenceStartToken
ScalarToken             1
FlowSequenceEndToken
BlockEndToken
StreamEndToken
```

The scanner puts a `KeyToken` before each key, inserting it once it has seen the colon. So a parser never has to read a
node to find out that it was a key, and the grammar does not backtrack over keys.

## The tree

```python
--8<-- "examples/yaml.py:tree"
```

## Matching a token by class

A ruamel.yaml token is matched by its class with `some()`. An implicit key has an empty `KeyToken`, and the explicit
`?` of a complex key has a one-character one, so `implicit()` tells them apart:

```python
--8<-- "examples/yaml.py:token"
```

## The grammar

Each rule reads the stream as the scanner writes it. Tags and directives have no rule, so their tokens fail the parse
where they appear:

```python
--8<-- "examples/yaml.py:grammar"
```

## Anchors and aliases

An alias refers to the anchor before it in the document, and a later anchor of the same name replaces it. The grammar
cannot follow that order: `|` may run the `>>` function of a branch it then abandons, so a function that recorded
anchors would see some that are not in the document. The grammar only builds the tree, and `resolve()` walks it in
order:

```python
--8<-- "examples/yaml.py:resolve"
```

An alias shares the object of its anchor, and a collection is anchored before its items are read, so an alias inside
its own anchor makes a cycle:

```pycon
>>> from examples.yaml import parse
>>> config = parse("""
... defaults: &defaults
...   retries: 3
...   timeout: 1.5
... service:
...   <<: *defaults
...   retries: 5
... """)
>>> config["service"]
{'retries': 5, 'timeout': 1.5}
>>> nodes = parse("a: &a {next: *a}")
>>> nodes["a"]["next"] is nodes["a"]
True

```

Sharing also means that a small document of nested aliases loads quickly and expands to an enormous one for code that
walks it, such as `json.dumps()`. A parser of untrusted input should limit how many aliases it follows.

## The core schema

Only a plain scalar is resolved; a quoted one is always a string:

```python
--8<-- "examples/yaml.py:schema"
```

```pycon
>>> parse("[no, 010, 0o17, '1', ~]")
['no', 10, 15, '1', None]

```

## Parsing a document

```python
--8<-- "examples/yaml.py:parse"
```

## Errors

A ruamel.yaml token has no `start` and `end`, so `describe` gives an error the token's text and place:

```python
--8<-- "examples/yaml.py:describe"
```

```pycon
>>> parse("a: [1, 2\nb: 3\n")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 2,2-2,2: got unexpected token: :, expected: ',' or ']'

```

The parsers recurse with the nesting of the document, so a deep enough one raises `RecursionError`. A parser of
untrusted input should catch it and report it as a parsing error.
