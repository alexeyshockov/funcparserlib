# Parsing Structured Fields

[RFC 9651](https://www.rfc-editor.org/rfc/rfc9651) gives new HTTP headers one syntax, so that each does not bring its
own: `Cache-Status`, `Priority` and `Signature-Input` are all written this way. A header says only whether it is a list,
a dictionary or an item. This recipe parses all three. The whole program is `examples/structured_fields.py`; every
block below is included from it.

```pycon
>>> from examples.structured_fields import Item, parse_dictionary, parse_item, parse_list
>>> parse_dictionary("u=2, i")
{'u': Item(value=2, parameters={}), 'i': Item(value=True, parameters={})}
>>> parse_item('"hello";lang=en')
Item(value='hello', parameters={'lang': Token(value='en')})

```

## A grammar of a specification

The RFC gives the syntax as an algorithm and asks for a strict parser: a field with one wrong character is thrown away
as a whole, and the other headers of the request stay. The [pieces of RFC 9110](http-headers.md#the-pieces) forgive
empty elements and a trailing comma, and these rules forgive nothing, so this is a grammar of its own.

It suits a grammar well. The first character of a value says what the value is, a number being the one choice left to
make, and what the algorithm does in loops is `many()` and `separated()`. This is the case for `separated()` without `trailing`,
as a comma at the end of a list fails it:

```pycon
>>> parse_list("a, b,")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,6-1,6: got unexpected end of input, expected: whitespace or '(' or decimal or integer or '"' or token or ':' or boolean or '@' or '%"'

```

The HTTP working group keeps a [suite of test cases](https://github.com/httpwg/structured-field-tests) for
implementations, and its hand-written cases run against this recipe in `tests/examples/test_structured_fields.py`.

## The tree

Eight types of value can have parameters. An integer and a string are Python's, a decimal is a `Decimal`, as the RFC
counts its digits, and three are types of their own, so that a token is not a string:

```python
--8<-- "examples/structured_fields.py:tree"
```

## The values

A value that the type of its characters cannot check goes through `Parser.validated()`: the alphabet of base64 does not
make text base64, and an escape in a display string is a byte, which only together with the others is UTF-8.

```python
--8<-- "examples/structured_fields.py:values"
```

## The grammar

The rest is a line for each rule of the RFC. A key with no value is the boolean true in parameters, and in a dictionary
a key with no value still takes parameters of its own. A repeated key takes the last value and keeps the first place,
which is what `dict()` does:

```python
--8<-- "examples/structured_fields.py:grammar"
```

## Parsing

A field can come on several lines, and the RFC joins them with a comma, so that is for the caller to do first:

```python
--8<-- "examples/structured_fields.py:parse"
```

```pycon
>>> [member.value for member in parse_list(", ".join(["sugar", "tea"]))]
[Token(value='sugar'), Token(value='tea')]

```
