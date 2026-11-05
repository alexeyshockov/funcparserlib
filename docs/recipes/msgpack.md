# Parsing MessagePack

[MessagePack](https://msgpack.org/) is a binary format for the values JSON holds, and a few more: byte strings,
extension types, timestamps. This recipe parses all of it with funcparserlib alone, from the
[specification](https://github.com/msgpack/msgpack/blob/master/spec.md). The whole program is
`examples/msgpack_format.py`; every block below is included from it.

```pycon
>>> from examples.msgpack_format import parse
>>> parse(b"\x82\xa4name\xa3Ann\xa3age\x1e")
{'name': 'Ann', 'age': 30}
>>> parse(b"\x93\x01\xc3\xcb\x40\x09\x21\xf9\xf0\x1b\x86\x6e")
[1, True, 3.14159]

```

## A grammar of a specification

A format that has a specification and describes itself, as MessagePack does, suits a grammar: the specification is a
table of the first byte of a value, and each row says what follows. A grammar written from it is that table, a row for
a row, with nothing to infer from a schema. A format whose layout comes from a schema of its own, such as a Protocol
Buffers message, has the layout in the schema, and its parser is generated from there.

What sets a binary format apart from text is that what comes next depends on a number just read: a length, a count, a
type. A [`Binary`](../api/binary.md) makes the parsers of a **byte grammar**, which read the bytes at the position the
parsing has reached, and three combinators take such a number:

* `prefixed(length)` reads as many bytes as `length` parses;
* `counted(count, item)` reads as many `item`s;
* `switch(key, cases)` reads `key` and runs the parser that `cases` holds for its value.

Each is a form the compiler reads, so the grammar compiles, as a text grammar does.

## The tree

A value is what `msgpack.unpackb(data, raw=False, strict_map_key=False, timestamp=0)` returns: `None`, `bool`, `int`,
`float`, `str`, `bytes`, `list` and `dict`, and two types for the extensions. An extension is a type number and bytes
the application gives a meaning to, and one type is the specification's own, number -1: a timestamp.

```python
--8<-- "examples/msgpack_format.py:tree"
```

## Numbers

The numbers are `struct` formats, big-endian as the specification says. `Binary.fixed()` reads one value of a format,
and `named()` gives each the name its errors use:

```python
--8<-- "examples/msgpack_format.py:numbers"
```

## Recursion

An array holds values, so a value is declared before it is defined, as in any recursive grammar. The name is what an
error says was expected:

```python
--8<-- "examples/msgpack_format.py:recursion"
```

## Lengths and counts

A string, an array and a map of every size are the same three rules over a different number: the specification
gives each of them a form with a length of 8, 16 or 32 bits, and a short one that holds the length in the type byte.
So a rule takes the parser of its length:

```python
--8<-- "examples/msgpack_format.py:lengths"
```

`prefixed(length)` is the bytes a string has, and `bytes.decode` turns them into a `str`. `counted(length, value)` is
an array. A map is a count of pairs, so it is `counted()` of a key and a value, joined with `dict`.

## A tag that holds a value

The short forms put the length, or the value itself, in the low bits of the type byte: `0xA5` is a string of 5 bytes,
`0x93` an array of 3, `0x07` the number 7 and `0xFF` the number -1. A `switch()` reads the type byte and runs a case
after it, so the case has to read that byte again without moving on. `previous(mask)` is that: the byte just read,
under a mask, with nothing consumed.

```python
--8<-- "examples/msgpack_format.py:tags"
```

`text(low5)` is `prefixed()` of a length that is the low 5 bits of the tag, so `0xA5` reads 5 bytes. A number is a
`previous()` too, and the negative ones add a function of the byte, as 8-bit two's complement is not a `struct` format
of its own.

## Extensions and `bind()`

A fixed extension keeps its size in its type byte, so `fixext(4)` reads the type number and 4 bytes. The other
extensions have the order the specification gives them: a length, then the type number, then as many bytes as the
length says. The length is read before the parser of the bytes can be made, and neither `prefixed()` nor `counted()`
takes a number that sits after the length it uses. That is what `Parser.bind()` is for: it makes a parser out of a
value read at run time.

```python
--8<-- "examples/msgpack_format.py:extensions"
```

`bind()` has no form for the compiler to read, so generated code calls the parser it makes as it is, and the rest of
the grammar stays compiled. The timestamp has three layouts, by the length of its data, and the functions turn its
bytes into seconds and nanoseconds.

## The table

The grammar is `switch()` on the type byte. Every key of the table is a byte or a range of them, and each row is a row
of the specification. A byte that has no row, `0xC1`, is reserved and fails the parse:

```python
--8<-- "examples/msgpack_format.py:grammar"
```

The value is named `MessagePack value`, and that is what an error says when the type byte is out of place. The switch
is named too, but an error at the first byte of a value reads the label of the value that holds it.

## Parsing a document

A document is one value and the end of the input. `Parser.compile()` makes the generated functions, and
`parse_bytes()` runs them over the bytes:

```python
--8<-- "examples/msgpack_format.py:parse"
```

## Errors

An error gives the byte in hexadecimal and its offset in the bytes. A byte that no row of the table has:

```pycon
>>> parse(b"\x92\x01\xc1")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: got unexpected token: 0xc1 at offset 0x2, expected: MessagePack value

```

Bytes that end where a value should be, and a number that ends in the middle of its width, which says the name given
to it above:

```pycon
>>> parse(b"\x92\x01")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: got unexpected end of input at offset 0x2, expected: MessagePack value
>>> parse(b"\xcd\x01")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: not enough bytes: 0x01 at offset 0x1, expected: uint16

```

A length that runs past the end of the bytes is reported where its bytes start:

```pycon
>>> parse(b"\xa5ab")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: length 5 runs past the end: 0x61 at offset 0x1

```

And bytes after the document:

```pycon
>>> parse(b"\x01\x02")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: got unexpected token: 0x02 at offset 0x1, expected: end of input

```

Some failures are not parsing errors. A string that is not UTF-8 raises `UnicodeDecodeError` from `bytes.decode`, and a
map key that is an array or a map raises `TypeError`, as it does in `msgpack.unpackb()`, and so does a timestamp of a bad
length with `ValueError`:

```pycon
>>> parse(b"\xa2\xff\xfe")
Traceback (most recent call last):
    ...
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
>>> parse(b"\x81\x90\x01")  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
TypeError: unhashable type: 'list'

```

The grammar recurses with the nesting of the document, so a deep enough one raises `RecursionError`.

## How it compares

The [`msgpack`](https://pypi.org/project/msgpack/) package reads the same documents with a C extension written for this
one format, and with a parser in pure Python that it falls back to. The `msgpack` group of `bench_comparison.py` runs
both and this grammar, interpreted and compiled, over a document of nested records, and asserts that each returns what
`msgpack.unpackb()` does: `just bench::micro -k bench_comparison`. The C extension is far ahead of anything written in
Python. The compiled grammar and the pure Python parser of the package, which is written by hand for this format, are
of the same order, and an interpreted grammar costs several times what a compiled one does. The figures
are in [Performance](../guide/performance.md#how-fast-is-it), and the decision in
[ADR 0005](../adr/0005-byte-grammars.md).

The grammar earns its place where the format is the question. It is the specification, row for row, and a new
extension type or a stricter rule is a row.
