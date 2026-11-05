# 0005 — Byte grammars

- **Status:** Accepted
- **Date:** 2026-11-05

## Context

Bytes iterate as integers, so `Parser.parse()` reads a binary format with `some()` and `bind()`. That
works and it is the wrong shape. A byte per step costs a call each, `bind()` makes a parser out of a
value read at run time, which generated code cannot look into, and an error says `197` where a reader
of binary wants a hexadecimal value at an offset. A Protobuf wire decoder written that way took 100 ms
on 219 KB, `.compile()` changed nothing, and a plain Python loop took 5 ms.

What sets a binary format apart is that what comes next depends on a number just read: a length, a
count, a tag. ADR 0002 generates code from what a parser is built from, so those dependencies need
forms the generator can read.

Protobuf is defined by a schema, and its parsers are generated from it, so a grammar of its wire
format is a grammar of one message. MessagePack is defined by a specification, describes itself, and
has a C parser and a parser in pure Python to measure against. A grammar for it is written once.

## Decision

`funcparserlib.binary.Binary` makes parsers of bytes, as `Text` makes parsers of text, and they read
the buffer at the position: `token()` and `literal()` as in `Text`, `byte`, `fixed()` for what the
`struct` module reads, `end`, and `previous()` for the byte just read, where a tag holds a value.
`Parser.parse_bytes()` runs them, and errors give the byte in hexadecimal and its offset.

Three combinators replace `bind()` for what binary formats do, and they work over text and tokens too:

- `prefixed(length)` reads as many bytes as `length` says, and `prefixed(length, body)` parses them with
  `body`, which has to read to their end.
- `counted(length, item)` reads as many items as `length` says.
- `switch(key, cases)` chooses a parser by a value, a key being a value or a `range`.

`bind()` stays for what they do not cover. It costs what it always did.

Measured on Python 3.12 on an arm64 laptop, CPU time, best of five, in a prototype, on a 450 KB
MessagePack document of nested records:

| Parser | Time |
| --- | ---: |
| `msgpack`, the C extension | 2.6 ms |
| Grammar, compiled | 26.4 ms |
| `msgpack`, its pure Python fallback | 38.7 ms |
| Grammar, interpreted | 120 ms |

The prototype agreed with `msgpack.unpackb()` on 100,000 valid and mutated documents.

Rejected:

- **A `switch` case as a function of the key.** It reads plainly, and hides the parser from generated
  code as `bind()` does. `previous()` reads the same byte, and the case stays a parser.
- **Bytes through `Text`.** A `Text` skips and counts lines, which bytes have no use for.
- **Readers for Protobuf or any one format in the library.** A varint is a `token()` and a function.

## Consequences

- A grammar of bytes compiles as one of text does, and `try_parse()` takes bytes.
- `previous()` depends on what ran before it, so it belongs right after what read the byte. It fails at
  the start of the input.
- A region of `prefixed(length, body)` is parsed as its own input: `previous()` and `Binary.end` see
  its bounds, and errors are moved back to offsets in the buffer.
- MessagePack's `ext` types put the type between the length and the data, which takes `bind()`.
