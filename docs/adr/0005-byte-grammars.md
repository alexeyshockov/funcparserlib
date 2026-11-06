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

`funcparserlib.binary` has parsers of bytes, as `lit()` and `regex()` read text, and they read the
buffer at the position: `regex()` and `lit()` as in text, `byte()`, `fixed()` for what the `struct`
module reads, and `previous()` for the byte just read, where a tag holds a value. A grammar ends with
`finished`. `Parser.parse_bytes()` runs them, and errors give the byte in hexadecimal and its offset.

Three combinators replace `bind()` for what binary formats do, and they work over text and tokens too:

- `prefixed(length)` reads as many bytes as `length` says, and `prefixed(length, body)` parses them with
  `body`, which has to read to their end.
- `counted(count, item)` reads as many items as `count` says.
- `switch(key, cases)` chooses a parser by a value, a key being a value or a `range`.

`bind()` stays for what they do not cover. It costs what it always did.

Measured on Python 3.12 on an arm64 laptop, the fastest round of `bench_comparison.py`, on a 450 KB
MessagePack document of nested records, with `examples/msgpack_format.py`:

| Parser | Time |
| --- | ---: |
| `msgpack`, the C extension | 2.7 ms |
| Grammar, compiled | 27.7 ms |
| `msgpack`, its pure Python fallback | 39.0 ms |
| Grammar, interpreted | 107 ms |

The example agrees with `msgpack.unpackb()` on mutated documents, but for extension types below -1,
which the specification reserves and `msgpack` refuses.

Rejected:

- **A `switch` case as a function of the key.** It reads plainly, and hides the parser from generated
  code as `bind()` does. `previous()` reads the same byte, and the case stays a parser.
- **A class of parsers, as `Text`.** It would hold nothing: a `Text` exists to hold `skip`.
- **Bytes through `Text`.** A `Text` skips and counts lines, which bytes have no use for.
- **Readers for Protobuf or any one format in the library.** A varint is a `token()` and a function.

## Consequences

- `funcparserlib.binary` and `Parser.parse_bytes()` are experimental: reading bytes is an uncommon use of
  the library, so they are outside its compatibility, and can change or go in any release. `prefixed()`,
  `counted()` and `switch()` read text and tokens too, and are not.
- A grammar of bytes compiles as one of text does, and `try_parse()` takes bytes.
- `previous()` depends on what ran before it, so it belongs right after what read the byte. It fails at
  the start of the input.
- A region of `prefixed(length, body)` is parsed as its own input: `previous()` and `finished` see
  its bounds, and errors are moved back to offsets in the buffer.
- MessagePack's `ext` types put the type between the length and the data, which reads most plainly with
  `bind()`.
- `prefixed()` fails after the length it read, so `commit()` and `recover()` see that read. `switch()`
  fails at its key, as a token that does not match does: the value is what an error should show.
