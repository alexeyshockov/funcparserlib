# Recipes

Each recipe is a complete parser of a real format, with the whole program in `examples/`. Pick the one closest to your
input. For a first read, [Parsing HTTP Headers](http-headers.md) is short and builds up from shared pieces, while
[Parsing KDL](kdl.md) and [Parsing YAML](yaml.md) have the largest programs.

## External Tokenizers

funcparserlib parses a sequence of any objects, so when a library already tokenizes your format, you can keep the hard
part there and write only the grammar:

* [Parsing Markdown with markdown-it](markdown.md): a typed syntax tree from a flat token stream
* [Parsing YAML with ruamel.yaml](yaml.md): a scanner's tokens, and a second pass that gives the tree its meaning

## Binary Data

When the input is bytes, the grammar reads them directly and follows the specification of the format, including lengths,
counts and type tags:

* [Parsing MessagePack](msgpack.md): all of a binary format with funcparserlib alone

## Rules of a Format

A document can be valid Markdown and still break the rules of its format. When a generic parser and a few lines of code
cover your problem, you do not need this library. When the format is in the details, a version, a status, what may follow
what, grammars read those details on top of the tokens of a Markdown tokenizer:

* [Checking a Changelog](changelog.md): Keep a Changelog, with a version grammar and every broken release in one run
* [Checking Decision Records](adr.md): one record as a grammar, and the rules across records in plain code

## Web Formats

The formats of the web come with a grammar in their RFCs, and with a standard library that reads them leniently. A grammar
is for the cases where a split gets it wrong or where invalid input has to be rejected, and each recipe says which:

* [Parsing URIs](uri.md): RFC 3986 next to `urlsplit()`, and a strict query string next to `parse_qsl()`
* [Parsing HTTP Headers](http-headers.md): `Accept`, `Content-Disposition`, `Link` and `WWW-Authenticate` from shared
  pieces, and why `Cookie` needs no grammar
* [Parsing Structured Fields](structured-fields.md): RFC 9651, checked against the test suite of the HTTP working group

## A Whole Language

* [Parsing KDL](kdl.md): a language defined from scratch, first with `make_tokenizer()`, then with no tokenizer at all
