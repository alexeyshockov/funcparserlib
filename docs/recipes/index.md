# Recipes

Each recipe is a complete parser of a real format, with the whole program in `examples/`. Pick the one closest to your
input.

## External Tokenizers

funcparserlib parses a sequence of any objects, so when a library already tokenizes your format, you can keep the hard
part there and write only the grammar:

* [Parsing Markdown with markdown-it](markdown.md): a typed syntax tree from a flat token stream
* [Parsing YAML with ruamel.yaml](yaml.md): a scanner's tokens, and a second pass that gives the tree its meaning

## Binary Data

When the input is bytes, the grammar reads them directly and follows the specification of the format, including lengths,
counts and type tags:

* [Parsing MessagePack](msgpack.md): all of a binary format with funcparserlib alone

## A Whole Language

* [Parsing KDL](kdl.md): a language defined from scratch, first with `make_tokenizer()`, then with no tokenizer at all
