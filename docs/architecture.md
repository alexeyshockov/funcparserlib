# Architecture

How funcparserlib is put together, and the principles a change keeps. Each decision behind them is a record in
[the ADRs](adr/index.md); read the ones a change touches before proposing it.

## A grammar is one parser

A grammar is a `Parser`, built from parsers with operators, functions and methods, each form saying what it does
(ADR 0008). A grammar has one definition of its language (ADR 0004), in one of three kinds:

- **A token grammar** reads a sequence of tokens. Its language is the tokens its tokenizer gives for the whole text,
  read from the start: a position inside one of those tokens is no place a token grammar starts.
- **A grammar without a tokenizer** reads text at the position it has reached, with `lit()`, `regex()` and the parsers
  of `skipping()`, and a parser of text skips what follows it.
- **A token grammar bound to its tokenizer**, `tokenized()`, is a parser of text (ADR 0003). From the start of a text
  it reads the tokens of the text; inside a grammar of text it reads them from where it is reached.

What a grammar can read first decides where it starts: the skip a text starts with (`leading_skip()`), and where a
scan tries it, at every offset or only where a token of the text starts (`leading_tokenizer()`). A rule that depends
on the kind of a grammar is decided there, from the forms of the grammar, so it holds through `>>`, `located()`, `+`
and every other combinator. A rule given to one parser object, as a method of it, holds only where that object is the
root.

## The parser decides, generated code follows

The interpreted parser is the definition of what a grammar gives and of every error (ADR 0002). `Parser.compile()`
generates code for the parses that succeed; where that code cannot decide, it gives the decision back to the parser,
which also makes every error. A compiled parser gives what its parser gives.

That is checked by comparing the two on inputs nobody chose, which finds what tests written by hand miss:

- the property tests run random grammars of each kind, compiled and interpreted, and compare what they give, parse
  and scan alike, and fail when a form of generated code is out of their reach;
- the `shadow` session runs every parse and scan of the suite and of the docs on the compiled parser too, and
  compares each result.

A form a new combinator adds to generated code is added to the random grammars in the same change.

## Positions

`State.pos` is an index into what is parsed, and `State._skipped` how much the last parser of text skipped after its
value, so a span ends where its value ends, before the skip (ADR 0009). Generated code carries the same end beside
the position, and keeps it wherever its paths join, as the parser keeps its state. A position in text is an offset;
`Lines` turns it into a line and a column, and tokens carry their line and column themselves.

## Scans

A scan tries a grammar at each position of its input, lazily, and goes past each match: `scan()` at each index of a
sequence of tokens, `scan_text()` at each offset of a text, or at each token of the text for a grammar that starts
with a bound part (ADR 0003, ADR 0009). A failed `commit()` or a rejected value raises, so a migration stops at a
broken construct. The tokens of another library, as markdown-it's, are scanned as they are, since such a tokenizer
reads a whole document and cannot start inside one.

## What stays out

A concrete syntax tree is no part of the library: a migration keeps the text outside its constructs, which spans
give, and the layout of a file read by another library is that library's (ADR 0009).
