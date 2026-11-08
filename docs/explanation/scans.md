# How a Scan Walks a Text

[`Parser.scan_text()`](../api/parser.md#funcparserlib.parser.Parser.scan_text) finds the parts of a text a grammar
reads, from left to right, without overlaps. This page is how it gets there, one offset at a time.
[Step through it](scans.html) with a grammar and a text of your own choosing.

## One Offset at a Time

A scan starts at offset 0 and tries the grammar there, as if the text started at that offset:

- **The grammar fails**: there is no match here, and the scan moves on by one character.
- **The grammar succeeds**: the scan gives a match, a `Located` with the value, where it starts and where the value
  ends, and goes on from where the match ended. What the match read is never tried again, so matches never overlap.
- **The scan reaches the end of the text**: it stops.

```pycon
>>> from funcparserlib.parser import lit, regex

>>> version = regex(r"\d+") + -lit(".") + regex(r"\d+")
>>> [(m.value, m.start, m.end) for m in version.scan_text("From 1.2 to 3.14, see above.")]
[(('1', '2'), 5, 8), (('3', '14'), 12, 16)]

```

The scan is lazy: it tries offsets only as far as you take matches.

## When a Scan Raises

Two failures are not "no match here" but errors in the text, and the scan raises what `Parser.parse_text()` would: a
`commit()` that failed, and a value `Parser.validated()` rejected. A match that reads nothing raises `ValueError`, since
the scan could not move past it: a grammar like `maybe(regex(r"\d+"))` succeeds everywhere without reading.

## Where a Token Grammar Starts

A grammar that starts with `tokenized()` is tried only where a token starts, as its tokenizer reads the text from its
start, so a match never starts inside a word, a string or a comment the tokenizer reads as one token.
[Find matches in a text](../guide/positions-and-matches.md#find-matches-in-a-text) shows both kinds.
