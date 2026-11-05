# Glossary

One word, one meaning. Read this page before naming something, and add the entry in the same change
that coins a term.

An entry is a line starting with a bold headword: `**term**` for a concept, `**`Name`**` for a name
the package defines (`**`A` / `B`**` for a pair). Say what it means, then which names implement it,
and which near-synonyms it is not.

**token**: one piece of the input text with a type and a value, such as a number or an operator.
`Token` implements it; `TokenSpec` describes how to find one. Not a character: a parser can read
characters, but the lexer turns text into tokens first.

**parser**: an object that reads a sequence of tokens and returns a value or fails. `Parser`
implements it; `Parser.parse()` runs it over a whole sequence.

**combinator**: a function or operator that makes a parser out of parsers: `p1 + p2`, `p1 | p2`,
`p >> f`, `-p`, `many()`, `maybe()`, `oneplus()`, `skip()`.

**forward declaration**: a parser used before it is defined, for recursive grammars.
`forward_decl()` makes one, `ForwardDeclParser.define()` defines it.

**parsing result**: what one step of parsing returns: `ParsingSuccess` with the value and the next
`State`, or `ParsingFailure` with where it stopped. `ParsingResult` is either.

**`NoParseError`**: raised by `Parser.parse()` when the tokens do not match the grammar. It names what every parser
that failed at the furthest position expected; `ParseErrors` groups it with the errors `Parser.check()` recorded.

**label**: a name given with `Parser.named()`, which an error shows instead of what the parser is made of when the
parser fails at its first token.

**committed failure**: a failure of a parser wrapped in `commit()` after it read a token; no alternative is tried for it.

**context**: a name given with `Parser.context()` to a part of the grammar, which errors inside it end with.

**`TokenDescription`**: how an error shows a token, its text and position; `Parser.parse(tokens, describe=...)` makes
one from each token.

**token grammar**: a grammar defined by a tokenizer: the tokens it makes, in the order of its specs, are what the
parsers of `tok()` read. `Parser.parse()` runs it over the tokens, and `Parser.compile()` with a tokenizer over the
text.

**text grammar**: a grammar defined by the text its parsers read at the position the parsing has reached, with no
tokenizer and no order of token specs. `Text` makes its parsers, and `Parser.parse_text()` runs it.

**`Text`**: makes the parsers of a text grammar: `Text.token()` for a regexp, `Text.literal()` for a fixed string and
`Text.end` for the end of the text. Its `skip` is what each of them skips after what it reads.

**`longest()`**: a choice that runs every alternative and keeps the longest match, the first when several tie. It has no
order to get right, unlike `p1 | p2`, which keeps the first alternative that succeeds.

**byte grammar**: a grammar defined by the bytes its parsers read at the position the parsing has reached, as a text
grammar is by its text. `Binary` makes its parsers, `Parser.parse_bytes()` runs it, and `Parser.compile()` compiles it.
Not a token grammar: a byte is not a token, and nothing reads it ahead of the grammar.

**`Binary`**: makes the parsers of a byte grammar: `Binary.byte`, `Binary.fixed()` for a number of a `struct` format,
`Binary.token()` for a regexp, `Binary.literal()` for fixed bytes, `Binary.previous()` for the byte just read and
`Binary.end` for the end of the bytes. Not `Text`, which reads `str` and skips what lies between tokens.

**`prefixed()`**: a parser of as many bytes as a length parser says, or with a body, of what the body parses of them.
It is the length field of a binary format. Not `many()`, which reads for as long as its parser matches.

**`counted()`**: a parser of as many items as a count parser says, in a list. It is the count field of a binary
format. Not `prefixed()`, which counts bytes.

**`switch()`**: a parser that reads a key and runs the parser a table holds for its value, where a key of the table is a
value or a `range`. It is the type tag of a binary format. Not `p1 | p2`, which tries each alternative in turn; a
`switch()` runs the one case.
