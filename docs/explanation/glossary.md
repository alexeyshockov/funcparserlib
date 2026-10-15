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
`State`, or `ParsingError` with where it stopped. `ParsingResult` is either.

**`NoParseError`**: raised by `Parser.parse()` when the tokens do not match the grammar. It names what every parser
that failed at the furthest position expected; `ParseErrors` groups it with the errors `Parser.check()` recorded.

**label**: a name given with `Parser.named()`, which an error shows instead of what the parser is made of when the
parser fails at its first token.

**committed failure**: a failure of a parser wrapped in `commit()` after it read a token; no alternative is tried for it.

**context**: a name given with `Parser.context()` to a part of the grammar, which errors inside it end with.

**`TokenDescription`**: how an error shows a token, its text and position; `Parser.parse(tokens, describe=...)` makes
one from each token.
