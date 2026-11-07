# Glossary

The words these docs use, each with one meaning.

## Grammars

**grammar**: the rules of a language, written in funcparserlib as parsers put together with combinators. The whole
grammar is itself one parser.

**token grammar**: a grammar that reads the tokens a tokenizer made before parsing began. `tok()` reads a token by its
type.

**tokenized grammar**: a token grammar bound to its tokenizer by `tokenized()`. It is a parser of text: it reads its
tokens from where it is reached, and combines with the parsers of a grammar without a tokenizer.

**grammar without a tokenizer**: a grammar that reads text at the position parsing has reached. `lit()`, `regex()` and
`skipping()` make its parsers, and `Parser.parse_text()` runs it.

**byte grammar**: a grammar that reads bytes at the position parsing has reached. `funcparserlib.binary` has its parsers,
and `Parser.parse_bytes()` runs it.

**left recursion**: a rule that reaches itself before reading anything. It never ends, so a grammar has to read a token
first.

## Tokens

**token**: one piece of the input text with a type and a value, such as a number or an operator. `Token` implements it;
`TokenSpec` describes how to find one. Not a character: a parser can read characters, but a tokenizer turns text into
tokens first.

**tokenizer**: what splits text into tokens, also called a lexer. `make_tokenizer()` makes one from a list of `TokenSpec`.

## Parsers

**parser**: an object that reads its input, tokens, text or bytes, and returns a value or fails. `Parser` implements it;
`Parser.parse()` runs it over a whole sequence.

**primitive parser**: a parser that is not made of other parsers: `tok()`, `a()`, `some()`, `finished`.

**combinator**: an operator, function or method that makes a parser out of parsers: `p1 + p2`, `p1 | p2`, `p >> f`, `-p`,
`many()`, `maybe()`, `oneplus()`, `separated()`, `Parser.when()`.

**alternative**: one of the parsers of a choice, such as `p1` in `p1 | p2`. `p1 | p2` keeps the first that succeeds, and
`longest()` the longest match.

**backtracking**: going back to where an alternative started to try another one, after it failed.

**forward declaration**: a parser used before it is defined, for recursive grammars. `forward_decl()` makes one,
`ForwardDeclParser.define()` defines it.

**parsing result**: what one step of parsing returns: `ParsingSuccess` with the value and the next `State`, or
`ParsingFailure` with where it stopped. `ParsingResult` is either.

**parse tree**: the value a grammar returns for the whole input, usually built with `p >> f` from your own classes.

## Errors

**parse error**: a failure to match the grammar. It names everything that was expected at the furthest position parsing
reached. `NoParseError` implements it, and `ParseErrors` groups several.

**label**: a name given with `Parser.named()`, which an error shows instead of what the parser is made of when the parser
fails at its first token.

**context**: a name given with `Parser.context()` to a part of the grammar, which errors inside it end with.

**committed failure**: a failure of a parser wrapped in `commit()` after it read a token; no alternative is tried for it.

## Positions

**place**: the line and the column of a position in a text, both counted from 1. `Token.start` and `Token.end` are
places, and `Lines` turns an offset into one and back.

**located value**: a value with the part of the input it was read from, as indexes into what was parsed: character
offsets in a text, token indexes in tokens. `Parser.located()` gives one, and `Located` implements it.

**match**: a part of a text a grammar reads, found by trying the grammar at each offset. `Parser.matches()` gives them,
and `Parser.replace()` rewrites them.

## Running a Grammar

**interpreted parser**: a grammar run as the tree of parsers it is made of. It tracks what was expected, and all errors
come from it.

**compiled parser**: a grammar turned into generated Python functions by `Parser.compile()`, faster for the parses that
succeed. `CompiledParser` implements it. Whenever the generated functions don't succeed, it runs the interpreted parser.
