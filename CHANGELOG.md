# Changelog

## [Unreleased]

Version 2.0 requires Python 3.12 or newer. For older Pythons, use `funcparserlib~=1.0`.

The grammar-writing API of 1.0 keeps working: `tok`, `a`, `some`, `many`, `oneplus`, `maybe`, `skip`,
`pure`, `finished`, `forward_decl`, `p1 + p2`, `p1 | p2`, `p >> f`, `-p`, `p.named()`, `p.parse()`,
`p.bind()`, `make_tokenizer`, `TokenSpec`, `Token` and `pretty_tree`, with the same parsed values. Error
messages name what was expected more precisely; see Changed.

### Added

- Error messages name everything that was expected at the position parsing stopped: `(p1 + p2) | (p1 + p3)` failing
  after `p1` expects `p2 or p3`, and `many(p) + -finished` expects `p or end of input`. A sequence is expected as its
  first parser, not by its shape
- A name given with `Parser.named()` is a label: when the parser fails at its first token, the error names it
- `NoParseError.token`, `.expected`, `.context` and `.description`, and `NoParseError.render(source)`, which adds the
  source line and a `^` marker under the token
- `Parser.parse(tokens, describe=...)`, a function from a token to the `TokenDescription` an error shows, for tokens
  that are not `funcparserlib.lexer.Token`
- `commit(p)`: once `p` has read a token, its failure is final, and `|`, `many` and `maybe` do not backtrack from it
- `Parser.context(label)`: errors inside the parser end with `in label`
- `Parser.check(pred, message)`: a non-fatal error; `parse()` raises every recorded one together as `ParseErrors`,
  an `ExceptionGroup` of `NoParseError`
- Support for Python 3.12, 3.13, 3.14 and 3.15, including the free-threaded 3.15t build
- `Parser.compile()`: a grammar as generated Python functions, several times faster for the parses that succeed. A
  `CompiledParser` gives the values and the errors of the parser it comes from
- `make_tokenizer(specs, skip=...)` leaves token types such as whitespace out of the result, and returns a `Tokenizer`
- `Parser.compile(tokenizer)` and `CompiledParser.parse_text(text)`: text parsed without making `Token` objects, as long
  as the grammar reads tokens through `tok()`
- `funcparserlib.binary.Binary`: parsers that read bytes at the position, and `Parser.parse_bytes(data)`, whose errors give
  the byte and its offset in hexadecimal
- `prefixed(length, body)`, `counted(count, item)` and `switch(key, cases)`: what depends on a number just read, in forms
  generated code runs, where `bind()` leaves it to the interpreted parser
- `funcparserlib.text.Text`: parsers that read text at the position parsing has reached, with no list of tokens, and
  `Parser.parse_text(text)`, whose errors give the line and column
- `longest(p1, p2, ...)`: the longest match of the parsers, whatever their order
- `Parser.recover(until)`: record a failure and go on from where `until` matches, so one parse reports every error
- `CompiledParser.try_parse(text)`: the value, or `None` without making an error
- `Parser.compile(source_file=path)`: the generated source is written to `path`, and frames of generated code report
  that file, so tracebacks, `pdb` and an IDE can show it
- `LexerError.render()`, which shows the line with a `^` under the character no token spec matches, as
  `NoParseError.render()` does
- `Parser.map()` (the method behind `>>`), `Parser.starmap()`, `Parser.when()`, `Parser.but_not()`
  and `~p` / `anything_but(p)`
- `funcparserlib.util.expand_tuple_args()`, which adapts a function of N arguments to the tuple that
  `p1 + ... + pN` produces
- The `parser(name)` decorator, which makes a named `Parser` from a parsing function
- `ParsingResult`, `ParsingSuccess` and `ParsingFailure`: the result a parsing function returns
- `ForwardDeclParser`, the type `forward_decl()` returns, generic in the token and value types
- Public type names for annotations: `Place`, `Spec` (lexer); `ParserFn`, `ParserObjOrFn`,
  `IgnoredParser`, `Tuple2Parser` to `Tuple5Parser`, `EndOfInput`, `IgnoredEndOfInput`, `IgnoredValue`, `State`,
  and the `State` fields' `Expectation` and `CheckFailure` (parser)
- `funcparserlib.__version__`
- `Token` is hashable, consistently with `==`: by type and value, ignoring positions

### Changed

- Error messages that named a sequence by its shape (`expected: ('x', 'y') or 'z'`) name what failed at the position
  instead (`expected: 'x' or 'z'`); `State` holds what failed there (`expected`), the active contexts and the recorded
  check failures, and a `ParsingFailure` can be `committed`
- Dropped support for Python 2.7 and 3.7 to 3.11
- A parsing function returns a `ParsingResult` instead of a `(value, state)` tuple and raising
  `NoParseError`. A result still unpacks as `value, state = p.run(tokens, state)`, and a failed one
  raises `NoParseError` when unpacked. A function written for 1.0 still works; see Deprecated.
- `State(pos, max, expected=(), context=(), problems=())`: the third argument is `expected`, the parsers that failed at
  `max`, and no longer `parser`. `State.parser` is still the last of them
- The value of `-p` no longer carries the value of `p`: `-x + -y` and `maybe(-x)` parse to `IgnoredValue()`, where 1.0
  gave `_Ignored('y')`
- `TokenSpec` compares by its type, pattern and flags instead of by identity, and a `Token` compared with a non-token
  is unequal instead of raising `AttributeError`
- `finished.named(name)` returns a new end-of-input parser; `finished` keeps its name in every other grammar
- `Token`, `TokenSpec`, `State` and the parsing results are immutable to type checkers. Every exported
  class except `Parser` is `@final` to type checkers.
- `Parser(f)` and `Parser.run` are public: a parsing function is what a grammar adds to the library. A subclass may
  add to a parser, and overriding `run` is not supported
- The type parameters use the Python 3.12 syntax, so `p1 + p2` and friends infer the same types in
  mypy, pyright and ty
- Built with uv and `uv_build`
- A name the library composes from the names of parsers is cut at 200 characters, with `...`: `Parser.name`, the parsing
  log and the `ValueError` of `many()` show the cut name. A name given with `Parser.named()` is kept whole
- `|` tries a parser it holds more than once a single time where it dispatches on the next token, so a function given
  to `>>` inside it can run fewer times. The value and the error stay the same

### Deprecated

- A parsing function that returns a `(value, state)` tuple or raises `NoParseError`, as in 1.0: `Parser(f)` and
  `ForwardDeclParser.define()` read its tuple as a success and its `NoParseError` as a failure that
  `|` backtracks from, and warn with a `DeprecationWarning` the first time each parser gets such a result

### Fixed

- A `parse()` inside a `>>` function no longer has its error rewritten by the outer `parse()`, which named the outer
  parse's token and repeated the message
- `p + -finished` is typed: mypy and pyright inferred `Any` for it (as in 1.0), because `-finished` was an
  `IgnoredParser[Any]` and matched both overloads of `+`. `finished` is an `EndOfInput` now, and `-finished` an
  `IgnoredEndOfInput`, whose value `+` drops
- A forward declaration defined as another one, `p.define(q)`, parses as `q` when `q` is defined later, instead of
  raising `NotImplementedError`
- A forward declaration defined as itself, `p.define(p)`, raises `ValueError` at `define()` instead of failing at
  `parse()`
- `many(p)` and `oneplus(p)` raise `ValueError` when `p` matches without consuming a token (`many(maybe(p))`,
  `many(pure(x))`, `many(many(p))`), instead of looping forever
- `make_tokenizer()` raises `ValueError` for a token spec that matches the empty string, and a tokenizer raises it for a
  spec that matches the empty string at some position (such as `\b`), instead of yielding empty tokens forever
- `LexerError.msg` is the line the error position counts: lines are split at `\n` only, as positions are, so a form
  feed, a lone `\r` or a Unicode line separator in a token no longer shifts it or raises `IndexError`

### Removed

- `with_forward_decls()`, deprecated since 0.3.1: use `forward_decl()` and `p.define()`
- `Parser.define()` outside forward declarations: `define()` is a method of `ForwardDeclParser`
- `funcparserlib.parser.log`: configure the `"funcparserlib"` logger by its name

## [1.0.1] — 2022-11-04

### Added

- Added support for Python 3.11

## [1.0.0] — 2022-05-02

The stable 1.0.0 release freezes the API of funcparserlib 0.3.6 which was released on
2013-05-02, with a few bug fixes and small features.

### Added

- Added support for Python 3.10
- Added support for Python 3.9
  ([#63](https://github.com/vlasovskikh/funcparserlib/pull/63))
  (Thanks to [@pkulev](https://github.com/pkulev))
- Added support for Python 3.8
- Added `-p` (the same as `skip(p)`) with more strict type hints for `-p` and `p1 + p2`
- Added `tok(type[, value])` for more compact grammars, better error messages
- Added `TokenSpec(type, pattern[, flags])` to simplify the use of `make_tokenizer()`
- Added type hints for the public API
- Added the new library homepage with the new Getting Started guide and the new API
  reference

### Changed

- Parse exceptions now show expected tokens and grammar rules at the stopped position
  ([#52](https://github.com/vlasovskikh/funcparserlib/issues/52))
- Dropped support for Python 3.4, 3.5, 3.6 (end of life)
- Dropped support for Python 2.5, 2.6, 3.3 (end of life), modernized code for Python
  3 to run without obsolete `2to3`
  ([#57](https://github.com/vlasovskikh/funcparserlib/pull/57))
  (Thanks to [@jdufresne](https://github.com/jdufresne))
- Removed documentation and unit tests from the distribution
- Switched from setuptools to Poetry
- Switched to poetry-core for lighter PEP 517 builds
  ([#73](https://github.com/vlasovskikh/funcparserlib/pull/73))
  (Thanks to [@fabaff](https://github.com/fabaff))
- Run unit tests on GitHub Actions for all supported Pythons

### Fixed

- Fixed `TypeError` in `oneplus` when applying it `parser + parser`
  ([#66](https://github.com/vlasovskikh/funcparserlib/issues/66))
  (Thanks to [@martica](https://github.com/martica))
- Fixed `AttributeError` when comparing `Token` objects to `None`
  ([#58](https://github.com/vlasovskikh/funcparserlib/pull/58))
  (Thanks to [@Halolegend94](https://github.com/Halolegend94))
- Fixed doctests in the tutorial
  ([#49](https://github.com/vlasovskikh/funcparserlib/issues/49))
- Fixed several cases of wrong expected tokens in error messages

## [0.3.6] — 2013-05-02

### Changed

- Python 3 compatibility
- More info available in exception objects
  ([#14](https://github.com/vlasovskikh/funcparserlib/issues/14))

### Fixed

- Fixed `many()` that consumed too many tokens in some cases
  ([#31](https://github.com/vlasovskikh/funcparserlib/issues/31))

## [0.3.5] — 2011-01-13

### Changed

- Python 2.4 compatibility
- More readable terminal names for error reporting

### Fixed

- Fixed wrong token positions in lexer error messages

## [0.3.4] — 2009-10-06

### Changed

- Switched from `setuptools` to `distutils`
- Improved the `run-tests` utility

### Fixed

- Fixed importing all symbols from `funcparserlib.lexer`

## [0.3.3] — 2009-08-03

### Added

- Added a FAQ question about infinite loops in parsers

### Changed

- Debug rule tracing can be enabled again

### Fixed

- Fixed a bug in results of skip + skip parsers

## [0.3.2] — 2009-07-26

### Added

- Added the Parsing Stages Illustrated page

### Fixed

- Fixed some string and number encoding issues in examples

## [0.3.1] — 2009-07-26

Major optimizations (10x faster than the version 0.3).

### Added

- Added the `forward_decl` function, that performs better than `with_forward_decls`
- Added the `pretty_tree` function for creating pseudo-graphic trees
- Added the Nested Brackets Mini-HOWTO
- Added `Makefile` and this `CHANGES.md` file

### Changed

- Use a single immutable input sequence in parsers
- Call a wrapped parser directly using `run` (without `__call__`)
- The slow `logging` is enabled only when the `debug` flag is set

## [0.3] — 2009-07-23

### Added

- Added `pure` and `bind` functions on `Parser`s making them monads
- Added the Funcparserlib Tutorial
- Added a JSON parser as an example

### Changed

- Translated the docs from Russian into English

## [0.2] — 2009-07-07

### Added

- Added the `with_forward_decls` combinator for dealing with forward declarations

### Changed

- Switched to the iterative implementation of `many`
- Un-curried the parser function type in order to simplify things
- Improvements to the DOT parser

## [0.1] — 2009-06-26

Initial release.
