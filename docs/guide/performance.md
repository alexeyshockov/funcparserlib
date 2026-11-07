# Performance

A parser is a tree of Python objects calling each other. For a config file or an expression typed by a person, that is
plenty fast. For large inputs, or for many of them, this chapter shows how to go faster. Compiling leaves your grammar as
it is, and you can stop at any step.

Let's use a tokenizer for numeric expressions again:

```pycon
>>> from funcparserlib.lexer import Token, TokenSpec, make_tokenizer
>>> from funcparserlib.parser import Parser, finished, many, maybe, tok

>>> tokenizer = make_tokenizer(
...     [
...         TokenSpec("whitespace", r"\s+"),
...         TokenSpec("int", r"\d+"),
...         TokenSpec("op", r"[*]"),
...     ],
...     skip={"whitespace"},
... )


>>> def tokenize(s: str) -> list[Token]:
...     return list(tokenizer(s))


>>> def op(name: str) -> Parser[Token, str]:
...     return tok("op", name)


>>> number = (tok("int") >> int).named("number")

```

## Compile a Grammar That Parses a Lot

[`Parser.compile()`](../api/parser.md#funcparserlib.parser.Parser.compile) generates Python functions from a finished
grammar. They run several times faster than the parsers do, and `parse()` works the same way:

```pycon
>>> expr = (number + many(-op("*") + number) + -finished).compile()

>>> expr.parse(tokenize("2 * 3 * 4"))
(2, [3, 4])

```

The functions keep no record of what was expected, so a parse that fails runs again in the parser it was compiled from.
Errors are the same as without compiling:

```pycon
>>> expr.parse(tokenize("2 * *"))
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected token: '*', expected: number

```

Compile a grammar once, when your program starts, and keep the result.

## Skip the Tokens

Given the tokenizer, a compiled grammar parses text itself. As long as the grammar reads tokens through `tok()` and the
specs of the tokenizer join into one regex, this makes no `Token` objects, which takes less time and much less memory.
Specs with their own flags, or with a global flag such as `(?i)` inside, do not join, and such a grammar makes the tokens
as usual:

```pycon
>>> expr = (number + many(-op("*") + number) + -finished).compile(tokenizer)

>>> expr.parse_text("2 * 3 * 4")
(2, [3, 4])

```

The token types to leave out have to be known here, so give them to `make_tokenizer()` as `skip`, as we did above. An
error says what was expected, also at text the tokenizer cannot read:

```pycon
>>> expr.parse_text("2 * x")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected token: 'x', expected: number

```

## Reject Invalid Input Cheaply

`parse_text()` makes a full error, and that costs more than the failed pass did. If the reason doesn't matter, such as for
a request that can only be logged and rejected, `CompiledParser.try_parse()` gives `None` and stops where the parse fails:

```pycon
>>> expr.try_parse("2 * 3")
(2, [3])

>>> expr.try_parse("2 * *") is None
True

```

A grammar whose value can be `None` cannot tell that value from a failure, so use `parse_text()` there. Text nested too
deeply gives `None` as well, where it would raise `RecursionError`.

## Know What Compiling Costs

Compiling is not free, which is why you ask for it:

* A failed parse takes longer, as it runs twice
* A compiled grammar is a snapshot: a `forward_decl()` defined again afterwards keeps its old meaning
* A rejected value or a failed `commit()` leaves the generated functions, so that parse takes the time of the parsers,
  and so does a document that needs `recover()`

The second run also calls the functions you gave to `>>` again:

```pycon
>>> calls = []


>>> def count(digits: str) -> int:
...     calls.append(digits)
...     return int(digits)


>>> digit = tok("int") >> count
>>> pair = (digit + digit + -finished).compile(tokenizer)

>>> pair.try_parse("1 2 3"), calls
(None, ['1', '2'])

>>> calls.clear()
>>> pair.parse_text("1 2 3")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,5-1,5: got unexpected token: '3', expected: end of input

>>> calls
['1', '2', '1', '2']

```

So compile a grammar when most of what it reads is valid and its `>>` functions only compute a value. A function that
must not run twice belongs after the parse, on the value it returned.

## Keep the Generated Code in Charge

A compiled grammar is only as fast as the parts that generated code can run:

* `some()`, `a()` or a parsing function read `Token` objects, so `parse_text()` has to tokenize first
* A parsing function and `Parser.bind()` run as the parsers do. For a length, a count or a type tag, use `prefixed()`,
  `counted()` and `switch()`, which compile
* A grammar stays interpreted if it is nested too deeply to generate, or has a `many()` of a parser that matches no tokens

To see what was generated, give `compile()` a file. Tracebacks, `pdb` and your IDE show that code from then on:

```python
expr = (number + many(-op("*") + number) + -finished).compile(source_file="generated_expr.py")
```

## Parse a Shared Prefix Once

When an alternative fails, `|` goes back to where it started and tries the next one. Alternatives that start the same
way parse their shared start once for each alternative tried:

```pycon
>>> p = (number + -op("*") + number) | number

>>> p.parse(tokenize("1"))
1

```

Here `number` runs twice on the `1`. It is cheap, but a shared start can be a large part of a grammar, and if it is
recursive the repeated work multiplies with each level of nesting.

Parse the shared start once and make the rest optional:

```pycon
>>> p = number + maybe(-op("*") + number)

>>> p.parse(tokenize("2 * 3"))
(2, 3)

>>> p.parse(tokenize("1"))
(1, None)

```

The result changes shape, so a function after `>>` has to tell the two cases apart.
[Parsing KDL](../recipes/kdl.md#the-grammar) does this for an entry of a node.

## How Fast Is It

These are figures from Python 3.15 on an arm64 laptop, so read them for their scale. The time is for the whole job, from
the input to Python values.

A 50 KB JSON document, the median of `benchmarks/bench_comparison.py`, where two runs differed by up to about 10%:

| Parser | Time |
| --- | ---: |
| orjson | 0.12 ms |
| stdlib `json` | 0.29 ms |
| funcparserlib, no tokenizer, compiled | 5.8 ms |
| funcparserlib, token grammar, compiled | 6.9 ms |
| funcparserlib, token grammar | 18.5 ms |
| lark, LALR | 23.8 ms |
| funcparserlib, no tokenizer | 25.1 ms |
| parsy | 57.2 ms |
| pyparsing | 72.8 ms |

A parser written for one format in C or Rust is out of reach for any grammar in Python. Among parser libraries, a compiled
grammar is about four times faster than lark's LALR parser and more than twelve times faster than pyparsing here.

A 461 KB MessagePack document, from the same benchmark:

| Parser | Time |
| --- | ---: |
| `msgpack`, its C extension | 2.8 ms |
| funcparserlib, compiled | 27 ms |
| `msgpack`, its pure Python fallback | 40 ms |
| funcparserlib | 92 ms |

A 1.8 MB KDL document, the document of `benchmarks/bench_grammars.py` 40 times over, best of three. The bad document fails
at its last token:

| Path | Good document | Peak memory | Bad document |
| --- | ---: | ---: | ---: |
| `parse()` | 3.27 s | 268 MB | 3.28 s |
| `compile()`, then `parse()` over tokens | 0.89 s | 287 MB | |
| `compile(tokenizer)`, then `parse_text()` | 0.47 s | 136 MB | 3.75 s |
| `compile(tokenizer)`, then `try_parse()` | | | 0.48 s |

The decisions behind these steps are in [ADR 0002](../adr/0002-a-compiled-pass-for-parses-that-succeed.md) and
[ADR 0003](../adr/0003-compiled-grammars-read-text-without-tokens.md).
