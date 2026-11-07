# Other Libraries

Python has several good parsing libraries. This page says where funcparserlib differs from the ones you are most likely
to know, when one of them fits your problem better, and how their names map to ours.

| Library | A grammar is | How it parses |
| --- | --- | --- |
| funcparserlib | Python expressions, over tokens, text or bytes | Recursive descent with backtracking, or generated code |
| [parsy](https://parsy.readthedocs.io/) | Python expressions, over text, bytes or lists | Recursive descent with backtracking |
| [pyparsing](https://pyparsing-docs.readthedocs.io/) | Python expressions, over text | Recursive descent with backtracking |
| [Lark](https://lark-parser.readthedocs.io/) | Grammar text in EBNF | Earley or LALR(1), to a tree |

[How Fast Is It](performance.md#how-fast-is-it) compares their speed on one JSON document.

Two more are worth knowing, and we have not measured either. [parsita](https://parsita.drhagen.com/) is the closest to
funcparserlib: typed combinators, where a grammar is the body of a class. [Parsimonious](https://github.com/erikrose/parsimonious)
reads a PEG grammar from text and leaves the tree to a visitor class.

## When Another Library Fits Better

* Your grammar is left-recursive or ambiguous, or you already have it in EBNF: use Lark. Its Earley parser takes any
  context-free grammar, where funcparserlib needs [left recursion written out](../getting-started/tips-and-tricks.md#watch-out-for-left-recursion)
* You want a railroad diagram of your grammar: pyparsing draws one with `create_diagram()`
* You need to write a binary format as well as read it: [Construct](https://construct.readthedocs.io/) does both from one
  definition, and [`funcparserlib.binary`](../api/binary.md) only reads
* A parser already exists for your format: use it. Nothing in pure Python gets close to a JSON parser written in C

## When funcparserlib Fits

* You want a type checker to follow your grammar. `p1 + p2` is a typed tuple and `p >> f` takes the type of `f`, so a
  mistake in the shape of a value shows up in your editor. A parsy parser has no type of its value, pyparsing gives a
  `ParseResults`, and Lark a tree
* You want your errors to name everything that was expected, and to [report every error in one run](errors.md#report-every-error-in-one-run)
* The input is already split up: funcparserlib parses a sequence of any objects, such as
  [the tokens of another library](../recipes/markdown.md)
* You want speed without a second grammar language: [`compile()`](performance.md#compile-a-grammar-that-parses-a-lot)
  generates Python functions from the same grammar
* You don't want dependencies: funcparserlib has none

## The Same Thing in Each Library

| funcparserlib | parsy | pyparsing | Lark |
| --- | --- | --- | --- |
| `p1 + p2` | `seq(p1, p2)` | `p1 + p2` | `p1 p2` |
| `p1 \| p2` | `p1 \| p2` | `p1 \| p2` | `p1 \| p2`, with no order |
| `longest(p1, p2)` | | `p1 ^ p2` | |
| `p >> f` | `p.map(f)` | `p.set_parse_action(f)` | A method of a `Transformer` |
| `-p` | `p1 >> p2`, `p1 << p2` | `p.suppress()` | `_rule`, and strings are left out |
| `many(p)` | `p.many()` | `p[...]` | `p*` |
| `oneplus(p)` | `p.at_least(1)` | `p[1, ...]` | `p+` |
| `maybe(p)` | `p.optional()` | `Opt(p)` | `p?` |
| `separated(p, sep)` | `p.sep_by(sep)` | `DelimitedList(p, sep)` | `p (sep p)*` |
| `forward_decl()`, `define()` | `forward_declaration()`, `become()` | `Forward()`, `<<=` | A rule that names itself |
| `p.named("x")` | `p.desc("x")` | `p.set_name("x")` | The name of the rule |
| `commit(p)` | | `p1 - p2` | |
| `skipping(r"\s+")` | Written by hand | The default | `%ignore` |
| `p + -finished` | The default | `parse_all=True` | The default |
| `p.located()` | `p.mark()` | `Located(p)` | `propagate_positions=True` |
| `p.scan_text(text)`, `p.scan(tokens)` | | `p.scan_string(text)` | |
| `p.replace_text(text, f)` | | `p.transform_string(text)` | |

## Coming From parsy

The operators look the same and mean something else. In parsy, `+` joins two results and `>>` keeps the right one. Here
`+` makes a tuple and `>>` applies a function:

```pycon
>>> from funcparserlib.parser import lit

>>> (lit("a") + lit("b")).parse_text("ab")
('a', 'b')

>>> (lit("a") >> str.upper).parse_text("a")
'A'

>>> (-lit("a") + lit("b")).parse_text("ab")
'b'

```

For what you would write with `@generate`, there is [`Parser.bind()`](../api/parser.md#extra-parser-monad). A length, a
count or a type tag is better read with `prefixed()`, `counted()` and `switch()`, which also compile.

## Coming From pyparsing

Whitespace is not skipped unless you say so. Use `skip` of [`make_tokenizer()`](../api/lexer.md) in a token grammar, or
the parsers of [`skipping()`](without-a-tokenizer.md#a-first-grammar-without-a-tokenizer) in a grammar without a
tokenizer.

The results are Python values, with no `ParseResults` in between. What you would keep with a results name is the
position in a tuple, and `p >> f` makes it a class of yours, as [Preparing the Parse
Tree](../getting-started/parse-tree.md) shows.

## Coming From Lark

A grammar is a Python expression that makes the value itself, so there is no tree to transform unless you want one. It
has no left recursion: where Lark has `sum: sum "+" term | term`, write `term + many(-lit("+") + term)` and fold the
list in a function after `>>`, as the [calculator of Getting Started](../getting-started/index.md#diving-in) does.

A parser goes back and tries the next alternative, so `|` has an order, and the first alternative that succeeds wins.
[Tips and Tricks](../getting-started/tips-and-tricks.md#how-to-handle-conflicting-alternatives) shows what it means for
a grammar, and `longest()` takes the order away.
