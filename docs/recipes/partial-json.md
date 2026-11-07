# Parsing JSON from a Language Model

A language model writes JSON a few characters at a time, and a program that shows the answer while it comes needs a
value for every prefix of the document. A model also writes things in place of JSON: a comma before a closing bracket,
single quotes, `True` and `None`, a Markdown code fence around the document. This recipe reads both with one grammar,
then adds two things a program does with a reply while it streams: [telling the finished parts](#which-parts-are-finished)
from the ones still being written, and [finding the document](#the-document-inside-a-reply) in the prose around it. The
whole program is `examples/partial_json.py`; every block below is included from it.

```pycon
>>> from examples.partial_json import parse
>>> parse('{"title": "Pars')
{'title': 'Pars'}
>>> parse('{"title": "Parsing", "tags": ["json", "llm"')
{'title': 'Parsing', 'tags': ['json', 'llm']}
>>> parse("```json\n{'done': True,}\n```")
{'done': True}

```

## What a prefix stands for

A prefix stands for what it has finished, and a value a program has shown only grows: a list gets items, an object gets
members, and the string at the end gets longer. Anything the next characters may still change is left out until
something ends it:

* a number at the end, as `2` may become `20` and `1.` is no number yet
* a literal at the end, as `tr` of `true`
* a key without its value
* an escape the text ends inside, as `\u00` of `é`, and the first half of a surrogate pair

```pycon
>>> parse('{"scores": [1.5, 2')
{'scores': [1.5]}
>>> parse('{"scores": [1.5, 20], "ok": tr')
{'scores': [1.5, 20]}
>>> parse('{"name": "Ren\\u00')
{'name': 'Ren'}

```

`pydantic_core.from_json(text, allow_partial="trailing-strings")` reads a prefix too, and gives a number at the end as it
stands. Over the prefixes of `{"a": [1.5, 20], "b": -3e2}`, pydantic_core 2.50.4 gives these values in turn, each a
value the document does not hold:

```text
{"a": [1        ->  {'a': [1]}
{"a": [1.       ->  {'a': []}
{"a": [1.5, 2   ->  {'a': [1.5, 2]}
..., "b": -3    ->  {'a': [1.5, 20], 'b': -3}
..., "b": -3e   ->  {'a': [1.5, 20]}
```

The document is an object or an array, so every number in it is ended by what follows it. A text that ends inside a
number, a model's answer cut off at its token limit, leaves that number out.

## Values

`MISSING` is the value of a part the text has not finished. The containers leave it out, and `parse()` gives it before
the document has an opening bracket:

```python
--8<-- "examples/partial_json.py:values"
```

## Tokens

The end of a prefix is where a token can be unfinished, so each token that can be cut off has a second spec, anchored
at the end of the text with `\Z`. The tokenizer takes the first spec that matches, so a number at the end is an
`unfinished_number` and never a `number`. A spec that has to match a complete token stays before its unfinished
variant, as `string` before `unfinished_string`.

```python
--8<-- "examples/partial_json.py:tokens"
```

## Scalars

A string the text ends inside is given as far as it goes, without an escape it ends inside. A name that begins a
literal is unfinished only at the end of the text: `+ -finished` reads nothing, and fails anywhere else.

```python
--8<-- "examples/partial_json.py:scalars"
```

## Containers

Every open bracket closes at the end of the text: `close()` reads its bracket or `finished`, which matches there without
reading a token, and says which of the two it read. `separated(..., trailing=True)` reads one comma after the last item,
which is a prefix that stops after a comma and a model's trailing comma alike. A member whose value is unfinished is left
out of its object, and so is an unfinished item of an array.

```python
--8<-- "examples/partial_json.py:containers"
```

## What stays an error

A repair is an alternative of the grammar where the input has one meaning. Where it has more, the grammar stops, and the
error says where:

```pycon
>>> parse("[1 2]")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,4-1,4: got unexpected token: '2', expected: ',' or ']' or end of input
>>> parse('{"a": 1} and more')
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,10-1,12: got unexpected token: 'and', expected: fence or end of input

```

`[1 2]` may be a list of two numbers with a comma missing or a list of one with a stray token, and text after the
document may be prose or a second document. A grammar that wants one of those readings says it with one more
alternative.

## Streaming

`parse()` reads the whole prefix each time, so a program calls it with what it has received so far:

```python
--8<-- "examples/partial_json.py:parse"
```

The grammar is compiled, as every prefix of a well-formed document parses, which is what the [compiled](../guide/performance.md)
pass is for. On a 10.5 KB document read every 20 characters, 529 prefixes, Python 3.12 on an arm64 laptop, CPU time,
best of three, the compiled grammar took 0.289 s, 0.55 ms a prefix, and the grammar without `compile()` 0.918 s.
pydantic_core, written in Rust, took 6.8 ms for the same prefixes. Reading every prefix costs time that grows with the
square of the length of the document, which suits the answers of a model, that come at the speed the model writes.

## Which parts are finished

A value of a prefix holds parts the rest of the text may still add to, and parts it cannot change. A program that acts
while the reply streams needs to tell them apart: it runs a tool call once its object is closed, and shows a field as
final once its string is. pydantic_core gives `'fe'` for a prefix ending in `"fe` and `'fetch'` for one ending in
`"fetch"`, and nothing in the value says that one of them is done.

The grammar knows it already: `close()` reads a bracket or the end of the text, and a string is a `string` or an
`unfinished_string` token. A string, an array or an object the text ends inside is given as an `Open` subclass of its
type:

```python
--8<-- "examples/partial_json.py:open"
```

`unfinished_string_of()` gives an `OpenStr`, and `make_object()` and `make_array()` give an `OpenDict` and an `OpenList`
where `close()` read no bracket. A value is still a `str`, a `list` or a `dict`, and compares, prints and serializes as
one. The text ends inside one part at each level, so the open parts are a chain from the document to the last part
reached, and everything else is finished:

```pycon
>>> from examples.partial_json import Open
>>> reply = parse('{"calls": [{"tool": "search", "query": "parsers"}, {"tool": "fe')
>>> [call for call in reply["calls"] if not isinstance(call, Open)]
[{'tool': 'search', 'query': 'parsers'}]
>>> isinstance(reply["calls"][1]["tool"], Open)
True

```

A finished part stays as it is in every longer prefix, so a program that ran the first call does not see it change.

An `Open` value prints as its type does, so `print()` and `pprint()` show none of this. `tree()` draws a value with
`pretty_tree()`, and marks each open part:

```python
--8<-- "examples/partial_json.py:tree"
```

```pycon
>>> from examples.partial_json import tree
>>> print(tree(reply))
{}  (open)
`-- 'calls': []  (open)
    |-- {}
    |   |-- 'tool': 'search'
    |   `-- 'query': 'parsers'
    `-- {}  (open)
        `-- 'tool': 'fe'  (open)

```

`python -m examples.partial_json` prints the tree of what it reads from standard input.

## The document inside a reply

A model asked for JSON often writes a sentence around it. `Parser.scan_text()` finds each match of a grammar in a text,
and `find()` gives the objects in a reply with where each one is, the last one possibly unfinished:

```python
--8<-- "examples/partial_json.py:find"
```

```pycon
>>> from examples.partial_json import find
>>> reply = 'Here\'s the plan: {"steps": ["parse", "test"]} Let me know if it\'s OK. {"draft": "rea'
>>> [(m.start, m.value) for m in find(reply)]
[(17, {'steps': ['parse', 'test']}), (70, {'draft': 'rea'})]

```

A grammar that starts with `tokenized()` is tried where a token of the text starts, as the tokenizer reads the text from
its start. The tokenizer of this recipe reads single quotes, so the `'s the plan: {...} Let me know if it'` of a reply
would be one string token, and the object inside it no match. `find()` starts with `regex(r"\{")`, a parser of text:
the grammar is tried at each offset, and the tokenizer starts after the brace. It finds objects only, as a reply that
cites `[1]` would give that as an array.

## Changing a rule

Each rule is one line of the grammar. To give a number at the end of the text as it stands, as pydantic_core does, take
`{NUMBER}` out of the `unfinished_number` spec, and a complete number at the end becomes a `number` token. To read keys
without quotes, `member` reads `string | tok("name")` where it reads `string`.
