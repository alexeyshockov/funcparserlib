# Parsing URIs

[RFC 3986](https://www.rfc-editor.org/rfc/rfc3986) defines a URI by a grammar, in an appendix a page long. This recipe
writes that grammar down. The whole program is `examples/uri.py`; every block below is included from it.

```pycon
>>> from examples.uri import parse, parse_query
>>> uri = parse("https://user@example.com:8443/a/b?x=1&y=%E2%82%AC#top")
>>> uri.scheme, uri.path, uri.fragment
('https', '/a/b', 'top')
>>> uri.authority
Authority(userinfo='user', host='example.com', kind='name', port=8443)
>>> parse_query(uri.query)
[('x', '1'), ('y', '€')]

```

## Why a grammar

To split a URL you trust, use `urllib.parse.urlsplit()`. On a valid URI it finds the same parts, and it is faster.

It does not check them. `urlsplit()` takes almost any text as a URL, and which host it finds in a broken one is a guess
that the next program may not share:

```pycon
>>> from urllib.parse import urlsplit
>>> urlsplit("http://evil.example\\@good.example/").hostname
'good.example'
>>> urlsplit("http://exa mple.com/").hostname
'exa mple.com'
>>> parse("http://evil.example\\@good.example/")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,20-1,20: got unexpected token: '\\@good.example/', expected: '@' or ':' or path or '?' or '#' or end of input

```

So a grammar is for when the URI comes from someone else: it says whether this is a URI, where it stops being one, and
what kind of host it names.

## The tree

`None` is a part that is absent, and `""` a part that is empty, which are different things in a URI: `http://a?` has a
query, and `http://a` has none.

```python
--8<-- "examples/uri.py:tree"
```

## Characters

Most rules of the RFC are a set of characters, repeated. A regex says that best, so one rule here is one `regex()`, and
the parsers begin where a rule has a choice in it.

```python
--8<-- "examples/uri.py:characters"
```

## The authority

The authority is where the grammar earns its keep. Three things in it are not a matter of characters:

* The RFC tells an IPv4 address from a name by which rule matches, and `1.2.3.4.example` matches both up to a point.
  `longest()` keeps the longer match, and the first of them in a tie, so an address that is all there is stays one
* A userinfo reads the same characters as a host, and only a following `@` says which it was. The `maybe()` around it
  goes back to where it started when no `@` comes
* A port and an IPv6 address have a shape and a range. `Parser.validated()` rejects a value without failing the parser,
  and the check of an IPv6 address is one call to `ipaddress`

```python
--8<-- "examples/uri.py:authority"
```

A value that is out of range is a `ParseErrors`, with its place:

```pycon
>>> from funcparserlib.parser import ParseErrors
>>> try:
...     parse("http://example.com:99999/")
... except ParseErrors as e:
...     print(e.exceptions[0])
1,20-1,24: port 99999 is above 65535

```

## A reference

A URI has a scheme, and a relative reference has none, but a path that starts with `a:` would read as both. The RFC
settles it with a rule of its own for a first segment without a colon, and the two forms are then alternatives.

`with_authority` ends in `starmap()` because `+` adds to a tuple on its left: without it, the parts would be flat after
an authority and nested without one.

```python
--8<-- "examples/uri.py:reference"
```

## A query string

The RFC says nothing about what is in a query. The `name=value&name=value` of a web form is defined by the
[URL Standard](https://url.spec.whatwg.org/#urlencoded-parsing), and `urllib.parse.parse_qsl()` reads it with a few calls
of `split()`. To read a query as a browser does, that is the right tool, and a grammar
that accepts what it accepts adds nothing.

A grammar is worth it for the opposite of that, when a query that is not well-formed should be rejected. `parse_qsl()`
has no error to give. It leaves a broken escape as it is, replaces bytes that are not UTF-8, and gives a value with no
name to a name that is empty:

```pycon
>>> from urllib.parse import parse_qsl
>>> parse_qsl("a=%zz&b=%FF&=v")
[('a', '%zz'), ('b', '�'), ('', 'v')]

```

The rule of this grammar is that what loses nothing stays valid, and what would lose data or make it up is an error. An
`&` at the end or a doubled one changes no pair, and the URL Standard skips it. `=v` has a value that nothing can look
up, and `%zz` and `%FF` stand for no text, so a program that went on would be working with something else than what was
sent.

The grammar reads the bytes of each escape and decodes a name or a value as a whole, as a character can take several
escapes:

```python
--8<-- "examples/uri.py:query"
```

```pycon
>>> parse_query("a=1&&b=2&")
[('a', '1'), ('b', '2')]
>>> parse_query("a=%zz")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,3-1,3: got unexpected token: '%zz', expected: text or escape or '+' or '&' or end of input
>>> try:
...     parse_query("b=%FF")
... except ParseErrors as e:
...     print(e.exceptions[0])
1,3-1,5: not UTF-8 once percent-decoded: invalid start byte

```

An error quotes the text it stopped at, and a query can hold a secret. A server that answers or logs a refusal should
not repeat it, and `CompiledParser.try_parse()` gives `None` with no text in it, where an empty query is an empty list:

```pycon
>>> from examples.uri import compiled_pairs
>>> compiled_pairs.try_parse("=s3cret") is None
True
>>> compiled_pairs.try_parse("")
[]

```

## Parsing

A URI is often read per request, so the grammars are [compiled](../guide/performance.md) when the program starts:

```python
--8<-- "examples/uri.py:parse"
```

`reference` stays a parser without `finished`, so a grammar can use it for a URI inside something else.
[Parsing HTTP Headers](http-headers.md#a-link) does.
