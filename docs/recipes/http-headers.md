# Parsing HTTP Headers

Most HTTP headers are built from the same few pieces, given in [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110): a
token, a quoted string, parameters and a comma-separated list. This recipe writes those pieces once and four headers on
top of them. The whole program is `examples/http_headers.py`; every block below is included from it.

```pycon
>>> from examples.http_headers import accept, content_disposition, links, www_authenticate
>>> [(m.subtype, m.q) for m in accept.parse_text("text/html, application/json;q=0.5")]
[('html', 1.0), ('json', 0.5)]
>>> content_disposition.parse_text("attachment; filename*=UTF-8''%e2%82%ac.txt").filename
'€.txt'

```

## Why a grammar

A header looks like something to `split()`, and most of the time that works. It stops working at quotes, where a comma
or a semicolon is text:

```pycon
>>> header = 'application/json;profile="a,b";q=0.1, text/html;q=0.5'
>>> header.split(",")
['application/json;profile="a', 'b";q=0.1', ' text/html;q=0.5']
>>> [(m.subtype, m.q) for m in accept.parse_text(header)]
[('json', 0.1), ('html', 0.5)]

```

The first range lost its `q`, so a client that prefers HTML would get JSON, and a `q=0` that refuses a type would read
as a type accepted.

## When a split is enough

A header with no quoted strings has nothing for a grammar to do, and `Cookie` is such a header. [RFC
6265](https://www.rfc-editor.org/rfc/rfc6265#section-4.2.1) gives it a strict syntax, but a server receives what sites
have set and browsers have kept, so a strict grammar would reject real requests, as for a value with a space in it. A
split on `;` reads them all, and this is the whole parser:

```pycon
>>> def cookies(header):
...     found = {}
...     for pair in header.split(";"):
...         name, eq, value = pair.partition("=")
...         if eq and (name := name.strip()):
...             found.setdefault(name, value.strip().removeprefix('"').removesuffix('"'))
...     return found
>>> cookies('a=1; noeq; q="x y"; path=/; a=2')
{'a': '1', 'q': 'x y', 'path': '/'}

```

`http.cookies.SimpleCookie` is not a parser of this header. It reads `Set-Cookie` attributes such as `path` in it, and
gives up on the whole header at a pair it does not like, with no error:

```pycon
>>> from http.cookies import SimpleCookie
>>> SimpleCookie('a=1; noeq; b=2')
<SimpleCookie: >

```

## The pieces

`token` and `quoted` are the two kinds of value, and `parameters` a run of `;name=value`. A name compares in any case,
so it is read in lowercase. `parameter` ends in `starmap()` so that a pair stays a pair where it starts a sequence.

`comma_list()` is `separated()` with what RFC 9110 asks of a recipient: empty elements are to be ignored, at any place
in the list, so a separator is one or more commas, and one may come first and last.

```python
--8<-- "examples/http_headers.py:pieces"
```

## A media type

`Content-Type` is a media type, and there `q` is a parameter like any other. `Accept` is a list of media ranges, and
there `q` is a weight, with a syntax of its own that allows three decimals and nothing above 1. So the two are different
rules, and a weight is a rule of the grammar, which fails at the place it stops being one:

```python
--8<-- "examples/http_headers.py:media"
```

```pycon
>>> accept.parse_text("text/html;q=1.5")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,15-1,15: got unexpected token: '5', expected: whitespace or ';' or ',' or end of input

```

## A file name

`Content-Disposition` has the name of a file twice when the name is not ASCII: as `filename` for old clients, and as
`filename*` in the encoding of [RFC 8187](https://www.rfc-editor.org/rfc/rfc8187). The two differ in how they are
written, so they are two rules, and the extended one has to be tried first, as a token can end in `*`:

```python
--8<-- "examples/http_headers.py:disposition"
```

## A link

A `Link` header is how many APIs say where the next page is. What is between `<` and `>` is a URI reference, which has
commas and semicolons of its own, so the grammar reuses [`reference`](uri.md#a-reference) rather than a regex for it:

```python
--8<-- "examples/http_headers.py:link"
```

```pycon
>>> header = '<https://api.example/items?page=2>; rel="next", <https://api.example/items?page=9>; rel="last"'
>>> {link.parameters["rel"]: link.target.query for link in links.parse_text(header)}
{'next': 'page=2', 'last': 'page=9'}

```

An error in a target has the place in the header:

```pycon
>>> links.parse_text("<http://a b>; rel=next")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 1,10-1,10: got unexpected token: ' ', expected: '@' or ':' or path or '?' or '#' or '>'

```

## A challenge

`WWW-Authenticate` is the header a split cannot read. A server offers several ways to authenticate in one list, each
with its own list of parameters, and they use the same comma:

```text
Digest realm="a, b", qop="auth", Basic realm="c", Negotiate dGVzdA==
```

After `qop="auth",` comes a parameter of the same challenge or a new challenge, and only what follows the next token
says which: a parameter has an `=`, and a scheme has a space. A `token68` such as `abc=` can end in `=` as well.

In the grammar that is all ordinary backtracking. `separated()` reads the next parameter if there is one. If there is
none, it leaves the comma where it was, for `comma_list()` to read as the end of the challenge. `when(bool)` makes
parameters that are not there a failure, so that `token68` is tried:

```python
--8<-- "examples/http_headers.py:challenge"
```

```pycon
>>> header = 'Digest realm="a, b", qop="auth", Basic realm="c", Negotiate dGVzdA=='
>>> for c in www_authenticate.parse_text(header):
...     print(c.scheme, c.token68, c.parameters)
Digest None {'realm': 'a, b', 'qop': 'auth'}
Basic None {'realm': 'c'}
Negotiate dGVzdA== {}

```

## Parsing

Headers arrive with each request, so each is [compiled](../guide/performance.md) once, when the program starts:

```python
--8<-- "examples/http_headers.py:parse"
```

A new header defined with the rules of [Structured Fields](structured-fields.md) has a grammar of its own, and this
recipe's pieces do not apply to it.
