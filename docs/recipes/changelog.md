# Checking a Changelog

[Keep a Changelog](https://keepachangelog.com/) is a convention on top of Markdown. A Markdown parser reads any heading
as a heading, but the rules of the format are about which headings: a release is a version and a date, a change is one
of six kinds, and the newest release comes first. This recipe writes those rules as a grammar, and the changelog of this
library is checked with it. The whole program is `examples/changelog.py`; every block below is included from it.

```pycon
>>> from examples.changelog import parse
>>> releases = parse("""# Changelog
...
... ## [Unreleased]
...
... ## [1.0.0] - 2017-06-20
...
... ### Added
...
... - A feature
... """)
>>> releases[1].title.version.release, releases[1].title.date
((1, 0, 0), datetime.date(2017, 6, 20))
>>> releases[1].changes[0].kind
'Added'

```

## Why a grammar

markdown-it-py gives the Markdown: which lines are headings. The format is in what those headings say and in how they
follow each other. So there are two grammars here. One reads the tokens of markdown-it-py, and a `Text` grammar reads the
text of a heading, because a version has a grammar of its own.

## The tree

```python
--8<-- "examples/changelog.py:tree"
```

## A version

A version is not a number. The versions of this library follow [PEP 440](https://peps.python.org/pep-0440/), so it has a
grammar of its own, and `Version.order` sorts `2.0.0b1` before `2.0.0`:

```python
--8<-- "examples/changelog.py:version"
```

## A date

A date has a shape, and not every value of that shape is a day. `Parser.validated()` makes the `date` in one step, and
`read_day()` says what counts as invalid, so nothing after it sees `2020-13-01`:

```python
--8<-- "examples/changelog.py:date"
```

## What a heading says

The dash before the date has two spellings, and this library's changelog writes `—`. The six kinds of change are
alternatives. Each grammar ends with `text.end`, so a heading has to be all of it:

```python
--8<-- "examples/changelog.py:headings"
```

## Reading the tokens

`describe()` gives a token its line in the file, and `reading()` joins the two grammars: `Parser.validated()` runs a `Text`
grammar on the text of an `inline` token. When that grammar fails the token is rejected, so an error is on the line of the
file, `label` says what the heading should have been, and parsing goes on. `heading()` reads the source text, because
`[1.0.0]` is a link when the file has a reference for it and plain text when it does not.

The grammar has no use for the Markdown under a heading, so `text_under()` takes every token up to the next heading it
cares about, an image or a table included, and `####` under a kind of change stays text:

```python
--8<-- "examples/changelog.py:tokens"
```

## The grammar

A broken heading is rejected, so the release has no value and no function after it runs, but the rest of it is
still read, and every error in it is found. `Parser.recover()` closes the part: a broken release gives `None`, which the
list leaves out. The rules on real values raise `Invalid` with `keep=True`, so they can name what they found and parsing
goes on with the value:

```python
--8<-- "examples/changelog.py:grammar"
```

## Reporting the errors

```python
--8<-- "examples/changelog.py:parse"
```

`ParseErrors` holds everything the run found, and `render()` shows each error where it is:

```pycon
>>> from funcparserlib.parser import ParseErrors
>>> source = """# Changelog
...
... ## [2.0.0] - 2020-13-01
...
... ## [1.0.0] - 2019-01-01
...
... ### Updated
... """
>>> try:
...     parse(source)
... except ParseErrors as group:
...     print(group.render(source))
3,1-3,1: got unexpected token: '[2.0.0] - 2020-13-01', expected: `[version] - date` or `[Unreleased]`
3 | ## [2.0.0] - 2020-13-01
    ^
<BLANKLINE>
7,1-7,1: got unexpected token: 'Updated', expected: Added, Changed, Deprecated, Removed, Fixed or Security
7 | ### Updated
    ^

```

An error covers what its parser read, so a check on all the releases points at the first of them, whichever two are out
of order.

## Checking this repository

`tests/test_examples.py` runs the parser on this library's `CHANGELOG.md`, so a broken rule there fails the tests. By
hand, it is:

```shell
python -m examples.changelog CHANGELOG.md
```
