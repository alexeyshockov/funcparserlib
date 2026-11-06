# Checking Decision Records

An architecture decision record in [Nygard's format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
has a title, a status, and the sections Context, Decision and Consequences. The post names those parts and leaves their
Markdown open, so records in the wild differ, and a status is a closed set that can point at another record. This recipe
reads them from the tokens of markdown-it-py, with `Text` grammars for what the lines say, and checks the
[records of this library](../adr/index.md). The whole program is `examples/adr.py`; every block below is included from it.

```pycon
>>> from examples.adr import parse
>>> record = parse("""# 0001 — A title
...
... - **Status:** Superseded by ADR 0002
... - **Date:** 2026-02-20
...
... ## Context
...
... Forces.
...
... ## Decision
...
... We will.
...
... ## Consequences
...
... Results.
... """)
>>> record.number, record.title, record.status.superseded_by
(1, 'A title', 2)

```

## The tree

```python
--8<-- "examples/adr.py:tree"
```

## The parts

A status is a set of alternatives, and the one that supersedes carries the number it names, written out or as the link
adr-tools makes, so nothing has to read the text again. A title has three forms, as this library writes it, as adr-tools
does, and as the post does, and they share the number parser. `read_day()` makes the `date`, and says what is not a day:

```python
--8<-- "examples/adr.py:parts"
```

## Reading the tokens

`reading()` joins the two grammars: `Parser.validated()` runs a `Text` grammar on the text of an `inline` token. When that
grammar fails the token is rejected, so the error is on the line of the file and parsing goes on to the next field.
`titled()` uses `Parser.when()` instead: a heading with another name is another part of the record, not an invalid one:

```python
--8<-- "examples/adr.py:tokens"
```

## One record

The status and the date come in two forms too, so they are alternatives that give the same `Details`. Whatever Markdown
sits in a section is kept as text:

```python
--8<-- "examples/adr.py:grammar"
```

```python
--8<-- "examples/adr.py:parse"
```

Every field that is wrong is reported in one run, and `make_record()` never sees them:

```pycon
>>> from funcparserlib.parser import ParseErrors
>>> source = """# 0001 — A title
...
... - **Status:** Done
... - **Date:** 2026-02-30
...
... ## Context
...
... ## Decision
...
... ## Consequences
... """
>>> try:
...     parse(source)
... except ParseErrors as group:
...     print(group.render(source))
3,1-3,1: got unexpected token: '**Status:** Done', expected: **Status:** Proposed, Accepted, Deprecated or Superseded by ADR NNNN
3 | - **Status:** Done
    ^
<BLANKLINE>
4,1-4,1: got unexpected token: '**Date:** 2026-02-30', expected: **Date:** and a day of the calendar
4 | - **Date:** 2026-02-30
    ^

```

## The rules across records

A gap in the numbers is a fact about a directory, and no parse of one file can see it. So this part is plain Python over
the parsed records, which is the point of parsing them into a tree. A broken record is left out of that tree, which
would look like a gap, so these rules wait until every record parses:

```python
--8<-- "examples/adr.py:across"
```

## Checking this repository

`tests/test_examples.py` runs `check()` on `docs/adr/`, so a broken rule there fails the tests. By hand, it is:

```shell
python -m examples.adr docs/adr
```
