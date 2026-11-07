# Migrating a File Format

A text format that lives for a while changes, and each new version needs a way to bring the old files along. The change
is usually small: a link is written another way, a tag has a new name. Everything around it has to stay exactly as it
was, so parsing a file into a tree and writing it back is not an option, as a tree does not keep the blank lines and the
spacing of a hand-written file.

Here the old format is Markdown with two additions of its own, and the new format writes them differently. The whole
program is `examples/migration.py`; every block below is included from it.

```pycon
>>> from examples.migration import migrate
>>> print(migrate("See [[intro|the intro]],   @due(2026-11-07).\n"))
See [the intro](intro),   *due 2026-11-07*.
<BLANKLINE>

```

## Describe Only What Changes

A grammar for [`Parser.replace_text()`](../api/parser.md#funcparserlib.parser.Parser.replace_text) describes the
constructs and nothing else. It is tried at each position of the text, so the rest of the file needs no rules:

```python
--8<-- "examples/migration.py:constructs"
```

`commit()` says that what starts as a construct has to be one. Without it, a `[[` that goes wrong would simply be no
match, and the old syntax would stay in the new file with nothing said about it.

## Leave Some Text Alone

A regular expression would also rewrite the `[[` in a code block that documents the old format. A grammar can read such
a block as one more construct, and it goes first, so that nothing inside it is tried:

```python
--8<-- "examples/migration.py:fence"
```

Each match reaches `render()` as a [`Located`](../api/parser.md#funcparserlib.parser.Located), and a code block gives
back what it read:

```python
--8<-- "examples/migration.py:migrate"
```

```pycon
>>> migrate("```\n[[a|b]]\n```\n[[a|b]]\n")
'```\n[[a|b]]\n```\n[b](a)\n'

```

## Stop at What Is Broken

`replace_text()` gives a new text only when all of it could be read. A broken construct raises the error of a parse,
with its line and column, and a date the calendar does not have is rejected by the same rule:

```pycon
>>> migrate("fine\n[[intro the intro]]\n")
Traceback (most recent call last):
    ...
funcparserlib.parser.NoParseError: 2,18-2,18: got unexpected token: ']]', expected: '|'

```

So a file is written only after the whole of it was migrated:

```python
--8<-- "examples/migration.py:file"
```

A format with several versions has one such function for each step, and a file goes through the ones after its version.
