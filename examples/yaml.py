"""A YAML subset parser: ruamel.yaml scans, funcparserlib parses, a second pass resolves.

ruamel.yaml's scanner does the context-sensitive part of YAML: it turns indentation into block start
and end tokens, reads every scalar style and marks each implicit key with a `KeyToken`. The grammar
below reads that stream and builds a tree. `resolve()` then walks the tree in document order, which
gives aliases and merge keys their meaning, and turns scalars into values by the YAML 1.2 core
schema.

The subset is one document of block and flow collections, scalars, anchors, aliases and merge keys.
A tag, a directive, a second document or a key that is not a scalar fails with `NoParseError`,
naming the token and its place. A repeated key or an undefined alias fails with `ResolveError`. The
result is made of dicts, so keys that Python takes as equal, such as `1` and `true`, repeat.
"""

from __future__ import annotations

import dataclasses as dc
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.tokens import (
    AliasToken,
    AnchorToken,
    BlockEndToken,
    BlockEntryToken,
    BlockMappingStartToken,
    BlockSequenceStartToken,
    DocumentEndToken,
    DocumentStartToken,
    FlowEntryToken,
    FlowMappingEndToken,
    FlowMappingStartToken,
    FlowSequenceEndToken,
    FlowSequenceStartToken,
    KeyToken,
    ScalarToken,
    StreamEndToken,
    StreamStartToken,
    Token,
    ValueToken,
)

from funcparserlib.lexer import Place
from funcparserlib.parser import (
    ForwardDeclParser,
    Parser,
    TokenDescription,
    finished,
    forward_decl,
    many,
    maybe,
    some,
)


# --8<-- [start:tree]
@dataclass(frozen=True, slots=True)
class Scalar:
    """A scalar; only a `plain` one, without quotes, is resolved to anything but a string."""

    text: str
    plain: bool
    place: Place
    anchor: str | None = None


@dataclass(frozen=True, slots=True)
class Sequence:
    """A block or flow sequence."""

    items: list[Node]
    anchor: str | None = None


@dataclass(frozen=True, slots=True)
class Pair:
    """A mapping entry."""

    key: Scalar
    value: Node


@dataclass(frozen=True, slots=True)
class Mapping:
    """A block or flow mapping."""

    pairs: list[Pair]
    anchor: str | None = None


@dataclass(frozen=True, slots=True)
class Alias:
    """A reference to the node last anchored as `name`."""

    name: str
    place: Place


type Node = Scalar | Sequence | Mapping | Alias
# --8<-- [end:tree]


# --8<-- [start:token]
def token(cls: type[Token], name: str) -> Parser[Token, Any]:
    """Any one ruamel.yaml token of the class `cls`, called `name` in errors."""

    def is_cls(t: Token) -> bool:
        return t.__class__ is cls

    return some(is_cls).named(name)


def implicit(t: Token) -> bool:
    """Whether `t` is the empty `KeyToken` of a `key: value` entry, and not an explicit `?`."""
    return t.__class__ is KeyToken and t.start_mark.index == t.end_mark.index


# --8<-- [end:token]


# --8<-- [start:describe]
def place_of(t: Token) -> Place:
    """Where the token starts, counted from 1 as funcparserlib does."""
    return t.start_mark.line + 1, t.start_mark.column + 1


def describe(t: Token) -> TokenDescription:
    """How an error shows a token: its text, or the scanner's `id` for its class, and its place."""
    value = getattr(t, "value", None)
    text = value if isinstance(value, str) else getattr(t, "id", "token")
    return TokenDescription(text, place_of(t), place_of(t))


# --8<-- [end:describe]


# --8<-- [start:grammar]
def make_scalar(t: ScalarToken) -> Scalar:
    """A scalar from its token."""
    return Scalar(t.value, t.plain, place_of(t))


def make_alias(t: AliasToken) -> Alias:
    """An alias from its token."""
    return Alias(t.value, place_of(t))


def anchored[N: Scalar | Sequence | Mapping](anchor_and_node: tuple[Any, N]) -> N:
    """The node with the name of its `AnchorToken`, if it has one."""
    anchor, node = anchor_and_node
    return node if anchor is None else dc.replace(node, anchor=anchor.value)


def separated[A, B](p: Parser[A, B], comma: Parser[A, Any]) -> Parser[A, list[B]]:
    """Any number of `p`, separated by `comma`, with an optional one after the last."""
    return maybe(p + many(-comma + p) + -maybe(comma)) >> (
        lambda r: [] if r is None else [r[0], *r[1]]
    )


def null(node: Node | None) -> Node:
    """The node, or the empty scalar that a missing node means."""
    return Scalar("", True, (0, 0)) if node is None else node


anchor = token(AnchorToken, "anchor")
comma = token(FlowEntryToken, "','")
dash = token(BlockEntryToken, "'-'")
colon = token(ValueToken, "':'")
key_mark = some(implicit).named("key")

node: ForwardDeclParser[Token, Node] = forward_decl().named("node")
optional_node = maybe(node) >> null
alias = token(AliasToken, "alias") >> make_alias
scalar = token(ScalarToken, "scalar") >> make_scalar
key = maybe(anchor) + scalar >> anchored
value = -colon + optional_node

# A key on another line than its colon has no `KeyToken`, which only flow collections allow.
flow_pair = -maybe(key_mark) + key + (maybe(value) >> null) >> (lambda kv: Pair(*kv))
flow_map = (
    -token(FlowMappingStartToken, "'{'")
    + separated(flow_pair, comma)
    + -token(FlowMappingEndToken, "'}'")
    >> Mapping
)
single_pair = -key_mark + key + value >> (lambda kv: Mapping([Pair(*kv)]))
flow_seq = (
    -token(FlowSequenceStartToken, "'['")
    + separated(single_pair | node, comma)
    + -token(FlowSequenceEndToken, "']'")
    >> Sequence
)

items = many(-dash + optional_node)
block_seq = (
    -token(BlockSequenceStartToken, "block sequence") + items + -token(BlockEndToken, "block end")
    >> Sequence
)
# A sequence as a mapping value may sit at the mapping's own indentation, with no tokens around it.
indentless_seq = -dash + optional_node + items >> (lambda r: Sequence([r[0], *r[1]]))
block_value = (maybe(anchor) + indentless_seq >> anchored) | optional_node
block_pair = -key_mark + key + (maybe(-colon + block_value) >> null) >> (lambda kv: Pair(*kv))
block_map = (
    -token(BlockMappingStartToken, "block mapping")
    + many(block_pair)
    + -token(BlockEndToken, "block end")
    >> Mapping
)

content = scalar | flow_seq | flow_map | block_seq | block_map
empty = anchor >> (lambda t: dc.replace(null(None), anchor=t.value))
node.define(alias | (maybe(anchor) + content >> anchored) | empty)

document = (
    -token(StreamStartToken, "start of stream")
    + -maybe(token(DocumentStartToken, "'---'"))
    + optional_node
    + -maybe(token(DocumentEndToken, "'...'"))
    + -token(StreamEndToken, "end of document")
    + -finished
)
# --8<-- [end:grammar]


# --8<-- [start:schema]
CONSTANTS: dict[str, object] = {
    "": None,
    "~": None,
    "null": None,
    "Null": None,
    "NULL": None,
    "true": True,
    "True": True,
    "TRUE": True,
    "false": False,
    "False": False,
    "FALSE": False,
}
CONVERSIONS: list[tuple[re.Pattern[str], Callable[[str], object]]] = [
    (re.compile(r"[-+]?[0-9]+"), int),
    (re.compile(r"0o[0-7]+"), lambda s: int(s[2:], 8)),
    (re.compile(r"0x[0-9a-fA-F]+"), lambda s: int(s[2:], 16)),
    (re.compile(r"[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?"), float),
    (re.compile(r"[-+]?\.(inf|Inf|INF)|\.(nan|NaN|NAN)"), lambda s: float(s.replace(".", ""))),
]


def core_schema(text: str) -> object:
    """The value of a plain scalar by the YAML 1.2 core schema."""
    if text in CONSTANTS:
        return CONSTANTS[text]
    for pattern, convert in CONVERSIONS:
        if pattern.fullmatch(text):
            return convert(text)
    return text


# --8<-- [end:schema]


# --8<-- [start:resolve]
class ResolveError(Exception):
    """Raised for a document that parses and has no meaning, such as an undefined alias."""


def resolve(root: Node) -> object:
    """The value of the tree: lists, dicts and scalars, an alias sharing its anchor's object."""
    anchors: dict[str, object] = {}

    def merge(into: dict[object, object], source: object, place: Place) -> None:
        for mapping in source if isinstance(source, list) else [source]:
            if not isinstance(mapping, dict):
                raise ResolveError(f"merge value is not a mapping at {place}")
            for k, v in mapping.items():
                into.setdefault(k, v)

    def fill(result: dict[object, object], pairs: list[Pair]) -> None:
        merged: dict[object, object] | None = None
        for pair in pairs:
            if pair.key.plain and pair.key.text == "<<":
                if merged is not None:
                    raise ResolveError(f"repeated key '<<' at {pair.key.place}")
                merged = {}
                merge(merged, value_of(pair.value), pair.key.place)
                continue
            k = value_of(pair.key)
            if k in result:
                raise ResolveError(f"repeated key {k!r} at {pair.key.place}")
            result[k] = value_of(pair.value)
        for k, v in (merged or {}).items():
            result.setdefault(k, v)

    def value_of(n: Node) -> object:
        match n:
            case Alias(name, place):
                if name not in anchors:
                    raise ResolveError(f"undefined alias {name!r} at {place}")
                return anchors[name]
            case Scalar(text, plain, _, name):
                result = core_schema(text) if plain else text
                if name is not None:
                    anchors[name] = result
                return result
            case Sequence(items_, name):
                # Anchored before its items are read, as an item may be an alias to it.
                result_seq: list[object] = []
                if name is not None:
                    anchors[name] = result_seq
                result_seq.extend(value_of(item) for item in items_)
                return result_seq
            case Mapping(pairs, name):
                result_map: dict[object, object] = {}
                if name is not None:
                    anchors[name] = result_map
                fill(result_map, pairs)
                return result_map

    return value_of(root)


# --8<-- [end:resolve]


# --8<-- [start:parse]
def parse(text: str) -> object:
    """The value of the YAML document `text`.

    Raises `ruamel.yaml.error.YAMLError` for text the scanner rejects, `NoParseError` for what the
    grammar does not cover, `ResolveError` for a document without a meaning, and `RecursionError`
    for one nested too deeply.
    """
    tokens = list(YAML(typ="safe", pure=True).scan(text))
    return resolve(document.parse(tokens, describe=describe))


# --8<-- [end:parse]

if __name__ == "__main__":
    import sys
    from pprint import pprint

    pprint(parse(sys.stdin.read()))  # noqa: T203 — the program's output
