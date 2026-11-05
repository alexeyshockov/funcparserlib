# Copyright © 2009/2023 Andrey Vlasovskikh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""A DOT language parser using funcparserlib.

The parser is based on [the DOT grammar][1]. It is pretty complete with a few not supported things:

* String escapes
* Ports and compass points
* XML identifiers

At the moment, the parser builds only a parse tree, not an abstract syntax tree (AST), or an API for
dealing with DOT.

  [1]: https://www.graphviz.org/doc/info/lang.html
"""

import os
import sys
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from re import MULTILINE
from typing import NamedTuple, TypeVar

from funcparserlib.lexer import LexerError, Token, TokenSpec, make_tokenizer
from funcparserlib.parser import (
    ForwardDeclParser,
    NoParseError,
    Parser,
    finished,
    forward_decl,
    many,
    maybe,
    oneplus,
    tok,
)
from funcparserlib.util import pretty_tree

ENCODING = "UTF-8"


class Graph(NamedTuple):
    strict: str | None
    type: str | None
    id: str | None
    stmts: list["Statement"]


class SubGraph(NamedTuple):
    id: str | None
    stmts: list["Statement"]


class Attr(NamedTuple):
    name: str
    value: str | None


class Node(NamedTuple):
    id: str
    attrs: list[Attr]


class Edge(NamedTuple):
    nodes: list[str | SubGraph]
    attrs: list[Attr]


class DefAttrs(NamedTuple):
    object: str
    attrs: list[Attr]


Statement = DefAttrs | Edge | SubGraph | Node


T = TypeVar("T")


def tokenize(s: str) -> Sequence[Token]:
    return list(_tokenizer()(s))


# Built once: the specs and the grammar do not depend on the input.
@cache
def _tokenizer() -> Callable[[str], Iterable[Token]]:
    """The DOT tokenizer."""
    specs = [
        TokenSpec("Comment", r"/\*(.|[\r\n])*?\*/", MULTILINE),
        TokenSpec("Comment", r"//.*"),
        TokenSpec("NL", r"[\r\n]+"),
        TokenSpec("Space", r"[ \t\r\n]+"),
        TokenSpec("Name", r"[A-Za-z\200-\377_][A-Za-z\200-\377_0-9]*"),
        TokenSpec("Op", r"[{};,=\[\]]|(->)|(--)"),
        TokenSpec("Number", r"-?(\.[0-9]+|[0-9]+(\.[0-9]*)?)"),
        TokenSpec("String", r'"[^"]*"'),  # '\"' escapes are ignored
    ]
    return make_tokenizer(specs, skip={"Comment", "NL", "Space"})


def parse(tokens: Sequence[Token]) -> Graph:
    return _grammar().parse(tokens)


@cache
def _grammar() -> Parser[Token, Graph]:
    """The DOT grammar, up to the end of input."""

    def un_arg(f: Callable[..., T]) -> Callable[[tuple], T]:
        return lambda args: f(*args)

    def flatten(xs: list[list[Attr]]) -> list[Attr]:
        return [x for attrs in xs for x in attrs]

    def n(s: str) -> Parser[Token, str]:
        return tok("Name", s)

    def op(s: str) -> Parser[Token, str]:
        return tok("Op", s)

    dot_id = (tok("Name") | tok("Number") | tok("String")).named("id")

    def make_graph_attr(args: tuple) -> DefAttrs:
        return DefAttrs("graph", [Attr(*args)])

    def make_edge(node: str | SubGraph, xs: list[str | SubGraph], attrs: list[Attr]) -> Edge:
        return Edge([node] + xs, attrs)

    node_id = dot_id  # + maybe(port)
    a_list = dot_id + maybe(-op("=") + dot_id) + -maybe(op(",")) >> un_arg(Attr)
    attr_list = many(-op("[") + many(a_list) + -op("]")) >> flatten
    attr_stmt = (n("graph") | n("node") | n("edge")) + attr_list >> un_arg(DefAttrs)
    graph_attr = dot_id + -op("=") + dot_id >> make_graph_attr
    node_stmt = node_id + attr_list >> un_arg(Node)
    # We use a forward_decl because of circular definitions like
    # (stmt_list -> stmt -> subgraph -> stmt_list)
    subgraph: ForwardDeclParser[Token, SubGraph] = forward_decl()
    edge_rhs = -(op("->") | op("--")) + (subgraph | node_id)
    edge_stmt = (subgraph | node_id) + oneplus(edge_rhs) + attr_list >> un_arg(make_edge)
    stmt = attr_stmt | edge_stmt | subgraph | graph_attr | node_stmt
    stmt_list = many(stmt + -maybe(op(";")))
    graph_body = -op("{") + stmt_list + -op("}")
    subgraph.define(-n("subgraph") + maybe(dot_id) + graph_body >> un_arg(SubGraph))
    graph_modifiers = maybe(n("strict")) + maybe(n("graph") | n("digraph"))
    graph = graph_modifiers + maybe(dot_id) + graph_body >> un_arg(Graph)
    return graph + -finished


def pretty_parse_tree(obj: object) -> str:
    class NamedValues(NamedTuple):
        name: str
        values: Sequence[object]

    def kids(x: object) -> Sequence[object]:
        if isinstance(x, (Graph, SubGraph)):
            return [NamedValues("stmts", x.stmts)]
        elif isinstance(x, (Node, DefAttrs)):
            return [NamedValues("attrs", x.attrs)]
        elif isinstance(x, Edge):
            return [NamedValues("nodes", x.nodes), NamedValues("attrs", x.attrs)]
        elif isinstance(x, NamedValues):
            return x.values
        else:
            return []

    def show(x: object) -> str:  # noqa: PLR0911 — one return per node type
        if isinstance(x, NamedValues):
            return x.name
        elif isinstance(x, Graph):
            return f"Graph [id={x.id}, strict={x.strict is not None!r}, type={x.type}]"
        elif isinstance(x, SubGraph):
            return f"SubGraph [id={x.id}]"
        elif isinstance(x, Edge):
            return "Edge"
        elif isinstance(x, Attr):
            return f"Attr [name={x.name}, value={x.value}]"
        elif isinstance(x, DefAttrs):
            return f"DefAttrs [object={x.object}]"
        elif isinstance(x, Node):
            return f"Node [id={x.id}]"
        else:
            return str(x)

    return pretty_tree(obj, kids, show)


def main() -> None:
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # import funcparserlib
    # funcparserlib.parser.debug = True
    try:
        stdin = os.fdopen(sys.stdin.fileno(), "rb")
        text = stdin.read().decode(ENCODING)
        tree = parse(tokenize(text))
        # print(pformat(tree))
        print(pretty_parse_tree(tree).encode(ENCODING))  # noqa: T201 — the CLI's output
    except (NoParseError, LexerError) as e:
        msg = (f"syntax error: {e}").encode(ENCODING)
        print(msg, file=sys.stderr)  # noqa: T201 — the CLI's output
        sys.exit(1)


if __name__ == "__main__":
    main()
