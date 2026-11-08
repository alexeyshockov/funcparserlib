"""The rules, one module per code family; importing this package registers them."""

from . import (
    adr,
    architecture,
    bounds,
    config,
    docs,
    gate,
    glossary,
    messages,
    names,
    release,
    sealing,
    source,
    surface,
)

__all__ = [
    "adr",
    "architecture",
    "bounds",
    "config",
    "docs",
    "gate",
    "glossary",
    "messages",
    "names",
    "release",
    "sealing",
    "source",
    "surface",
]
