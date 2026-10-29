"""Link inline code that names a documented Python object to its API reference.

A Python-Markdown extension for a site whose reference mkdocstrings renders: `` `Pool` `` becomes
the link ``[`Pool`][pkg.Pool]`` would, resolved by autorefs once the site is built. A name the
reference has no entry for stays plain, and so does code marked `{.nolink}`.
"""

# Generated; propose changes in an issue rather than editing this file.

from mdautolink._extension import AutolinkExtension, makeExtension
from mdautolink._index import NameIndex, build_index, rendered_modules

__all__ = ["AutolinkExtension", "NameIndex", "build_index", "makeExtension", "rendered_modules"]
