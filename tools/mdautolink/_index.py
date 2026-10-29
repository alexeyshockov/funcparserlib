"""The names a page may write for a documented object, read with griffe from the rendered ones."""

# Generated; propose changes in an issue rather than editing this file.

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import griffe

DIRECTIVE = re.compile(r"^::: ([\w.]+)")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class NameIndex:
    """Each spelling a page may write, mapped to the identifier the reference registers for it.

    `ambiguous` holds the spellings that name more than one object; one links only from a page
    scoped to the module of exactly one of them.
    """

    targets: Mapping[str, str]
    ambiguous: Mapping[str, frozenset[str]]

    def resolve(self, name: str, prefer: str | None = None) -> str | None:
        """The identifier `name` links to, or `None`."""
        if (target := self.targets.get(name)) is not None:
            return target
        if prefer is None or name not in self.ambiguous:
            return None
        preferred = [t for t in self.ambiguous[name] if t == prefer or t.startswith(prefer + ".")]
        return preferred[0] if len(preferred) == 1 else None


def _directives(text: str) -> Iterable[str]:
    """The modules `text` renders with `:::`, outside fenced blocks, where one is an example."""
    fence: str | None = None
    for line in text.splitlines():
        if opened := FENCE.match(line):
            mark = opened[1]
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = None
            continue
        if fence is None and (directive := DIRECTIVE.match(line)):
            yield directive[1]


def rendered_modules(docs_dir: str | Path) -> tuple[str, ...]:
    """The module paths the pages under `docs_dir` render with mkdocstrings' `:::` directive."""
    pages = Path(docs_dir).rglob("*.md")
    return tuple(sorted({m for page in pages for m in _directives(page.read_text())}))


def _spellings(module_path: str, name: str) -> Iterable[str]:
    yield name
    yield f"{module_path}.{name}"
    # The module's last segment, as a page writes a function of a namespace module: `tcp.connect`.
    if "." in module_path:
        yield f"{module_path.rsplit('.', 1)[1]}.{name}"


def build_index(modules: Iterable[str], search_paths: Iterable[str]) -> NameIndex:
    """Index the exports of `modules`, and the public members of the classes among them.

    A member is indexed with its class (`Pool.take`), never alone: a bare `take` is too common a
    word to link.
    """
    loader = griffe.GriffeLoader(search_paths=list(search_paths))
    # spelling -> {(identifier, canonical path)}
    candidates: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    loaded = {path: loader.load(path) for path in modules}
    for module_path, module in loaded.items():
        # A `:::` may render a class alone; only a module has exports to index.
        if not isinstance(module, griffe.Module):
            continue
        # `__all__ += other.__all__` names another module's list until expanded; griffe's own load
        # expands it only below a top-level package that has an `__all__`.
        loader.expand_exports(module)
        # A reference expansion could not follow stays an expression, and is not a name.
        names = sorted(n for n in module.exports or () if isinstance(n, str))
        for name in names:
            if name.startswith("_") or name not in module.members:
                continue
            obj = module[name]
            full = f"{module_path}.{name}"
            try:
                canonical = obj.canonical_path
                members = obj.all_members if obj.is_class else {}
            except griffe.AliasResolutionError:
                canonical, members = full, {}
            for spelling in _spellings(module_path, name):
                candidates[spelling].add((full, canonical))
            for member in members:
                if not member.startswith("_"):
                    entry = (f"{full}.{member}", f"{canonical}.{member}")
                    for spelling in _spellings(module_path, name):
                        candidates[f"{spelling}.{member}"].add(entry)
    targets: dict[str, str] = {}
    ambiguous: dict[str, frozenset[str]] = {}
    for spelling, found in candidates.items():
        if len({canonical for _, canonical in found}) == 1:
            # One object exported from several modules: its shortest public path.
            targets[spelling] = min((t for t, _ in found), key=lambda t: (t.count("."), t))
        else:
            ambiguous[spelling] = frozenset(t for t, _ in found)
    return NameIndex(targets, ambiguous)
