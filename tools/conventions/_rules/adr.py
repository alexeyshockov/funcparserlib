"""CON3xx: ADR citations."""

import re
from collections.abc import Iterator
from pathlib import Path

from .._engine import Finding, rule
from .._project import Project
from ._ast import line_at

# The one accepted spelling.
CITATION = re.compile(r"\bADR (\d{4})\b")
# Only the near misses (hyphen, underscore, no separator, lowercase), never the canonical form.
LOOSE = re.compile(r"\bADR[-_]\d{4}\b|\bADR\d{4}\b|\badr[ _-]\d{4}\b")
# A citation of a decision recorded outside this repository: a link whose label carries the number
# and whose target is an absolute URL. Masked before the bare scan, because its number is not this
# repository's.
FOREIGN = re.compile(r"\[[^\]]*?ADR \d{4}[^\]]*?\]\(https?://[^)]*\)")
# A Markdown link whose label carries a number and whose target is an ADR document, local or
# published.
LINKED = re.compile(r"\[[^\]]*?ADR (\d{4})[^\]]*?\]\(([^)]*?/(\d{4})-[a-z0-9-]+(?:\.md|/)?)\)")


def _texts(project: Project) -> Iterator[tuple[Path, str]]:
    for path in project.files:
        if path.suffix in {".py", ".md"} and (text := project.text(path)) is not None:
            yield path, text


def _blank(match: re.Match[str]) -> str:
    """The match with every character but newlines blanked, so offsets keep their lines."""
    return re.sub(r"[^\n]", " ", match.group())


def allocated(project: Project) -> set[str]:
    """The ADR numbers this repository has files for."""
    return {p.name[:4] for p in (project.root / "docs" / "adr").glob("0[0-9][0-9][0-9]-*.md")}


@rule("CON300", "CON301", "CON302", "CON303")
def citations(project: Project) -> Iterator[Finding]:
    """An ADR citation resolves to an ADR in this repository.

    CON300: A bare `ADR NNNN` names a file under `docs/adr/`.

    Reason: an ADR number is an address only inside the repository that allocated it. A number with
    no file points at nothing, or at whatever later takes the number.

    Fix: cite the right number. A decision recorded outside this repository is a link to its
    published page, every time it is mentioned: `[ADR NNNN](https://…/adr/NNNN-…/)`.

    CON301: An ADR citation is spelled `ADR NNNN`.

    Reason: CON300 reads one spelling; a hyphenated or unspaced citation passes it unread.

    Fix: write `ADR` and the four digits with one space.

    CON302: A linked citation's number agrees with the document it links to.

    Reason: a link labelled with one number and pointing at another renders, resolves, and names the
    wrong decision. Renumbering is when the two drift apart.

    Fix: correct the label or the target.

    CON303: The repository has a decision record under `docs/adr/`.

    Reason: without it every citation dangles; this says so once instead of once per file.

    Fix: add `docs/adr/`, or remove the citations.
    """
    known = allocated(project)
    if not known:
        yield Finding("docs/adr", None, "CON303", "no ADRs under docs/adr/")
        return
    for path, text in _texts(project):
        relative = project.relative(path)
        bare = FOREIGN.sub(_blank, text)
        for match in CITATION.finditer(bare):
            if match.group(1) not in known:
                yield Finding(
                    relative,
                    line_at(text, match.start()),
                    "CON300",
                    f"ADR {match.group(1)} names no file under docs/adr/",
                )
        for match in LOOSE.finditer(text):
            yield Finding(
                relative,
                line_at(text, match.start()),
                "CON301",
                f"`{match.group()}` should be spelled `ADR NNNN`",
            )
        for match in LINKED.finditer(text):
            label, target, number = match.groups()
            if label != number:
                yield Finding(
                    relative,
                    line_at(text, match.start()),
                    "CON302",
                    f"ADR {label} links to {target}",
                )
