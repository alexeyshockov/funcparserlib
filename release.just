# The release recipes.
# Generated; propose changes in an issue rather than editing this file.
#
# The tree always holds the next release's development version, X.Y.Z.devN, in every
# distribution, every `==` pin between them. A release changes no file:
#
#   just release-check           # type coverage + API BC vs the previous stable tag
#   <land the CHANGELOG section, if there is a CHANGELOG>
#   just release X.Y.Z           # check the tree, open a draft GitHub release of this commit
#   <review the draft on GitHub, click Publish>  -> .github/workflows/pypi-publish.yaml
#   just version X.Y.Z+1.dev0    # the next version, committed by you whenever it is chosen

# Latest tag matching strictly `vX.Y.Z` (no `b1` / `rc1` / `.dev0` suffix).
previous_stable_tag := `git tag --sort=-v:refname --list 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -n1`

# Exit 1, a change found, is informational: a change is read and decided on, not refused. Exit 2,
# a base it cannot read or a crash, fails the recipe.
[doc("Show what a caller of the public API at [base] (default: previous stable tag) meets now")]
api-diff base=previous_stable_tag:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{ base }}" ]; then
        echo "no previous stable tag found; pass [base] explicitly" >&2
        exit 1
    fi
    {{ tools }} conventions api-diff "{{ base }}" || [ $? -eq 1 ]

[doc("Pre-release report: current type coverage + API BC vs [base]")]
release-check base=previous_stable_tag:
    @just type-coverage
    @echo
    @just api-diff {{ base }}

# `uv lock` records each distribution's version, so a new version is relocked with it.
[doc("Print the version, or write VERSION into every distribution and pin, and relock; commits nothing")]
version *version:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{ version }}" ]; then
        {{ tools }} conventions version
        exit
    fi
    {{ tools }} conventions version "{{ version }}"
    uv lock
    if [ -f Cargo.toml ]; then cargo update -w --offline --quiet; fi

[doc("Check the tree may release X.Y.Z, then open a draft GitHub release of the pushed HEAD")]
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "$(git status --porcelain)" ]; then
        echo "working tree dirty; commit or stash first" >&2
        exit 1
    fi
    if [ "$(git rev-parse --abbrev-ref HEAD)" != "master" ]; then
        echo "not on master; checkout master first" >&2
        exit 1
    fi
    git fetch --quiet origin master
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master)" ]; then
        echo "HEAD is not origin/master; push or pull first" >&2
        exit 1
    fi
    {{ tools }} conventions release "v{{ version }}"
    notes=""
    if [ -f CHANGELOG.md ]; then
        notes=$(awk -v v="{{ version }}" '$0 ~ "^## \\[" v "\\]" {flag=1; next} /^## \[/ {flag=0} flag' CHANGELOG.md)
    fi
    gh release create "v{{ version }}" --target "$(git rev-parse HEAD)" --draft \
        --title "v{{ version }}" --notes "$notes"
    echo
    echo "Draft release created. Review on GitHub and click Publish to trigger the PyPI workflow."
