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

[doc("Show API breaking changes vs [base] (default: previous stable tag) using griffe")]
api-diff base=previous_stable_tag:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{ base }}" ]; then
        echo "no previous stable tag found; pass [base] explicitly" >&2
        exit 1
    fi
    for package in $({{ tools }} conventions packages); do
        name=$(basename "$package")
        echo "▸ $name public API vs {{ base }} (griffe check, breaking changes only)"
        {{ tools }} griffe check "$name" -a "{{ base }}" -s "$(dirname "$package")" || true
    done

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
    if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
        echo "not on main; checkout main first" >&2
        exit 1
    fi
    git fetch --quiet origin main
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
        echo "HEAD is not origin/main; push or pull first" >&2
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
