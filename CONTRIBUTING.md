# Contributing

This file is the process. How code and configuration are written is `CONVENTIONS.md`; the map of
the code is `ARCHITECTURE.md`; what is this project's own (layout, extra prerequisites, working
rules) is `AGENTS.md`, written for people as much as for agents.

## Reporting issues

- **A bug**: the version, the Python version and platform, and the smallest code that shows it.
- **A feature**: the problem first, then the API you would like to write.
- **A vulnerability**: privately, as `SECURITY.md` says, never in a public issue.

## Development setup

You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).
Then:

```shell
just doctor                  # install the host tools and the commit hooks
just venv                    # build .venv with every dependency group and extra
```

Day to day:

```shell
just check path/to/file.py   # read-only verdict for one file: the commit hooks plus mypy
just fix path/to/file.py     # apply what `check` can only report
just test path/to/tests.py   # pytest in .venv
just tests                   # the local test matrix (the noxfile's sessions tagged `local`)
just gate                    # everything a pull request must pass
just docs-build              # build the site
just --list                  # every recipe
```

## Pull requests

Pull requests are squash-merged, so the commits inside one are yours to shape; the title and the
description are what stays. Before opening:

- [ ] `just gate` passes, and the smallest focused test for the change is added or updated
- [ ] A user-visible change has a `CHANGELOG.md` entry, and the README, `docs/` and an example
      reflect it
- [ ] A design decision is recorded in an ADR (`docs/adr/`), and an invariant you changed is
      updated in the guide that owns it
- [ ] Work you deliberately deferred is named in the description, with why

Verify from narrow to wide: `just check` on the changed files, the focused test, the checks the
change touches (`just conventions`, `just imports`, `just types-api` for public inference), then
`just gate`. `just types-info` gives second opinions that never block: fix what is cheap, name the
rest.

A change is breaking only if it breaks a released version; say so in the title (`!`) and in the
changelog entry.

`master` is the only stable branch and is never force-pushed. Any other branch may be rebased at any
time.

## Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions
follow [semantic versioning](https://semver.org/spec/v2.0.0.html). An entry goes in the topmost,
unreleased section, under Added, Changed, Fixed or Removed, with a breaking change called out.

## Releasing

For maintainers. A release is a GitHub release cut from `master`; publishing it creates the tag
`vX.Y.Z` and uploads every distribution to PyPI through trusted publishing. It commits nothing.

The tree always holds `X.Y.Z.devN`, where `X.Y.Z` is the next release; every distribution has the
same version and pins the others with `==`. A patch release is for fixes that do not change the
public API.

1. `just release-check` reports what changed for a caller since the last stable tag. Read it.
2. Name the changelog's unreleased section `## [X.Y.Z]`, dated, and push.
3. `just release X.Y.Z` checks the tree and opens a draft GitHub release with that section.
4. Review the draft on GitHub and publish it.
5. `just version X.Y.Z+1.dev0` (or `X.Y+1.0.dev0`) sets the next development version; commit it.
   The gate fails until it lands.

A broken release is yanked on PyPI, never deleted, and fixed by the next patch release.
