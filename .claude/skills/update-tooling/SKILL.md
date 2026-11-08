---
name: update-tooling
description: Bring the repo's toolchain up to date — project deps, commit hooks, dev-tool pins, GitHub Actions — and triage the fallout. Use when the user asks to update/refresh tooling or dependencies, or when a version drift is suspected.
---

# Updating the toolchain

<!-- Generated; propose changes in an issue rather than editing this file.
     What is specific to this repository lives in PROJECT.md beside it, which is not generated. -->

There is no `just update`: driving the channels means deciding what a bump broke and what to hold
back, which is judgement, not a recipe. This skill is that judgement, and it drives the mechanical
steps itself.

Read `PROJECT.md` beside this file first. It names the channels, gates and hazards this repository
has beyond the shared ones below, and it is the authority where the two disagree.

## The shared channels

| Channel | Pinned in | Bumped by |
| --- | --- | --- |
| GitHub Actions | `uses:` refs in `.github/workflows/` | this skill, step 4; Dependabot too, monthly, where the repository has a GitHub remote |
| Linters/formatters | `rev:` in `.pre-commit-config.yaml` | maintained upstream; do not bump them here; `just frozen-revs` checks them |
| Project deps (deptry, slotscheck in `lint`) | `uv.lock` | `just venv-upgrade` |
| Dev tools | `tools/uv.lock`, constrained by `tools/pyproject.toml` | `just tools-upgrade` — `just pins` only *reports* |

Three invariants that must survive any update:

- **A tool is in the toolbox unless it has to run on the project's interpreter.** Tools live in
  `tools/pyproject.toml` and run from `.venv-tools`. The exceptions are deptry and slotscheck, which
  read the interpreter they run on and so sit in the project's `lint` group. Never "fix" any other
  tool problem by adding it to a dependency group.
- **No bare tool binaries in recipes.** `ruff` goes through `prek run`; the toolbox through
  `uv run --project tools`, deptry and slotscheck through `uv run --no-sync` in `.venv`. A bare
  `ruff`/`ty`/`vulture`/`griffe` in the justfile or a workflow is the drift bug returning — the
  local binary silently diverges from the locked one.
- **A recipe either asserts or it writes, never both.** The commit hooks are read-only at the
  `pre-commit` stage; each fixing tool's write-mode twin lives at `manual`, which only `just fix`
  invokes. After a hook bump, confirm the gate half still cannot write: a hook can bake a fix flag
  into its `entry:` (prettier does, so `args: [--check]` leaves `--write` in place) or declare its
  own `stages:` and ignore `default_stages` (`typos` and `check-added-large-files` do). Either way
  a gate that rewrites the file it is judging reports Passed on work it silently changed.

## Procedure

1. Confirm a clean tree (`git status`), then `just venv-upgrade` and `just pins`. Hook revs in
   `.pre-commit-config.yaml` are maintained upstream; do not bump them here; `just frozen-revs` checks them.
2. Check `just frozen-revs` passed. Some revs are pinned on purpose: `shared_frozen_revs` in
   `shared.just` for the shared hooks, `frozen_revs` in the justfile for any of the project's own,
   each with its reason commented beside its hook in `.pre-commit-config.yaml`. The shared record and
   the revs arrive together through `copier update`, so a failure means a rev moved in this
   repository; restore it from the `master` branch.
3. `just tools-upgrade` moves `tools/uv.lock` within the constraints
   `tools/pyproject.toml` states. When several tools move at once and the fallout needs to be
   attributable, take them one at a time with `uv lock --project tools --upgrade-package <tool>`.
   A tool that must stay below a release gets a ceiling in `tools/pyproject.toml` with the reason
   beside it — an ordinary requirement, not a second mechanism.
4. Bump every `uses:` ref in the workflows this repository owns to the action's latest release
   (`gh api repos/<owner>/<repo>/releases/latest`, or its tags where it publishes no releases).
   Leave the workflows the template owns, `tests.yaml` and `pypi-publish.yaml`: they arrive through `copier update`.
   Keep each ref's form: a tag stays a tag at the same precision, a commit SHA becomes the new tag's
   commit with the tag in the comment beside it, and a branch ref (`release/v1`) stays as it is.
   Read the release notes of every major and check each input the workflow passes still means the
   same. Do this whether or not Dependabot is configured: it runs only on a repository with a
   GitHub remote, and a ref already at its latest release is left unchanged.
5. Re-run the gates and triage, plus whatever `PROJECT.md` adds:
   - `just check` — new ruff rules, new typo/markdownlint/editorconfig findings, actionlint and
     zizmor over the bumped workflows
   - `just types-all` and `just types-api` — new diagnostics per checker
   - `just imports deps slots` — the structural gates; a ruff or deptry bump changes what they see
   - `just tests` — dependency regressions

   Remember the tiers: `just gate` is what blocks — mypy over the package, the examples, the tests
   and the checks (`types`, `types-checks`), pyrefly coverage on API completeness (`type-coverage`),
   basedpyright on each `typesafety/` (`types-api`). The `-`-prefixed lines (`types-info`, the
   non-basedpyright checkers in `types-api`) are informational on purpose — don't promote them to gating to "fix" a diagnostic.
6. For each new finding, decide: fix the code, `noqa`/ignore with a reason, or configure the rule
   in `pyproject.toml`. Prefer fixing. A blanket ignore needs a comment saying why.
   Read every hunk a hook *rewrote* — `git diff` after `just fix`, don't just trust a green run.
   An auto-fixer applying a "correction" to a third-party identifier can produce working-looking
   code that silently binds the wrong name.
7. Read the changelog for anything with a major bump, or anything whose diagnostics changed
   shape — don't infer intent from the diff alone.
8. Commit. Deps/hooks/pins/actions in one commit is fine; **code fixed as fallout goes in its own commit**,
   so a revert of the bump doesn't drag source changes with it. Include in the message which
   version moved and what it forced.

## Things that are not this skill's job

- Setting the project's own version, or anything in `CHANGELOG.md` — that's `just version` and the release.
- Host tools (`uv`, `just`, `prek`): unpinned by design,
  `brew upgrade` / `just doctor`. Nothing gates on them. Everything that does gate is in
  `tools/uv.lock`, which is row 4.
- Widening or narrowing dependency floors in `pyproject.toml`. That's a compatibility decision,
  not an update — raise it separately.

`PROJECT.md` lists the rest.
