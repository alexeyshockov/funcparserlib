# Conventions

Rules for code, tooling and releases in this repository. `CONTRIBUTING.md` covers how to contribute
and adds what is this project's own. Parts of the tooling are generated; propose changes to them in
an issue.

## Tooling

Tools live in three places, each with its own pins:

| Where | Examples | Version in force | Run via |
| --- | --- | --- | --- |
| Commit hooks | ruff, typos, prettier, actionlint, zizmor | `rev:` in `.pre-commit-config.yaml` | `just check`, `just fix` |
| The pinned tool environment, `tools/` (installed in `.venv-tools`) | mypy, ty, pyrefly, basedpyright, import-linter, vulture, griffe, the docs builder | `tools/uv.lock`, within `tools/pyproject.toml` | the `just` recipes |
| The project's `.venv` | deptry, slotscheck | `uv.lock` (the `lint` group) | the `just` recipes |

uv, just and prek are host tools, installed unpinned by `just doctor`. `just pins` prints every
version in force and what could move; `just tools-upgrade` moves `tools/uv.lock`. Every other tool
stays out of the dependency groups.

Run tools through the recipes, never directly: a check fails when the pinned version disagrees with
your code, whatever a local install says.

Dependency groups have fixed names and meanings, so a CI job can say which it
needs and nothing else arrives:

| Group | Holds | Synced by |
| --- | --- | --- |
| `tests` | the runner, its plugins, the second async backend, test-only libraries | through `dev` |
| `tests-<what>` | a test subset whose dependency has no wheel for some interpreter; its tests `importorskip` it | by name, in the noxfile sessions on interpreters that have the wheel |
| `lint` | the tools that run on the project's interpreter: deptry, slotscheck | by name, in the `gate` job |
| `dev` | `{ include-group = "tests" }` and each group the suite imports (`examples`, ...): what installs on every interpreter the tests run on | by default |

`dev` is uv's default group, which `uv run` installs everywhere, so it holds nothing that fails to
install on an interpreter a test job runs. The `gate` job, and a developer's environment, sync
`--all-groups`. A library the package's own code imports is an extra, because an extra is published
and a user installs it; a library only the tests or the tooling use is a group, because a group is
not.

A requirement is written `name[extras] >=X, <Y; marker`: one space after the name, none after an
operator, a comma and a space between clauses. By default a required dependency is a floor alone,
`>=X.Y`, so a new major that breaks nothing reaches users without a release here; extras and
dependency groups are `~=X.Y` at a version the lock has run, since a user can drop an extra and pin
the package directly. The default gives way case by case, with the reason beside the requirement: a
pre-1.0 dependency that changes between minors keeps its ceiling, an extra whose floor a test
session installs keeps `>=floor, <next major`. A distribution released from this repository is
pinned `==` its version (Versioning). `tools/pyproject.toml` keeps its own constraints, above.

The noxfile defines every environment the tests run in: the interpreter, the free-threaded build,
and any axis that is the project's own, such as a server version or a dependency's floor. It is
this project's to edit. Each session is tagged with the matrices it belongs to:

| Tag | Runs in | Selected by |
| --- | --- | --- |
| `local` | `just tests`, one session after another, and the end of `just gate` | the project: the cases worth a local run, at least the floor and the newest free-threaded build |
| `ci` | `.github/workflows/tests.yaml`, one job per session, called by `ci.yaml` | every environment CI tests |
| `windows` | a `ci` session's job on `windows-latest` instead of `ubuntu-latest` | |
| `optional` | a `ci` session's job that may fail without failing the run: a pre-release interpreter, a platform not yet claimed | |

Coverage is measured in one session, never added to the others.

A job aims to finish within five minutes, and a session that does not is a candidate for splitting;
`timeout-minutes: 15` on every job is the hard limit. The one
required status check is an aggregate job in `ci.yaml` that needs every other job, runs
`if: always()`, and fails unless each result is `success`. Two reasons: session names change with
the matrix, and GitHub reports a job skipped because a dependency failed as passed, so a required
job downstream of a failure would read green.

CI runs the recipes, never a second spelling of them. `just gate` is the list of gates a pull
request must pass, cheapest first, and the workflow's `gate` job runs each as its own named step
(CON700); the tests run as nox sessions, never as `pytest` spelled out in YAML. Tool versions reach
CI through `tools/uv.lock`, so a local run and a CI run use identical versions.

The rules here run as one command, `conventions`, which the pinned tool environment installs from
`tools/conventions/` and `just tools::conventions` calls in the gate. `conventions check` reads the
whole repository as git lists it, new files included, and prints `path:line: CODE message`;
`conventions explain CODE` gives a rule's reason and fix, and `conventions fix` applies the fixers
some codes have. `tools/conventions/` is generated: never edit it to make a finding
go away. Fix the code, or suppress the finding on its line with `conventions: ignore[CODE]` beside
the reason; a suppression that no longer suppresses anything is a finding itself. The rules' data
is `pyproject.toml`'s `[tool.conventions]`.

Repository-specific checks, if any, live in `tools/checks/<topic>_check.py` and run under
`just tools::conventions`.

A recipe either asserts or it writes, never both. `just check [FILES]` and `just gate` are read-only;
`just fix [FILES]` applies what they can only report, and any other recipe that writes says so in
its name.

Ruff's configuration is `shared.ruff.toml`, generated and reached from
`pyproject.toml` through `[tool.ruff] extend`. The project adds to it with `extend-select`,
`extend-ignore` and `extend-per-file-ignores`, each addition with its reason; it never writes the
bare `select` / `ignore` / `per-file-ignores` keys, which would replace the shared lists wholesale,
and the conventions runner refuses them (CON803) along with any `ruff.toml` that would shadow the
section (CON800). The code width is the one setting the project answers for itself.

## Types

- **The public API is fully typed, and `py.typed` ships.** Presence is gated by `just type-coverage`
  (`pyrefly coverage check --public-only`, at 100%): every public function, parameter, attribute and
  variable is annotated, except a constant assigned a literal. Correctness is gated by `just types` (mypy over the
  package, the examples and the tests, configured in
`mypy.ini`). ty, pyrefly and basedpyright run as second opinions and never block.
- **Each distribution's `typesafety/`, beside its package, pins what an end user's checker infers**
  from the public API, through `assert_type`. basedpyright gates it (`just types-api`), and mypy as
  package code (`just types`); a change in public inference is a behaviour change, even when runtime
  behaviour is identical.
- **Internal code** uses types where they aid readability; the gate is over the package, not a
  demand for annotations on every internal `def`.

**Which tree gets which depth.** A project that deviates writes the deviation down next to its
config.

| tree | mypy | ty | basedpyright | pyrefly | recipe |
| --- | --- | --- | --- | --- | --- |
| the package | **gate** | informational | informational | informational; **`coverage` at 100%** | `types`, `types-info`, `type-coverage` |
| `examples/` | **gate** | informational | — | — | `types`, `types-info` |
| `tests/` | **gate**, at the tests' depth | informational | — | — | `types`, `types-info` |
| each distribution's `typesafety/` | **gate** | informational | **gate** | informational | `types`, `types-api` |
| the `benchmarks/` directories | **gate**, at the tests' depth | — | — | — | `types-benchmarks` |

One checker gates the code a project owns, so a finding has one semantics to meet: mypy, and
`just check` runs it on the paths it is given. basedpyright gates what a user's checker infers,
as Pylance and the agents' editors run it. ty runs as a second opinion everywhere.

**Every checker runs at the floor**, the version `requires-python` starts from, except for a file
that ruff's `per-file-target-version` (`[tool.ruff]` in `pyproject.toml`) gives a version of its
own: the toolbox checks that file at that version, in a run of its own, with every checker. It is
for a file that shows a newer Python than the package needs, an example of a newer syntax say, and
the one place that version is stated. A test that runs the file skips it on an older interpreter.

Tests and benchmarks are read first, and mypy checks them for what they call on the package: a
section of `mypy.ini` turns off what would make one harder to read (annotations a fixture or helper
does not need, a helper returning `Any`, an assertion against what a type says a value cannot be,
`seen.append(v) or v`, a fake server's `match` without a default arm). A test that hands the
package a value of the wrong type on purpose says so with a `# type: ignore[code]` and the reason.
The typesafety suites are code at the package's level, not tests: they live beside the package and are
checked at its depth.

## Source

- **Every exported class is `@final`** unless it is a Protocol, an exception, or carved out in
  `pyproject.toml`'s `[tool.conventions.subclassable]` with the reason, which the class's docstring
  repeats as "Extensible by design". Sealed by default: a return annotation is exhaustive for a
  caller's checker only if its arms cannot be extended.
- **Visibility has three levels, and a name says which.** Public: what a public module's `__all__`
  exports. Package: what other modules of the package use, with no underscore. Module: everything
  else, with a leading underscore, so the call site says the name is this file's alone. `__all__`
  appears in public modules only, and in one every name is exported or underscored; a helper other
  modules need lives in a private module. It is one literal list (CON409), read from source by the
  runner and by pyrefly's public-API report, and the two count the same definitions (CON415);
  every name it lists is bound where the runner reads it (CON418).
- **A member has two levels.** On an exported class a member without an underscore is API, and one
  with an underscore is not: any module of the package may use it. Nothing marks a member as its
  class's alone, so no check that reads `obj._x` that way runs over the package: ruff's SLF rules
  are not selected, and the shared PyCharm profile (`.idea/`) runs its protected-member inspection
  on tests, examples and benchmarks only.
- **A class a public signature hands out is exported** (CON416): a caller annotates with the type
  of what it is handed or has to pass. A type alias, exported or not, is read through to the
  classes it names and need not be exported itself; a private base class is not a signature. A
  class users only get from a factory says so in its docstring, naming the factory: "Users get it
  from `connect()`; its constructor is internal." Its constructor's parameters are then not API,
  and the factory is a public function or method that returns the class, or, for a class users
  only read off another, a public attribute that holds it, named without a call: "Users get it
  from `Match.target`" (CON417).
- **An exception carries the name it is imported by.** A public module that exports an exception
  defined in a private module below it ends with `adopt_exceptions(__name__, __all__, globals())`
  from `_commons`, so a traceback, a class repr and a pickle name the public module (CON408). Other
  classes keep the module that defines them.
- **A package states its version once**, in its private `_version.py`: `__version__: str` read
  from `importlib.metadata.version`, `"0.0.0.dev0"` on `PackageNotFoundError` (a source checkout;
  a fallback that is not a PEP 440 version breaks whatever parses it). `__init__.py` re-exports it
  (`from ._version import __version__`, in `__all__`); the package's own modules import it from
  `_version`, never through the root (CON506), and nothing else reads the metadata (CON405-CON407).
- **Initialisms stay capitalized in identifiers**: `HTTP`, `API`, `RPC`, `ID`, `URL`, and the
  ones the project adds in `[tool.conventions] initialisms`. Provider-defined wire names keep their exact casing
  (`operationId`, `MessageId`), and an upstream project's own spelling survives in a name built
  from it.
- **Async naming: say what the async-ness belongs to.** The call itself, when it is the awaitable
  twin of a sync call that does something: the `a` prefix on the sync name (`aclose`, `aget`, as the
  stdlib's `anext`), never an `_async` suffix. An entity, what a call produces or a callable named for
  what it is: the `async_` prefix (`async_client`, `async_ok_handler`); classes are entities by
  construction (`AsyncRouter`). What the call runs, serves or takes: the adjective stays
  with that noun, sync and async spelled alike (Trio's `to_thread.run_sync`, `run_sync_side`,
  `is_async_callable`). A name from a protocol the package implements keeps the protocol's spelling
  (`handle_async_request`). A private helper or a test is free.
- **Two dataclass idioms, both from the package's `_commons`.** `@valueclass` for value types,
  compared and hashed by their contents. `@refclass` for identity objects, a class holding a
  client, a session or a cache, with the default identity `__eq__` / `__hash__`. Both are frozen for
  the type checkers alone, so constructing one costs a plain `__init__`.
- **A component's settings are one frozen dataclass of plain values**, one field per command-line
  option or configuration-file key: numbers, strings, booleans, `Literal` choices, addresses. Each
  field is documented once, as an attribute docstring, and validated in `__post_init__`, so a bad
  value fails before anything opens. Resources (an executor, a socket, a client) and code (a
  handler, a callback) are arguments beside it, never fields. A field takes a scope prefix only when
  its noun alone does not say what it belongs to.
- **An implementation is chosen by name while no user implements its seam**: a `Literal` field,
  resolved inside the package. An optional implementation is imported only once named, and fails
  with an `ImportError` naming the extra. The seam's Protocol stays private until a user has to
  implement it.
- **`<package>/_commons.py` is generated.** It holds `valueclass`, `refclass`, `with_hint` and
  `adopt_exceptions`, and parses on the oldest supported Python. The package's own helpers go in
  `<package>/_utils.py`, a private module like any other.
- **Ruff, `line-length = 100`, `pyupgrade.keep-runtime-typing = true`.** Wrap comments and
  docstrings to the same width. Docstrings follow the Google convention, kept minimal: prose where
  the name and the types do not already say it. Docstrings, comments and pages write inline code as
  Markdown does, `x`, never with reStructuredText's double backticks, and a code block as a fenced
  block with its language, never as reStructuredText's `::` literal block.
- **A module imports its own directory's modules relatively** (`from ._core import X`); an import
  from another directory is absolute.
- **Nothing in a package imports through its root.** `from pkg import X` inside `pkg` is an import
  cycle; import the module that defines `X`. A subpackage that builds on the package as a user
  would is listed in `[tool.conventions] consumers`, and imports it as a user does.
- **Another package's underscore module is never imported.**
- **Structural section headers** use exactly `# --- Sentence case title ---` in every file whose
  comment is `#`: source, `pyproject.toml`, justfiles, workflows, scripts. Bare decorative
  separators are not used. A title opening with someone else's lowercase identifier keeps
  it.
- **A `TYPE_CHECKING` block carries an adjacent rationale comment** naming the concrete import
  cycle, optional dependency, or static-conformance reason.
- **A time bound in a test is a claim about scheduling.** A sub-second bound is a named constant
  or carries a comment saying why it is safe at that size. A time-bounded negative assertion (`x
  is None` after a wait) is shown in the same test that the object can deliver, or says why the
  absence is the promise.
- **Exception and `assert` messages** have their own section below.

The conventions runner holds the mechanical rules of this section (CON1xx, CON2xx, CON4xx,
CON5xx); `conventions explain CODE` gives each one's reason and scope.

## Observability

- The core reports what it does through a seam of its own and imports no vendor SDK. Where the
  library has a middleware chain, the chain is the seam and the core exposes the facts an adapter
  needs on the request context. Elsewhere the seam is an observer protocol, recorded in an ADR of
  the repository.
- Each vendor is an adapter over the seam, an extra of the distribution it observes:
  `<package>.otel`, `<package>.sentry`. An adapter depends on the vendor's API package where one
  exists, never its SDK. A test spawns an interpreter and asserts that importing the core loads no
  vendor module.
- An observer protocol is one `__call__(event, /)` over a closed union of events; an adapter matches
  with a `case _` arm. An event is an identity object for one operation, not a value.
- An operation with a duration is a block: the observer returns a context manager the core enters
  before the work and leaves after it, with the exception the caller gets. A phase timed separately
  is a nested event. An event with nothing to wrap, such as one reported after ownership has passed
  on, is a plain call returning `None`.
- Failure and cancellation reach the block's exit as Python raises them: an `Exception` is a
  failure; any other `BaseException` is the operation not finishing, never an error. A result the
  caller gets only on success is a field the core sets before the block closes, read when the
  block exits without an exception; where success ends in more than one way, the field is a closed
  union.
- An `Exception` an observer raises goes to `sys.unraisablehook` and leaves the operation's outcome
  unchanged; a `BaseException` propagates. An observer's exit never suppresses the operation's
  exception. The test configuration fails on an unraisable exception (CON807), so an adapter's own
  bug fails its tests.
- With no observer the core allocates nothing per operation. Each seam has a micro-benchmark with no
  observer and with its adapters over a no-op vendor API.
- Events carry what debugging needs and never values that are data or secrets: query parameters,
  credentials. An adapter applies its vendor's own privacy controls.
- An adapter names its tracer, meter or SDK scope after the distribution and passes its version.
  Attribute names follow the vendor's conventions; others go under `<package>.*`.
- Logging: a `NullHandler` on the package's root logger, loggers named `<package>.<module>`, errors
  logged with `exc_info`. The library configures no handler.

## Benchmarks

- Benchmark code may reach into the package's internals; the package never imports it.
- It is type-checked at the tests level: `just types-benchmarks` runs mypy over it, blocking, in
  `just gate`, at the depth `mypy.ini` gives tests. An environment missing a benchmark's dependency
  shows up as an unresolved import — fix the environment (`just venv`), never the rule.
- Micro-benchmarks are pytest-benchmark cases, `bench_*.py` files in a `benchmarks/` directory
  beside a distribution's `tests/`, and run in-process with no server or network. A case whose
  timing includes threads or sockets is compared with others of the same run only.
- `just bench::micro [DIST] [ARGS]` runs them in the noxfile's sessions tagged `bench` under
  `benchmarks/pytest.ini`, which no test suite reads, and saves each run under
  `benchmarks/results/`, with the machine, the interpreter and the commit. The `bench` session
  runs on one interpreter each; the tag is on the newest by default, and a project whose benchmark
  dependencies have no wheel for it yet moves the tag in its noxfile. A figure is quoted with the
  interpreter it ran on. `just bench::compare [RUN ...]` compares saved runs, by default the two
  newest of one machine and interpreter.
- Macro-benchmarks run against a server or a driver, or drive a load generator. They are noisy
  and local: they stay out of CI and routine verification, and run for performance work.
- Compare results from the same machine and run, control background load, and rerun anomalies. A
  single absolute number is not a regression signal.
- Preserve CLI flags, filter semantics, output formats and the benchmark matrix unless the task
  changes them deliberately; then update the nearest benchmark README with it.

## Exception messages

Modelled on [PostgreSQL's Error Message Style Guide][pg-style], which splits a report into a primary
line, a detail and a hint. Python has the same three: the exception's message, and notes attached
with `add_note()` (PEP 678). A note prints under the traceback and stays out of `str(exc)`, so the
message can stay one line for a log while a person reading a traceback still gets everything.

[pg-style]: https://www.postgresql.org/docs/current/error-style-guide.html

### The message

One line, lowercase, no closing period, no semicolon, one fact. It is what a log line carries, so it
holds nothing a reader cannot afford to lose. It surfaces as `RuntimeError: deadlock happened`,
which is why a mid-line capital reads wrong.

- **The subject is something the reader can resolve.** Either a name from their own call
  (`max_workers must be greater than 0`, `ssl_context must be an instance of ssl.SSLContext, not
  {kind}`), or the single participant the operation has (`the server described no statement`). An
  article attached to an object the caller may hold several of — *this pool*, *this channel*, *this
  stream* — resolves nothing. Name the method, state the rule as a rule, or leave the fact to the
  exception class.
- **The message says what the class does not.** A class specific enough to be the whole fact
  (`PoolClosed`) takes no message at all; a generic one (`ValueError`, `InterfaceError`) carries it.
- **Quote an interpolated identifier or value, and name its kind**: `table "foo.bar"`, not
  `foo.bar`. An unquoted value that is empty or contains spaces reads as a missing word.
- **Tense is a claim.** Past tense says the attempt failed and could succeed later (`could not open
  ...`); present says it cannot work at all (`cannot open ...`).
- **Always give the reason.** If no reason is known, the code is wrong, not the message.
- **Never name the reporting function.** Say what the code was trying to do.
- **Active voice, never first person.** Drop the subject where it would be the program itself.
- **Words to avoid**: `unable` (say `cannot` or `could not`), `bad` and `illegal` (say `invalid`,
  and why), `unknown` (say `unrecognized`, and print the value), `may` (say `can` or `might`),
  contractions, and `non-negative` (say `greater than or equal to zero`). Spell words out:
  `authentication`, not `auth`.

A capital is right when the first word *is* an identifier, class name, or initialism. Protocol- and
RFC-mandated strings are not exception messages and keep their own casing.

### The notes

A note is a **detail** or a **hint**, and there is at most one of each. Both are complete sentences,
capitalized, ending in a period.

- A **detail** states a value or an implementation fact the message left out. It has to be safe to
  lose — anything a reader cannot do without belongs in the message instead.
- A **hint** states an action: `Call connection_pool() again for another pool.`
- A third note means the message is wrong. Rewrite the message rather than adding to it.

The mechanical half of this section is the conventions runner's CON6xx; what is the project's own
is `[tool.conventions.messages]`.

## Documentation

**`docs/` is published whole.** Zensical copies every file under `docs_dir` into the site and has no
exclusion mechanism, so the directory holds pages and the static assets they reference, and nothing
else: no guide, no build helper, no script. Tooling the build loads by path (the griffe extension)
lives in `tools/`; the rules for writing pages are this section.

- A page's home is its [Diátaxis](https://diataxis.fr/) role, and `zensical.toml`'s `nav` is the
  authority on the concrete tree. `reference/` is generated from docstrings and is hand-edited
  nowhere; `adr/` holds architectural decisions. The hand-written user guide is `modules/` where
  the public surface is a set of modules and `guide/` where it reads as one narrative; `recipes/`,
  `explanation/` and `design/` exist where the project has that material.
- Keep module READMEs concise and contributor-facing. Do not copy their full text into the site or
  make the README the canonical API guide.
- Source runnable Python snippets from the runnable examples with `--8<--` includes — `examples/`,
  or each package's own `examples/` in a multi-package repository. Do not duplicate example code in
  Markdown when it can be included.
- Highlight lines of an include with an `<!-- hl: section ... -->` comment above its fence, naming
  one or more nested sections of the example; `just docs-hl-fix` writes the fence's `hl_lines`.
- Update docs and examples together when public behavior changes. Ensure new pages are reachable
  from the existing navigation where appropriate.
- `docs/explanation/glossary.md` is the project's vocabulary: one word, one meaning. An entry is a
  line starting with a bold headword, `**term**` or `**`Name`**`, followed by its meaning and the
  names that implement it. Read it before naming something, use its terms in code, docs, commit
  messages and conversation, and add the entry in the same change that coins a term. The terms
  below are fixed; `glossary.md` repeats one only to name its implementations:
    - **engine**: a sans-IO protocol state machine for one connection: bytes in, events out, what is
      sent encoded back to bytes, no I/O. Its owner does the I/O around it.
    - **codec**: the encoding of one value or one message, such as a column's type, a packet or a
      body's compression. Not an engine.

The conventions runner's CON9xx holds the site to itself: highlights, includes, links, snippets
and the glossary. One is a warning, printed and never failing the run: CON917, where the API
reference renders a top-level name the module's `__all__` does not export, or leaves out one it
does. The reference may differ on purpose; read it and decide.

Verification: `just tools::conventions`, then `just docs-build`, which clears Zensical's
cache before building so generated API reference pages reflect current source. `just docs-hl-fix`
repairs highlight-line drift; inspect its changes.

## Specs and decisions

`openspec/specs/<capability>/spec.md` holds the current requirements and `docs/adr/` the decisions:
the spec says what holds, the ADR says why, and those two are what is kept. A change is worked
through [OpenSpec](https://openspec.dev) as a draft under `openspec/changes/` and is not kept once
archived; how a change is written, applied and archived is `openspec/config.yaml`. A guide states an
invariant as rule, enforcing gate and ADR link; it does not restate the ADR.

- **Read the requirements that already govern the paths you touch**, not only the ones a change
  deltas: grep `openspec/specs/` for the paths and verbs in scope. No gate checks that a spec still
  describes the code; this rule is the defence.
- **An ADR is cited as `ADR NNNN`.** A bare `ADR NNNN` is one in `docs/adr/`; a decision recorded
  outside this repository is cited by link to its published page every time it is mentioned.
  The conventions runner (CON3xx) resolves every bare citation against `docs/adr/` and holds a
  link's label to its target.
- Files `openspec update` generates (under `.claude/`, `.agents/`, `.pi/` and `.omp/`) are not
  edited by hand; workflow guidance goes in `openspec/config.yaml`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), with a scope naming the module or
distribution; recent history is the fastest way to internalise the pattern. Common types: `feat`,
`fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `build`, `style`.

- **A change is breaking only if it breaks a *released* version**, measured from the last published
  tag. Then it takes `!` after the scope (`feat(http)!: ...`) and a CHANGELOG entry under
  **Changed (BREAKING)**. Renaming something no release has shipped is an ordinary `refactor:` plus a
  CHANGELOG line.
- **A decision is recorded in an ADR, never only in a commit message**: a message cannot be
  corrected without rewriting history. A design-touching commit names the ADR it adds or changes.
- A commit message cites ADRs and specs, never an untracked file.
- **Never skip hooks** (`--no-verify`). If a hook fails, fix what it found.
- **Prefer a new commit over `--amend`**, unless you are correcting the very last commit you made
  and have not pushed.
- **Reference another commit as `abc1234 ("its subject line")`**, never a bare hash: a rebase
  changes the hash, the subject still finds it.
- Update `README.md` and `CHANGELOG.md` in the same commit as a user-visible API change.

## Pull requests

Before opening:

- [ ] `just check` is clean (the commit hooks enforce this too)
- [ ] `just types` and `just type-coverage` are clean, with no regression on public API you touched
- [ ] `just tests` passes, and you added or updated the smallest focused test for the change
- [ ] A CHANGELOG entry under `## [Unreleased]` (Added / Changed / Fixed / Removed, breaking
      changes called out)
- [ ] Public API changes are reflected in the README, `docs/`, and an example
- [ ] A design invariant you changed is updated in the guide that owns it, in the *same* commit
- [ ] Work you deliberately deferred is called out in the description, with why

Verification scales with risk. Start narrow and widen only as the change requires: `just check` on
the changed files, then the narrowest focused test (`just test PATH`), then the checks it touches (`just
conventions`, `just imports`, `just deps`, `just types-api` for a public inference change), then
broader unit checks, and only then integration or container checks.

The `master` branch is the only stable branch and will never be force-pushed. Any other branch,
including release branches, may be rebased or force-pushed at any time.

## Releases

A repository is one library and is released as one version: every distribution in it ships at
every release, pinned to the others exactly.
Releases are cut from `master` by a maintainer, and a release is a GitHub release: a draft opened by
`just release X.Y.Z`, reviewed on GitHub, and published by a person. Publishing creates the tag
`vX.Y.Z` and fires the publish workflow, which uploads the distributions to PyPI through
[trusted publishing](https://docs.pypi.org/trusted-publishers/) with Sigstore-signed PEP 740
attestations, so no API token is stored anywhere. A release commits nothing to `master`.

### Layout

- A repository with one distribution keeps it at the root: `pyproject.toml`, `<package>/`,
  `tests/`.
- A repository with several is a uv workspace whose root is not a distribution. Each distribution
  (a package this repository publishes) lives at `<package>/`, with its import package at
  `<package>/<package>/` and its tests at `<package>/tests/`. `[tool.uv.workspace] members` is the
  one list of distributions: the checks and test sessions read it on each run.
- Test fixtures several members share are a pytest plugin in the lowest member that needs them,
  public as `<package>.testing`.
- Each member is tested alone in an environment that holds it and its declared dependencies only
  (`uv sync --package <member>`), one noxfile session per member, and `just deps` runs deptry per
  member.

### Versioning

- Released versions: `X.Y.Z`.
- In the tree, always: `X.Y.Z.devN`, where `X.Y.Z` is the **target** of the next release. PEP 440
  sorts `X.Y.Z.dev0 < X.Y.Z < X.Y+1.0.dev0`.
- Every member's version is the same, and a member depending on another pins it with `==` at that
  version.
- The conventions runner holds all of it (CON410-CON414).
- Patch releases (`X.Y.Z+1`) are reserved for bug fixes that do not change the public API.

### Cutting a release

1. Run the pre-release report against the previous stable tag:

   ```shell
   just release-check
   ```

   It prints the current type-completeness score and what a caller of the public API at the latest
   stable tag (no pre-release suffix) meets now: each removal, each changed parameter (its kind,
   position, default and annotation), return, overload or annotation, a function made a coroutine
   function or the reverse, and an attribute no longer settable, read from the same surface the
   conventions rules check. Both are informational; read them and decide.

2. Land the CHANGELOG, if the project has one: rename the unreleased section to `## [X.Y.Z]`,
   dated, and finalise the entries. Everything the release ships is settled in the tree before it
   is tagged.

3. From a clean, pushed `master` checkout whose tree holds `X.Y.Z.devN`:

   ```shell
   just release X.Y.Z
   ```

   This checks the tree against the version, then creates a **draft** GitHub release targeting
   the current commit, pre-filled with that version's CHANGELOG section. It changes no file and
   pushes no commit.

4. Review the draft on GitHub and click **Publish**. Publishing creates the tag and fires
   `pypi-publish.yaml`, which runs the tests on the tagged commit, checks the tag against the
   tree (`conventions release vX.Y.Z`), rewrites every version and exact pin to `X.Y.Z` in its own
   checkout, builds every distribution and uploads them with attestations.

5. Set the next development version, whenever it is chosen:

   ```shell
   just version X.Y.Z+1.dev0     # or X.Y+1.0.dev0
   ```

   It writes the version into every member, every exact pin, relocks, and
   leaves the commit to you. The gate fails on every commit after a release until this lands.

### Hotfixes

Same flow with the patch component bumped: land the fix with the tree at `X.Y.Z+1.devN`, then
`just release X.Y.Z+1`.

### Yanking

A broken release is yanked, never deleted (pip skips a yanked version for new installs but keeps
it resolvable for pins). Yank from the project's release management page on PyPI and include a
short reason, then ship the fix as the next patch release.
