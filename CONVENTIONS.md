# Conventions

How code and configuration are written in this repository, where a tool cannot say it for you.
`CONTRIBUTING.md` is the process (setup, pull requests, releases); `ARCHITECTURE.md` is the map;
`AGENTS.md` holds what is this project's own. Part of the tooling, this file included, is rendered
from [python-project-template](https://github.com/alexeyshockov/python-project-template), whose
README lists the files it owns: propose a change to one of them in an issue.

Most rules below are checked by `just conventions`, which prints `path:line: CODE message`;
`conventions explain CODE` gives a rule's reason, fix and scope, so a code beside a rule stands for
its details. Fix the code, or suppress a finding on its line with `conventions: ignore[CODE]` beside
the reason. `tools/conventions/` is generated and never edited; the rules' data is
`[tool.conventions]` in `pyproject.toml`, and a check of this repository's own is
`tools/checks/<topic>_check.py`.

## Tooling

- **Run tools through the `just` recipes, never directly.** Versions are pinned in three places:
  commit hooks by `rev:` in `.pre-commit-config.yaml`; the tools that read the project from outside
  (type checkers, import-linter, vulture, the docs builder) in `tools/uv.lock`; the two that run on
  the project's interpreter (deptry, slotscheck) in the `lint` group. No other tool enters a
  dependency group. `just pins` prints every version in force.
- **A recipe either asserts or it writes.** `just check [FILES]` and `just gate` are read-only;
  `just fix [FILES]` applies what they report, and any other recipe that writes says so in its name.
- **CI runs the recipes, never a second spelling of them.** The workflow's `gate` job runs each step
  of `just gate` as its own step (CON700), and the tests run as nox sessions. Every job has
  `timeout-minutes: 15` and aims at five. The one required status check is the aggregate job that
  needs every other job.
- **Ruff is configured in `shared.ruff.toml`**; the project adds with `extend-select`,
  `extend-ignore` and `extend-per-file-ignores`, each with its reason (CON800, CON803).
- **A check of this repository's own** is a pytest file that imports pytest and the standard
  library only and reads the repository rather than importing the package. It asserts nothing a
  `CON` code holds: a shared rule that needs this project's data takes it from `[tool.conventions]`,
  and one that would hold in a sibling project belongs in the shared runner. A check that asserts
  an absence carries a companion test showing it finds a planted case.

Dependency groups have fixed names:

| Group | Holds |
| --- | --- |
| `tests` | the runner, its plugins, the second async backend, test-only libraries |
| `tests-<what>` | a test subset whose dependency has no wheel for some interpreter; its tests `importorskip` it |
| `lint` | deptry, slotscheck |
| `dev` | `{ include-group = "tests" }` and each group the suite imports; uv's default, so it holds nothing that fails to install on an interpreter a test job runs |

A library the package's own code imports is an extra; one only the tests or the tooling use is a
group.

A requirement is written `name[extras] >=X, <Y; marker`. A required dependency is a floor alone,
`>=X.Y`; extras and dependency groups are `~=X.Y` at a version the lock has run. An exception
carries its reason beside the requirement. A distribution released from this repository is pinned
`==` its version.

The noxfile defines every environment the tests run in and is this project's to edit. Each session
is tagged: `local` (run by `just tests`: at least the floor and the newest free-threaded build),
`ci` (one job each in `tests.yaml`), `windows` (a `ci` session on `windows-latest`), `optional` (a
`ci` session that may fail).

Coverage is measured in the noxfile's `coverage` session, never added to the others, and run by
`just coverage`: branch coverage of every distribution, held at `fail_under = 85` in
`[tool.coverage.report]`, a regression floor rather than a target.

## Layout

- A repository is one library, released as one version (CON410-CON414).
- One distribution sits at the root: `pyproject.toml`, `<package>/`, `tests/`.
- Several are a uv workspace whose root is not a distribution: each at `<package>/`, its import
  package at `<package>/<package>/`, its tests at `<package>/tests/`, each tested alone with its
  declared dependencies only.
- Test fixtures several members share are a pytest plugin in the lowest member that needs them,
  public as `<package>.testing`.
- Where a part goes is read from signs: a part users write against and choose for itself is a
  repository; one that keeps another client's API working, or is the library's batteries, stays in
  the library's repository; a part whose users face the wrapped stack's API, or whose dependencies
  weigh on the rest, is a distribution, and otherwise a module, behind an extra when it needs a
  third-party client. One distribution is the starting shape.

## Types

| Recipe | Target | Checkers | Blocks |
| --- | --- | --- | --- |
| `types` | the package, `examples/`, `tests/`, `benchmarks/` | mypy | yes |
| `types-api` | each distribution's `typing_tests/` | mypy, ty, pyrefly, basedpyright | yes |
| `type-coverage` | the public API's annotations | pyrefly `coverage`, at 100% | yes |
| `tools::types` | `tools/checks/` | mypy | yes |
| `types-info` | the package | ty, pyrefly, basedpyright | no |

- **The public API is fully typed, and `py.typed` ships.** Internal code is typed where it aids
  reading.
- **`typing_tests/` pins what a user's checker infers**: `assert_type` for what a call returns, one
  ignore per checker, with its code, on each call the API must reject. A change in public inference
  is a behaviour change.
- **Every checker runs at the floor** of `requires-python`, except for a file that ruff's
  `per-file-target-version` gives a newer version.
- A test that hands the package a wrong type on purpose says so with `# type: ignore[code]` and the
  reason.

## Source

- **Every exported class is `@final`** unless it is a Protocol, an exception, or listed in
  `[tool.conventions.subclassable]` with the reason, which its docstring repeats as "Extensible by
  design" (CON2xx).
- **A name says its visibility.** Public: what a public module's `__all__` exports. Package: no
  underscore, used by other modules. Module: leading underscore. `__all__` is one literal list, in
  public modules only, and every other name there is underscored (CON402-CON404, CON409).
- **A member without an underscore on an exported class is API**; one with an underscore is open to
  the whole package.
- **A class a public signature hands out is exported** (CON416). One users only get from a factory
  says so in its docstring: "Users get it from `connect()`; its constructor is internal" (CON417).
- **An exception carries the name it is imported by**: a public module re-exporting one from a
  private module ends with `adopt_exceptions(__name__, __all__, globals())` (CON408).
- **The version is stated once**, in `_version.py`, read from `importlib.metadata` with
  `"0.0.0.dev0"` as the fallback; `__init__.py` re-exports it (CON405-CON407).
- **Initialisms stay capitalized**: `HTTP`, `API`, `RPC`, `ID`, `URL`, plus `[tool.conventions]
  initialisms` (CON100). A wire name or an upstream project's spelling keeps its own casing.
- **Two dataclass idioms, from the generated `<package>/_commons.py`**: `@valueclass` for values
  compared by content, `@refclass` for identity objects that hold a client, a session or a cache
  (CON502, CON503). The package's own helpers go in `_utils.py`.
- **A component's settings are one frozen dataclass of plain values**, one field per option, each
  documented as an attribute docstring (never an `Attributes:` section) and validated in
  `__post_init__`. Resources and callbacks are arguments beside it, never fields.
- **An implementation is chosen by name while no user implements its seam**: a `Literal` field,
  imported only once named, failing with an `ImportError` that names the extra. The seam's Protocol
  stays private until a user has to implement it.
- **Imports**: relative within a directory (CON501), absolute across directories; never through
  the package's own root (CON506); never another package's underscore module (CON509).
- **A `TYPE_CHECKING` block carries a comment** naming the import cycle, optional dependency or
  conformance reason (CON500).
- **Docstrings follow the Google convention**, written where the name and the types do not already
  say it, at `line-length = 100`. Docstrings, comments and pages are Markdown, never
  reStructuredText (CON505).
- **Section headers** are exactly `# --- Sentence case title ---`
  (CON504).
- **A time bound in a test is a claim about scheduling.** A sub-second bound is a named constant or
  says why it is safe. A time-bounded negative assertion is shown, in the same test, that the object
  can deliver (CON51x).

## Benchmarks

- Benchmark code may reach into the package's internals; the package never imports it. It is
  type-checked with the tests.
- Micro-benchmarks are pytest-benchmark cases in `bench_*.py` files under a `benchmarks/` directory
  beside a distribution's `tests/`, in-process, with no server or network; `just bench::micro` runs
  them and `just bench::compare` compares saved runs.
- Macro-benchmarks (a server, a driver, a load generator) stay out of CI and routine verification.
- A figure is quoted with its interpreter and compared only within one machine and run. A single
  absolute number is not a regression signal.

## Exception messages

Modelled on [PostgreSQL's Error Message Style Guide][pg-style]: the exception's message is its
primary line, and notes attached with `add_note()` are the detail and the hint. The mechanical half
is CON6xx; a project's own data is `[tool.conventions.messages]`.

[pg-style]: https://www.postgresql.org/docs/current/error-style-guide.html

- **The message is one line**: lowercase, no closing period, no semicolon, one fact. A capital is
  right when the first word is an identifier, a class name or an initialism.
- **The subject is something the reader can resolve**: a name from their own call (`max_workers must
  be greater than 0`) or the operation's single participant (`the server described no statement`),
  never *this pool* or *this channel*.
- **The message says what the class does not.** A class specific enough to be the whole fact
  (`PoolClosed`) takes no message.
- **Quote an interpolated value and name its kind**: `table "foo.bar"`.
- **Tense is a claim**: `could not open` failed this time, `cannot open` can never work.
- **Always give the reason**, and say what the code was trying to do, never the reporting function.
- **A note is a detail or a hint**, at most one of each, a complete sentence. A detail is safe to
  lose; a hint states an action. A third note means the message is wrong.

## Documentation

- **`docs/` is published whole**: it holds pages and the assets they reference, nothing else. Build
  tooling lives in `tools/`.
- `zensical.toml`'s `nav` is the authority on the tree. `reference/` is generated from docstrings
  and never hand-edited; `adr/` holds the decisions. A hand-written page's home is its
  [Diátaxis](https://diataxis.fr/) role: `modules/` where the public surface is a set of modules,
  `guide/` where it reads as one narrative, and `recipes/`, `explanation/` and `design/` where the
  project has that material.
- **A module README is short and contributor-facing**: what the module is, a minimal example, its
  internal layout and gotchas. The site is the user's reference; a README is never copied into it
  and never grows into a guide.
- Runnable snippets are included from the examples with `--8<--`, never copied into Markdown. An
  `<!-- hl: section ... -->` comment above a fence names the lines to highlight; `just docs-hl-fix`
  writes them.
- `docs/explanation/glossary.md` is the project's vocabulary: one word, one meaning, each entry a
  line starting with a bold headword. Use its terms, and add the entry in the change that coins one.
- `just conventions` (CON9xx) and `just docs-build` verify the site.

## Architecture

`ARCHITECTURE.md` holds, in order: what the system is, the codemap, the main path, the invariants
and the cross-cutting concerns.

- **An architectural invariant is about the shape**: a boundary, the direction a dependency points,
  what a part never knows, the concurrency model. It is written as the rule, the gate that enforces
  it and the ADR that says why; the ADR is cited, never restated. A rule for working inside one part
  lives in the nearest `AGENTS.md`.
- **Name code as it is written, in backticks**; every path named is held to the repository
  (CON311, CON312). Diagrams are ```` ```mermaid ```` fences, and the prose beside one names the
  same parts.
- **A concept a static diagram cannot carry**, such as a state machine stepping through its input or
  two tasks interleaving, may get an interactive explainer: one self-contained HTML file, with no
  network access and no build step, opened from a file as well as from a server.
  It lives beside the explanation page it illustrates, which links it and stays the source of
  truth; the site publishes both.
  No gate reads an explainer; review it with the change that alters what it shows.

## Specs and decisions

`openspec/specs/<capability>/spec.md` holds the current requirements and `docs/adr/` the decisions:
the spec says what holds, the ADR says why. A change is an [OpenSpec](https://openspec.dev) draft
under `openspec/changes/`, not kept once archived; `openspec/config.yaml` says how one is written.

- **Read the requirements that govern the paths you touch**: grep `openspec/specs/` for them. No
  gate checks that a spec still describes the code.
- **An ADR is cited as `ADR NNNN`** (CON3xx); a decision recorded outside this repository is cited
  by link.
- **A decision is recorded in an ADR**, never only in a commit message or a pull request.
- Files `openspec update` generates are not edited by hand.
