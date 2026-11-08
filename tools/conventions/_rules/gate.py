"""CON7xx: the gate is one list, and CI names the same one."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from .._engine import Finding, rule
from .._project import Project
from ._ast import line_at

WORKFLOW = ".github/workflows/ci.yaml"
MATRIX_WORKFLOW = ".github/workflows/tests.yaml"

# The gate's test recipe. CI runs the tests as the matrix of `MATRIX_WORKFLOW`, not as a step of the
# gate job.
TESTS = "tests"

# A recipe a step may run without `gate` depending on it: `types-info` swallows its own exit codes,
# so it is a report rather than a gate. A project adds its own in `[tool.conventions]
# informational`.
INFORMATIONAL = frozenset({"types-info"})

# Fewer than this many on either side means the scan has stopped finding what it reads.
FLOOR = 5

JOB = re.compile(r"^  [a-z][a-z0-9-]*:$", re.MULTILINE)
# A recipe, or a module's recipe as `module::recipe`.
STEP = re.compile(r"^\s+(?:- )?run: just ([a-z][a-z0-9-]*(?:::[a-z][a-z0-9-]*)*)$", re.MULTILINE)
CALLS_MATRIX = re.compile(rf"^\s+uses: \./{re.escape(MATRIX_WORKFLOW)}(?:\s+#.*)?$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class _Listed:
    name: str
    line: int


def _recipes_file(project: Project) -> Path | None:
    """`shared.just` when it defines `gate`, else the `justfile`; `None` when neither does."""
    for name in ("shared.just", "justfile"):
        path = project.root / name
        if path.is_file() and _gate_line(project, path) is not None:
            return path
    return None


def _gate_line(project: Project, path: Path) -> int | None:
    for number, line in enumerate((project.text(path) or "").splitlines(), 1):
        if line.startswith("gate:"):
            return number
    return None


def _gate_dependencies(project: Project, path: Path) -> list[_Listed]:
    """The recipes `just gate` depends on, in order, each on the `gate:` line."""
    number = _gate_line(project, path)
    assert number is not None
    line = (project.text(path) or "").splitlines()[number - 1]
    return [_Listed(name, number) for name in line.removeprefix("gate:").split()]


def _job_steps(text: str, job: str) -> list[_Listed] | None:
    """The recipes one named workflow job runs, in order; `None` when there is no such job.

    Read by a line scan: the runner has no YAML dependency.
    """
    header = f"\n  {job}:\n"
    if header not in text:
        return None
    start = text.index(header) + len(header)
    # To the next job, whatever it is called: naming the following job would let a job inserted
    # between the two silently join this one.
    following = JOB.search(text, start)
    end = len(text) if following is None else following.start()
    return [_Listed(m.group(1), line_at(text, m.start())) for m in STEP.finditer(text, start, end)]


@rule("CON700", "CON701", "CON702", "CON703", "CON704")
def gate(project: Project) -> Iterator[Finding]:
    """`just gate` and the CI workflow's `gate` job name the same recipes in the same order.

    CON700: CI runs every recipe the gate depends on.

    Reason: a recipe `gate` depends on that the workflow's `gate` job does not run never runs on a
    pull request.

    Fix: add a `run: just <recipe>` step to the `gate` job of `.github/workflows/ci.yaml`.

    CON701: CI's gate job runs nothing the gate does not.

    Reason: a step the recipe lacks makes a local `just gate` weaker than a pull request.

    Fix: add the recipe to `gate`, or name it in `[tool.conventions] informational` if it cannot
    gate.

    CON702: CI runs the gate's recipes in the gate's order.

    Reason: order is the gate's fail-fastest property, and it is easy to lose.

    Fix: reorder the workflow's steps to match `gate:`.

    CON703: The gate and the workflow job exist and the scan finds them.

    Reason: every rule above is satisfied by two empty lists, so a missing recipe, workflow or job,
    or a scan that finds almost nothing, would pass them silently.

    Fix: define `gate:` in `shared.just` or the `justfile`, and a `gate` job in
    `.github/workflows/ci.yaml` running each recipe as `run: just <recipe>`.

    CON704: CI runs the tests when the gate does.

    Reason: the gate's `tests` recipe runs the noxfile's local sessions, and CI runs the tests as
    the matrix of `.github/workflows/tests.yaml` instead of a gate-job step. A `ci.yaml` that does
    not call that workflow runs no test on a pull request, and the rules above cannot see it.

    Fix: add a job to `.github/workflows/ci.yaml` with `uses: ./.github/workflows/tests.yaml`.
    """
    recipes = _recipes_file(project)
    workflow = project.root / WORKFLOW
    if recipes is None:
        yield Finding("justfile", None, "CON703", "no `gate:` recipe in shared.just or justfile")
        return
    if not workflow.is_file():
        yield Finding(WORKFLOW, None, "CON703", "no CI workflow to hold the gate job")
        return
    recipes_path = project.relative(recipes)
    text = project.text(workflow) or ""
    steps = _job_steps(text, "gate")
    if steps is None:
        yield Finding(WORKFLOW, None, "CON703", "the workflow has no `gate` job")
        return
    dependencies = _gate_dependencies(project, recipes)
    expected = [dep for dep in dependencies if dep.name != TESTS]
    tests = next((dep for dep in dependencies if dep.name == TESTS), None)
    if tests is not None and not CALLS_MATRIX.search(text):
        yield Finding(
            recipes_path,
            tests.line,
            "CON704",
            f"`gate` depends on `{TESTS}` and `{WORKFLOW}` does not call `{MATRIX_WORKFLOW}`",
        )
    if len(expected) <= FLOOR:
        yield Finding(
            recipes_path,
            expected[0].line if expected else _gate_line(project, recipes),
            "CON703",
            f"the scan found {len(expected)} gate dependencies; it has stopped finding them",
        )
    if len(steps) <= FLOOR:
        yield Finding(
            WORKFLOW,
            text.count("\n", 0, text.index("\n  gate:\n")) + 2,
            "CON703",
            f"the scan found {len(steps)} steps in the gate job; it has stopped finding them",
        )
    informational = INFORMATIONAL | set(project.settings.get("informational", []))
    actual = [step for step in steps if step.name not in informational]
    actual_names = [step.name for step in actual]
    expected_names = [dep.name for dep in expected]
    for dep in expected:
        if dep.name not in actual_names:
            yield Finding(
                recipes_path,
                dep.line,
                "CON700",
                f"`gate` depends on `{dep.name}` and the CI gate job does not run it",
            )
    for step in actual:
        if step.name == TESTS and tests is not None:
            yield Finding(
                WORKFLOW,
                step.line,
                "CON701",
                f"the CI gate job runs `{TESTS}`, which CI runs as the `{MATRIX_WORKFLOW}` matrix",
            )
        elif step.name not in expected_names:
            yield Finding(
                WORKFLOW,
                step.line,
                "CON701",
                f"the CI gate job runs `{step.name}` and `gate` does not depend on it",
            )
    # Order over the recipes both lists hold: a missing or extra one is reported above.
    shared_expected = [name for name in expected_names if name in actual_names]
    shared_actual = [step for step in actual if step.name in expected_names]
    # A step run twice makes the two lengths differ; the first divergence is reported either way.
    for step, wanted in zip_longest(shared_actual, shared_expected):
        if step is None or step.name != wanted:
            yield Finding(
                WORKFLOW,
                step.line if step is not None else shared_actual[-1].line,
                "CON702",
                f"the CI gate job runs `{step.name if step else 'nothing'}` where `gate` runs "
                f"`{wanted or 'nothing'}`",
            )
            break
