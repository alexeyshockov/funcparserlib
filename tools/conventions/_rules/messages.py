"""CON6xx: what an exception message and its notes may say.

`CONVENTIONS.md` models this on PostgreSQL's Error Message Style Guide, which splits a report into a
primary line, a detail and a hint. Python has the same three since PEP 678: the message, and notes
attached with `add_note()`. A project's own data is `[tool.conventions.messages]`.
"""

import ast
import builtins
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from .._engine import Finding, rule
from .._project import Project, RunError

# Words PostgreSQL's guide rules out, with what to write instead.
AVOID = {
    "unable": "cannot / could not",
    "bad": "invalid, and why",
    "illegal": "invalid, and why",
    "unknown": "unrecognized, and print the value",
    "non-negative": "greater than or equal to zero",
    "nonnegative": "greater than or equal to zero",
}

# Contractions and abbreviations the guide asks to be spelled out. The lookarounds exclude a
# package whose owner spells it that way (`google-auth`), which a bare `\b` does not.
SPELL_OUT = re.compile(
    r"(?<![-.\w])(?:\w+n't|spec|stats|parens|auth|xact)(?![-.\w])", re.IGNORECASE
)

OPENING_ARTICLE = re.compile(r"^(?:the|this)\s+", re.IGNORECASE)
NOUN = re.compile(r"([a-z_]+(?:'s)?)\b", re.IGNORECASE)

# Nouns a caller can hold more than one of at a single call site.
SHARED_AMBIGUOUS = frozenset(
    {
        "batch",
        "channel",
        "cursor",
        "future",
        "pool",
        "portal",
        "reader",
        "registry",
        "resolver",
        "result",
        "scope",
        "slot",
        "stream",
        "subscription",
        "target",
        "template",
    }
)

# Nouns naming the one participant an operation has.
SHARED_SINGULAR = frozenset(
    {
        "address",
        "call",
        "cancellation",
        "connection",
        "deadline",
        "engine",
        "extended",
        "handshake",
        "instance",
        "last",
        "operation",
        "peer",
        "platform",
        "server",
        "session",
        "startup",
        "statement",
        "transaction",
        "worker",
    }
)

STOPWORDS = frozenset({"a", "an", "the", "this", "is", "was", "has", "have", "been", "already"})

# Every package's builder, generated into the first package's `_commons.py`.
SHARED_BUILDER = "with_hint"

# The period in `e.g.` and `i.e.` ends an abbreviation, not a sentence.
ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|etc|vs|cf|approx|no)\.", re.IGNORECASE)
SECOND_SENTENCE = re.compile(r"[.!?]\s+\S")

# A note may close on an interpolated value; the placeholder is stripped before it is judged.
TRAILING_VALUE = re.compile(r"(?:\{\})+$")

BUILTIN_EXCEPTIONS = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)


def judged(
    ambiguous: frozenset[str], singular: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """The shared sets with a project's own nouns moved into the set it names."""
    return (
        (SHARED_AMBIGUOUS - singular) | ambiguous,
        (SHARED_SINGULAR - ambiguous) | singular,
    )


class _Vocabulary:
    """A project's names and noun sets, from `[tool.conventions.messages]`."""

    def __init__(self, settings: Mapping[str, object]) -> None:
        self.known_names: tuple[str, ...] = tuple(_strings(settings, "known_names"))
        own_ambiguous = frozenset(_strings(settings, "ambiguous"))
        own_singular = frozenset(_strings(settings, "singular"))
        if both := sorted(own_ambiguous & own_singular):
            raise RunError(f"[tool.conventions.messages] names a noun in both lists: {both}")
        self.ambiguous, self.singular = judged(own_ambiguous, own_singular)
        self.builders: tuple[str, ...] = tuple(_strings(settings, "builders"))
        home = settings.get("builders_home")
        self.builders_home: str | None = home if isinstance(home, str) else None


def _strings(settings: Mapping[str, object], key: str) -> list[str]:
    value = settings.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RunError(f"[tool.conventions.messages] {key} is a list of strings")
    return value


def _vocabulary(project: Project) -> _Vocabulary:
    return project.memo(
        "messages.vocabulary", lambda: _Vocabulary(project.settings.get("messages", {}))
    )


def _sources(project: Project) -> Sequence[tuple[Path, ast.Module]]:
    """The shipped packages' modules, parsed; a message there reaches a user."""

    def parse() -> Sequence[tuple[Path, ast.Module]]:
        roots = list(project.package_roots.values())
        return tuple(
            (path, tree)
            for path in project.under(roots, frozenset({".py"}))
            if (tree := project.tree(path)) is not None
        )

    return project.memo("messages.sources", parse)


def _exception_names(project: Project) -> frozenset[str]:
    """The builtin exceptions and the packages' own, found by their bases."""

    def compute() -> frozenset[str]:
        names = set(BUILTIN_EXCEPTIONS)
        classes = [
            node
            for _, tree in _sources(project)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        grew = True
        while grew:
            grew = False
            for node in classes:
                bases = {ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases}
                if node.name not in names and bases & names:
                    names.add(node.name)
                    grew = True
        return frozenset(names)

    return project.memo("messages.exceptions", compute)


def text_of(node: ast.expr, constants: Mapping[str, str] | None = None) -> str | None:
    """The message with every interpolation collapsed to a single placeholder."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and constants is not None:
        return constants.get(node.id)
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(part.value) if isinstance(part, ast.Constant) else "{}" for part in node.values
        )
    return None


def name_of(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else ast.unparse(call.func)


def constants_of(tree: ast.AST) -> dict[str, str]:
    """Module-level names bound to a string, which a message can be passed by."""
    body = tree.body if isinstance(tree, ast.Module) else []
    return {
        target.id: node.value.value
        for node in body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def message_argument(call: ast.Call) -> ast.expr | None:
    """The message: second after a non-text first argument, as `OSError(errno, message)`."""
    if not call.args:
        return None
    first = call.args[0]
    if len(call.args) > 1 and not isinstance(first, ast.Constant | ast.JoinedStr):
        return call.args[1]
    return first


def messages_in(tree: ast.AST, exceptions: frozenset[str]) -> Iterator[tuple[int, str, str]]:
    """Every `(line, exception class, message)` a caller could read, wherever it is built."""
    constants = constants_of(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and name_of(node).rsplit(".", 1)[-1] in exceptions:
            if (source := message_argument(node)) and (text := text_of(source, constants)):
                yield node.lineno, name_of(node), text
        elif isinstance(node, ast.Assert) and node.msg is not None:
            text = text_of(node.msg)
            if text:
                yield node.lineno, "assert", text


def notes_in(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Every note, attached with `add_note()` or through `with_hint()`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = name_of(node)
        if name == "add_note" and node.args:
            source: ast.expr = node.args[0]
        elif name == SHARED_BUILDER and len(node.args) > 1:
            source = node.args[1]
        else:
            continue
        text = text_of(source)
        if text:
            yield node.lineno, text


def opens_with_a_name(text: str, known_names: Sequence[str]) -> bool:
    first = text.split(" ", 1)[0]
    if first.isupper() or "." in first or "(" in first:
        return True
    return any(text.startswith(name) for name in known_names)


def wrongly_capitalized(text: str, known_names: Sequence[str]) -> bool:
    return bool(text) and text[0].isupper() and not opens_with_a_name(text, known_names)


def ends_in_a_period(text: str) -> bool:
    return text.rstrip().endswith(".") and not text.rstrip().endswith("...")


def says_two_things(text: str) -> bool:
    return ";" in text or bool(SECOND_SENTENCE.search(ABBREVIATION.sub("", text)))


def _class_words(name: str) -> frozenset[str]:
    stem = name.rsplit(".", 1)[-1]
    return frozenset(word.lower() for word in re.findall(r"[A-Z][a-z]*|[a-z]+", stem))


def paraphrases_its_class(raised: str, text: str) -> bool:
    """Whether the message repeats only the class's own words; a value is never a paraphrase."""
    if raised == "assert" or "{}" in text:
        return False
    words = frozenset(re.findall(r"[a-z]+", text.lower())) - STOPWORDS
    return bool(words) and words <= _class_words(raised)


def article_noun(text: str, known_names: Sequence[str]) -> str | None:
    """The noun an opening article points at, lowercased, read past a known name."""
    article = OPENING_ARTICLE.match(text)
    if article is None:
        return None
    rest = text[article.end() :]
    for name in known_names:
        if rest.startswith(name + " "):
            rest = rest[len(name) + 1 :]
            break
    noun = NOUN.match(rest)
    return noun.group(1).removesuffix("'s").lower() if noun else None


def avoided_words(text: str) -> list[str]:
    lowered = text.lower()
    found = [f"{word} (say {instead})" for word, instead in AVOID.items() if word in lowered]
    found += [f"{hit} (spell it out)" for hit in SPELL_OUT.findall(text)]
    return found


def malformed_note(text: str, known_names: Sequence[str]) -> bool:
    stripped = TRAILING_VALUE.sub("", text.strip()).rstrip()
    if not stripped:
        return True
    opens_wrong = stripped[0].islower() and not opens_with_a_name(stripped, known_names)
    return opens_wrong or not stripped.endswith((".", "?"))


@rule("CON600", "CON601", "CON602", "CON603", "CON604", "CON605", "CON606")
def messages(project: Project) -> Iterator[Finding]:
    """An exception message is one lowercase fragment a reader can act on.

    CON600: A message opens lowercase unless its first word is a name.

    Reason: it surfaces as `RuntimeError: deadlock happened`, so a capital reads wrong. An
    initialism, a dotted or called identifier, or a name in `[tool.conventions.messages]
    known_names` may open it.

    Fix: lowercase the first word, or add the product name to `known_names`.

    CON601: A message does not end in a period.

    Reason: the primary line is a fragment on a log line, not a sentence in prose.

    Fix: drop the period. An ellipsis stays.

    CON602: A message says one thing.

    Reason: a second sentence is a hint and a semicolon joins two facts. Both belong in
    `add_note()`, which prints under the traceback and stays out of `str(exc)`.

    Fix: move the second fact to a note, or raise through `with_hint(error, note)`.

    CON603: A message says what its class does not.

    Reason: `PoolClosed: this pool is closed` spends a line repeating its own name. A class specific
    enough to be the whole fact takes no message; a message carrying a value never repeats it.

    Fix: drop the message, or say what the class cannot: the value, the cause.

    CON604: A message opening on an article names something the reader can resolve.

    Reason: `this pool is closed` names one of the pools a caller may hold; the traceback gives a
    frame and a line, never which object.

    Fix: name the method, state the rule as a rule, or leave the fact to the class. A noun a project
    holds one of moves to `[tool.conventions.messages] singular`.

    CON605: Every noun an article opens on is judged ambiguous or singular.

    Reason: a noun in neither set passes CON604 unread.

    Fix: add the noun to `[tool.conventions.messages] ambiguous` or `singular`.

    CON606: A message avoids the words PostgreSQL's guide rules out.

    Reason: each hides a fact: `unable` hides whether it is cannot or could not, `unknown` hides
    what was looked for. Contractions and abbreviations are spelled out.

    Fix: write the replacement the finding names.
    """
    vocabulary = _vocabulary(project)
    names = vocabulary.known_names
    exceptions = _exception_names(project)
    for path, tree in _sources(project):
        relative = project.relative(path)
        for line, raised, text in messages_in(tree, exceptions):
            if wrongly_capitalized(text, names):
                yield Finding(relative, line, "CON600", f"message opens with a capital: {text!r}")
            if ends_in_a_period(text):
                yield Finding(relative, line, "CON601", f"message ends in a period: {text!r}")
            if says_two_things(text):
                yield Finding(relative, line, "CON602", f"message says two things: {text!r}")
            if paraphrases_its_class(raised, text):
                yield Finding(
                    relative, line, "CON603", f"message repeats its class {raised}: {text!r}"
                )
            noun = article_noun(text, names)
            if noun in vocabulary.ambiguous:
                yield Finding(
                    relative, line, "CON604", f"article on a {noun} the reader cannot pick out"
                )
            elif noun is not None and noun not in vocabulary.singular:
                yield Finding(
                    relative, line, "CON605", f"noun {noun!r} is in neither set: {text!r}"
                )
            if hits := avoided_words(text):
                yield Finding(relative, line, "CON606", f"{text!r}: {', '.join(hits)}")


@rule("CON610")
def notes(project: Project) -> Iterator[Finding]:
    """A note is a complete sentence.

    Reason: the message is the one line a log carries; a note is what a person reading the traceback
    gets in addition, a detail or a hint. One that reads as a fragment has usually been cut from the
    wrong place.

    Fix: open it with a capital and end it with a period. It may close on an interpolated value.
    """
    names = _vocabulary(project).known_names
    for path, tree in _sources(project):
        for line, text in notes_in(tree):
            if malformed_note(text, names):
                yield Finding(
                    project.relative(path), line, "CON610", f"note is not a sentence: {text!r}"
                )


def _functions(project: Project, path: Path) -> set[str] | None:
    """The function names `path` defines, or `None` if it is missing or does not parse."""
    if not path.is_file() or (tree := project.tree(path)) is None:
        return None
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


@rule("CON620", "CON621", "CON622")
def builders(project: Project) -> Iterator[Finding]:
    """The exception builders a message is raised through exist and are used.

    CON620: `with_hint` is defined in the first package's `_commons.py`.

    Reason: the note scan reads every message raised through it; a rename or a removal takes them
    all out of the scan without a sound.

    Fix: restore `<package>/_commons.py` from the default branch.

    CON621: Every name in `[tool.conventions.messages] builders` is defined in `builders_home`.

    Reason: a renamed builder left in the list orphans it, and nothing else notices.

    Fix: correct the name, or set `builders_home` to the file that defines the builders.

    CON622: Every listed builder is raised through somewhere.

    Reason: a builder nothing raises through is dead or misspelt; either way the scans read fewer
    messages than they think.

    Fix: remove the name from `builders`, or correct it.
    """
    first_root = next(iter(project.package_roots.values()))
    commons = first_root / "_commons.py"
    if SHARED_BUILDER not in (_functions(project, commons) or set()):
        yield Finding(
            project.relative(commons), None, "CON620", f"{SHARED_BUILDER} is not defined here"
        )
    vocabulary = _vocabulary(project)
    pyproject = project.root / "pyproject.toml"
    line = project.line_of(pyproject, "builders")
    if vocabulary.builders_home is None:
        if vocabulary.builders:
            yield Finding(
                "pyproject.toml", line, "CON621", "builders are listed without a builders_home"
            )
    else:
        defined = _functions(project, project.root / vocabulary.builders_home) or set()
        if missing := sorted(set(vocabulary.builders) - defined):
            yield Finding(
                "pyproject.toml",
                line,
                "CON621",
                f"{vocabulary.builders_home} does not define {missing}",
            )
    raised = {
        name_of(node.exc)
        for _, tree in _sources(project)
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
    }
    if unused := sorted(set(vocabulary.builders) - raised):
        yield Finding("pyproject.toml", line, "CON622", f"nothing raises through {unused}")


@rule("CON630")
def read_something(project: Project) -> Iterator[Finding]:
    """Every package holds a message for the scans to read.

    Reason: the message rules assert an absence, and an absence over an empty tree says nothing. A
    package with no message at all is far likelier a scan that stopped reading.

    Fix: check `[tool.conventions] packages`. A package that truly raises nothing suppresses the
    finding on the first line of its `__init__.py`, where it is reported.
    """
    exceptions = _exception_names(project)
    for root in project.package_roots.values():
        found = any(
            True
            for path, tree in _sources(project)
            if path.is_relative_to(root)
            for _ in messages_in(tree, exceptions)
        )
        if not found:
            init = root / "__init__.py"
            location = (
                (project.relative(init), 1)
                if init.is_file()
                else (
                    project.relative(root),
                    None,
                )
            )
            yield Finding(*location, "CON630", "no exception message in this package")
