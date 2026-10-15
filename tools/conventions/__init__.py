"""The shared conventions of `CONVENTIONS.md`, checked over the whole repository.

Run `python -m conventions check` from anywhere inside the repository. Each finding is printed as
`path:line: CODE message`; `conventions explain CODE` gives the rule's reason and fix. The project's
own data is `pyproject.toml`'s `[tool.conventions]`.
"""
