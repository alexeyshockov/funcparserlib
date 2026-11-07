# Examples guide

- Examples are runnable user-facing code and the source of snippets included in the
  documentation. Use public APIs only — an example that reaches for an underscore module
  is describing something the package has not offered yet.
- Keep each one small and focused on a single thing a reader came to learn. Show an async variant
  where it adds something, not for symmetry.
- Preserve `# --8<-- [start:name]` / `# --8<-- [end:name]` regions used by docs. For highlighted
  regions, use nested named sections rather than hand-maintained line numbers.
- When an example changes, search `docs/` for its include path and update the surrounding
  explanation if behaviour or output changed.
- Example-only dependencies belong in a development dependency group, not in the shipped package
  unless the package actually requires them.

Examples are gated, not decorative: mypy checks them in `just types`, at the depth `mypy.ini`'s
`[mypy-examples.*]` section gives them, and ty reads them as a second opinion. They are checked at
the floor. An example that needs a newer Python than the package does
states it once, as ruff's `per-file-target-version` in `pyproject.toml`
(`"examples/t_strings.py" = "py314"`), and every checker reads it from there.

```shell
just types
just check examples/<file>.py
```

If snippet highlighting drifts, run `just docs-hl-fix` and review the resulting documentation diff.
