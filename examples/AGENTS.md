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

Examples are gated, not decorative: mypy reads them through `mypy.ini`'s file list and ty blocks on
them.

```shell
just types-examples
just types
```

If snippet highlighting drifts, run `just docs-hl-fix` and review the resulting documentation diff.
