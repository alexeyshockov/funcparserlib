# The docs recipes.
# Generated; propose changes in an issue rather than editing this file.
# Imported by shared.just, whose `tools` they use.

# --- Docs ---

# mkdocstrings reads the package's source through griffe, from its `paths` option (the repository
# root by default), and imports nothing.
[doc("Build docs to ./site")]
docs-build:
    # Zensical's .cache does not invalidate on Python source changes, so the
    # mkdocstrings API reference can go stale. Clear it for a correct build.
    rm -rf .cache
    {{ tools }} zensical build

[doc("Serve docs with live reload (doesn't clear .cache — API ref may be stale; use docs-build for a clean build)")]
docs-serve:
    {{ tools }} zensical serve

[doc("Rewrite drifted hl_lines on --8<-- doc includes in place (CON902)")]
docs-hl-fix:
    {{ tools }} conventions fix CON902
