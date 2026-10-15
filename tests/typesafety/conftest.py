"""Keep pytest away from this directory.

These files are type-checked, never executed (`just types-api`): they assert what a downstream type
checker infers from the public API.
"""

collect_ignore_glob = ["*.py"]
