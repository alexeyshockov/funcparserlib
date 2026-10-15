"""The installed distribution's version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("funcparserlib")
except PackageNotFoundError:
    # A valid PEP 440 version, not a bare "dev": `Version("dev")` raises, so anything
    # parsing `__version__` breaks on a source checkout.
    __version__ = "0.0.0.dev0"
