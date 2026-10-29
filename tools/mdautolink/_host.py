"""What the extension reads from the site generator running it: the URL of the page being rendered.

Zensical is the one host read here; under any other, every page is unscoped.
"""

# Generated; propose changes in an issue rather than editing this file.


def current_page_url() -> str | None:
    try:
        from zensical.extensions.autorefs import get_autorefs_store
    except ImportError:
        return None
    page = get_autorefs_store().current_page
    return None if page is None else page.url
