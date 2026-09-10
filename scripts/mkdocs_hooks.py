"""MkDocs build hooks.

``mkdocs-literate-nav`` reads ``reference/SUMMARY.md`` — written during the
build by ``gen_ref_pages.py`` — to lay out the API reference section, and then
only marks it as "not in nav", so it would still be rendered as a stray page and
indexed by search. It is nav data, not documentation, so drop it from the build
entirely once literate-nav has consumed it.
"""

from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files, InclusionLevel

NAV_FILE = "reference/SUMMARY.md"


@event_priority(-200)  # after literate-nav's on_files, which runs at -100
def on_files(files: Files, config: object) -> Files:
    """Exclude the generated literate-nav summary from the built site."""
    nav_file = files.get_file_from_path(NAV_FILE)
    if nav_file is not None:
        nav_file.inclusion = InclusionLevel.EXCLUDED
    return files
