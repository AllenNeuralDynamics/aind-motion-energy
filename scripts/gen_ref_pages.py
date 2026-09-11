"""Generate the API reference pages from the package source.

Run by ``mkdocs-gen-files`` during the MkDocs build: every public module under
``src/`` becomes a virtual ``reference/<module>.md`` page holding a single
mkdocstrings identifier, so the rendered reference always tracks the code and
its numpydoc docstrings rather than a hand-maintained page list.
"""

from pathlib import Path

import mkdocs_gen_files

REFERENCE_DIR = Path("reference")

nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "src"

for path in sorted(src.rglob("*.py")):
    parts = tuple(path.relative_to(src).with_suffix("").parts)

    if parts[-1] == "__init__":
        # The package itself lands on the reference index page.
        parts = parts[:-1]
        doc_parts = parts[1:] + ("index",)
    elif parts[-1].startswith("_"):
        continue
    else:
        # Drop the top-level package name from the URL; it is already the
        # section title, and repeating it makes every reference URL longer.
        doc_parts = parts[1:]

    doc_path = Path(*doc_parts).with_suffix(".md")
    full_doc_path = REFERENCE_DIR / doc_path
    nav[(".".join(parts),)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {'.'.join(parts)}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open(REFERENCE_DIR / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
