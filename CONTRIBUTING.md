# Contributing

## Branch strategy

`dev` is the default branch and the integration point for day-to-day work.
`main` tracks released code.

Both `dev` and `main` are protected: 1 required human approval, squash-merge
only, no direct pushes.

Flow: create a feature branch off `dev`, open a PR into `dev`. When `dev` is
ready to release, open a PR from `dev` into `main`.

## Commit / PR title conventions

PR titles (which become the squash-commit message) follow
[Conventional Commits](https://www.conventionalcommits.org/), since the
release workflow computes the version bump from them:

| Prefix | Effect |
|---|---|
| `fix:` | patch release |
| `feat:` | minor release |
| `feat!:` or a `BREAKING CHANGE` footer | major release |

Other prefixes (`docs:`, `chore:`, `refactor:`, `test:`, `ci:`) do not trigger
a release.

## Pull requests

- Keep PRs scoped to one change.
- CI (lint, type check, tests) must pass before merge.
- Squash-merge only — keep the individual commit history on the feature
  branch as messy as you like; the PR title is what matters.

## Documentation

The docs site is MkDocs + [mkdocstrings](https://mkdocstrings.github.io/); the
API reference is generated from the numpydoc docstrings in `src/`, so new public
functions appear on the site without touching `docs/`.

```bash
uv run mkdocs serve          # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict # what Read the Docs runs
```

`--strict` fails on broken internal links and docstring-parsing warnings; run it
before opening a PR that touches docstrings or `docs/`.
