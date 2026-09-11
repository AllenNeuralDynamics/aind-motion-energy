# Contributing

## Branch strategy

`dev` is the default branch and the integration point for day-to-day work.
`main` tracks released code.

Both `dev` and `main` are protected: 1 required human approval, no direct
pushes. `dev` additionally requires linear history, which in this repo (no
rebase-merge enabled) means squash-merge only — the AIND standard's actual
requirement ("Feature branches ... must be Squash and Merged back into
`dev`"). `main` does not require linear history.

Flow: create a feature branch off `dev`, open a PR into `dev`, squash-merge.
When `dev` is ready to release, open a PR from `dev` into `main` and merge it
with a **merge commit** (not squash).

**Why the `dev` → `main` PR is the one exception:** the AIND standard only
specifies squash-merge for feature branches into `dev`; it says nothing about
promoting `dev` into `main`. Squashing that promotion collapses it to a
single-parent commit with no history link back to `dev`, so the next
promotion's conflict check falls back to whatever `dev` and `main` last
shared — commonly nothing more recent than when `dev` was created. Any file
both branches touch since (`CITATION.cff` on every release, for example)
then shows as a false add/add conflict, even when the change is a trivial,
non-conflicting one-liner. A real merge commit keeps that shared-history
link intact, so this only has to be solved once instead of on every release.

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
- Feature branches into `dev`: squash-merge only — keep the individual
  commit history on the feature branch as messy as you like; the PR title
  is what matters.
- `dev` into `main`: merge commit, not squash — see "Branch strategy" above.
