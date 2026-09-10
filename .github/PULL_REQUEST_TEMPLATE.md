## Description

<!-- What does this PR change, and why? -->

## PR title

<!--
PR titles become the squash-commit message and drive the release version
bump, so use a Conventional Commits prefix:
  fix:   patch release
  feat:  minor release
  feat!: or a BREAKING CHANGE footer -> major release
Other prefixes (docs:, chore:, refactor:, test:, ci:) do not trigger a release.
-->

## Checklist

- [ ] `uv run ruff check` and `uv run ruff format --check` pass
- [ ] `uv run pytest --cov=aind_motion_energy --cov-report=term-missing` passes
- [ ] Docs/docstrings updated if public behavior changed
- [ ] Linked to the relevant issue
